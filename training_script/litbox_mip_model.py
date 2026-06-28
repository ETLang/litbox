import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_to_identity(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.dirac_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class MicroFilter(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='reflect'):
        super(MicroFilter, self).__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Middle convolution has dilation of 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        initialize_to_identity(self)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        return self.final_relu(out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode):
        super(ResidualBlock, self).__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels))
        initialize_to_identity(self)
        self.final = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.primary(x)
        out += residual
        return self.final(out)

class LitboxMipDenoiserNet(nn.Module):
    def __init__(self, num_mips=5, input_channels=8, micro_features=16, unet_features=16, padding_mode='reflect'):
        super(LitboxMipDenoiserNet, self).__init__()
        self.num_mips = num_mips
        
        self.micro_filter = MicroFilter(input_channels, micro_features, padding_mode)
        
        self.unet_encoders = nn.ModuleList()
        self.unet_downsamplers = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()
        self.unet_skipconns = nn.ModuleList()
        
        pipeline_channels = micro_features
        for i in range(num_mips - 1):
            self.unet_encoders.append(ResidualBlock(pipeline_channels, unet_features, padding_mode))
            self.unet_downsamplers.append(nn.MaxPool2d(2))
            pipeline_channels = unet_features + micro_features
            
        self.bottleneck = ResidualBlock(pipeline_channels, unet_features, padding_mode)
        pipeline_channels = unet_features
        
        for i in range(num_mips - 1):
            self.unet_decoders.append(nn.Sequential(
                nn.Conv2d(pipeline_channels, unet_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            ))
            self.unet_skipconns.append(ResidualBlock(unet_features + unet_features, unet_features, padding_mode))
            pipeline_channels = unet_features
            
        self.conv_out = nn.Conv2d(pipeline_channels, num_mips, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, *mip_inputs):
        micro_features = [self.micro_filter(mip) for mip in mip_inputs]
        unet_skip_sources = []
        pipeline_state = micro_features[0]
        
        for i in range(self.num_mips - 1):
            pipeline_state = self.unet_encoders[i](pipeline_state)
            unet_skip_sources.append(pipeline_state)
            pipeline_state = self.unet_downsamplers[i](pipeline_state)
            pipeline_state = torch.cat([pipeline_state, micro_features[i + 1]], dim=1)
            
        pipeline_state = self.bottleneck(pipeline_state)
        
        for i in range(self.num_mips - 1):
            pipeline_state = self.unet_decoders[i](pipeline_state)
            skip_feature = unet_skip_sources[self.num_mips - 2 - i]
            pipeline_state = torch.cat([pipeline_state, skip_feature], dim=1)
            pipeline_state = self.unet_skipconns[i](pipeline_state)
            
        logits = self.conv_out(pipeline_state)
        if torch.onnx.is_in_onnx_export():
            return logits
        return self.softmax(logits)

    def export_onnx(self, path, input_channels=8, resolution=256):
        self.eval()
        dummy_inputs = []
        res = resolution
        for i in range(self.num_mips):
            dummy_inputs.append(torch.randn(1, input_channels, res, res).to(next(self.parameters()).device))
            res //= 2
            
        input_names = [f'input_mip{i}' for i in range(self.num_mips)]
        dynamic_axes = {name: {0: 'batch_size', 2: 'height', 3: 'width'} for name in input_names}
        dynamic_axes['output'] = {0: 'batch_size', 2: 'height', 3: 'width'}
        
        torch.onnx.export(self, 
                         tuple(dummy_inputs), 
                         path,
                         input_names=input_names, 
                         output_names=['output'],
                         dynamic_axes=dynamic_axes,
                         opset_version=11)