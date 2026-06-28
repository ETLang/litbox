import torch
import torch.nn as nn
import argparse

def initialize_to_identity(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # Sets weights so output matches input (center pixel = 1)
            nn.init.dirac_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            # Ensure normalization doesn't shift the identity
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode):
        super(ResidualBlock, self).__init__()
        # Optional: 1x1 conv if in_channels != out_channels for shortcut
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
        out += residual # Add residual connection
        out = self.final(out)
        return out

class LitboxDenoiserNet(nn.Module):
    def __init__(self,
                 upsample_factor, 
                 use_sigmoid=False, 
                 use_log_space=True, 
                 normalize_input=True, 
                 initial_features=16, 
                 unet_size=3, 
                 epsilon=1e-6,
                 padding_mode='reflect'):
        super(LitboxDenoiserNet, self).__init__()        
        self.normalize_input = normalize_input
        self.unet_size = unet_size
        self.previous_range = -1
        self.use_sigmoid = use_sigmoid
        self.use_log_space = use_log_space
        self.upsample_factor = upsample_factor
        self.epsilon = epsilon
        self.padding_mode = padding_mode
        self.unet_encoders = nn.ModuleList()
        self.unet_downsamplers = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()
        self.unet_skipconns = nn.ModuleList()
        pipeline_channels = 8

        # TODO: Incorporate a scalar representation of SPP into the model
        
        #########################
        # Initial Feature Extraction
        self.conv_in, pipeline_channels = self.make_feature_extraction(pipeline_channels, initial_features)

        #########################
        # Encoder (Downsampling path)
        for i in range(unet_size):
            next_encoder, pipeline_channels = self.make_encoder(pipeline_channels)
            self.unet_encoders.append(next_encoder)
            self.unet_downsamplers.append(nn.MaxPool2d(2))

        ##########################
        # Bottleneck
        self.bottleneck, pipeline_channels = self.make_bottleneck(pipeline_channels)

        self.short_circuit = nn.Conv2d(pipeline_channels, 3, kernel_size=3, padding=1)

        ##########################
        # Decoder (Upsampling path) - Using PixelShuffle for efficient upsampling
        # The number of input channels to PixelShuffle must be factor^2 * C_out
        for i in range(self.unet_size):
            next_decoder, pipeline_channels =  self.make_decoder(pipeline_channels)
            self.unet_decoders.append(next_decoder)
            self.unet_skipconns.append(self.make_skip_connector(pipeline_channels))

        # Final Convolution
        self.conv_out = nn.Conv2d(pipeline_channels, 3, kernel_size=3, padding=1)
        
        if use_sigmoid:
            self.clamp_output = nn.Sigmoid() # Or adjust based on your image range [0,1] or [-1,1]

    def pre_transform(self, x): # [B, 1, H, W]
        if self.use_log_space:
            x = torch.log2(torch.clamp(x, min=0.0) + self.epsilon)
        if self.normalize_input:
            if self.previous_range != -1:
                raise Exception('Cannot pre_transform without first matching a post_transform call for the previous call to pre_transform')
            # compute mean and std for normalization
            self.mean = x.mean(dim=[2, 3], keepdim=True)
            self.std = x.std(dim=[2, 3], keepdim=True)
            self.previous_range = 1

            # Normalize input mean to 0 and std to 1
            x = (x - self.mean) / (self.std + self.epsilon)
            # x = torch.cat([x, x_log], dim=1)  # Concatenate log and original for dual input
        return x
        
    def forward(self, x_lr, mask=None):
        # x_lr: low-resolution image with potential gaps (e.g., [B, 3, H, W])
        # mask: binary mask, 1 for valid pixels, 0 for gaps (e.g., [B, 1, H, W])

        # If you're incorporating partial convolutions, you'd need to pass the mask through
        # your convolution layers. For this example, let's assume standard convs.

        # Initial feature extraction
        f_in = self.conv_in(x_lr) # [B, 64, H, W]

        # Encoder
        unet_skip_sources = []
        pipeline_state = f_in
        for i in range(self.unet_size):
            pipeline_state = self.unet_encoders[i](pipeline_state)
            unet_skip_sources.append(pipeline_state)
            pipeline_state = self.unet_downsamplers[i](pipeline_state)
        # f_enc1 = self.encoder1(f_in) # [B, 128, H, W]
        # p_enc1 = self.pool1(f_enc1) # [B, 128, H/2, W/2]

        # f_enc2 = self.encoder2(p_enc1) # [B, 256, H/2, W/2]
        # p_enc2 = self.pool2(f_enc2) # [B, 256, H/4, W/4]

        # Bottleneck
        pipeline_state = self.bottleneck(pipeline_state) # [B, 512, H/4, W/4]
        short_circuit_output = pipeline_state

        # Decoder
        # Upsample 1 (from H/4 to H/2)
        for i in range(self.unet_size):
            pipeline_state = self.unet_decoders[i](pipeline_state)
            pipeline_state = torch.cat([pipeline_state, unet_skip_sources[self.unet_size - 1 - i]], dim=1)
            pipeline_state = self.unet_skipconns[i](pipeline_state)

        output = self.conv_out(pipeline_state)

        if self.use_sigmoid:
            return self.clamp_output(output) # Apply sigmoid to ensure [0,1] output if needed
        else:
            return output

    def post_transform(self, x):
        if self.normalize_input:
            if self.previous_range == -1:
                raise Exception('Cannot post_transform without a prior call to pre_transform')
            x *= (self.std + self.epsilon)
            x += self.mean
            self.previous_range = -1
        if self.use_log_space:
            x = torch.exp2(x) - self.epsilon
        return x
    
    def make_feature_extraction(self, channels_in, features):
        channels_out = features
        module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, padding_mode=self.padding_mode),
            nn.ReLU(inplace=True),
            ResidualBlock(channels_out, channels_out, self.padding_mode)
        )
        return module, channels_out
    
    def make_encoder(self, channels_in):
        channels_out = channels_in # * 2  (Evidence suggests each layer only uses 32 channels no matter how deep)
        module = ResidualBlock(channels_in, channels_out, self.padding_mode)
        return module, channels_out
    
    def make_bottleneck(self, channels_in):
        channels_out = channels_in # * 2  (Evidence suggests each layer only uses 32 channels no matter how deep)
        module = ResidualBlock(channels_in, channels_out, self.padding_mode)
        return module, channels_out
    
    def make_decoder(self, channels_in):
        channels_out = channels_in # // 2  (Evidence suggests each layer only uses 32 channels no matter how deep)
        module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out * (2*2), kernel_size=3, padding=1), # Output for 2x upsample
            nn.PixelShuffle(2), # Upsamples features by 2x
        )
        return module, channels_out
    
    def make_skip_connector(self, channels):
        return ResidualBlock(2 * channels, channels, self.padding_mode)

    def export_onnx(self, path, input_channels=8, resolution=256):
        self.eval()
        dummy_input = torch.randn(1, input_channels, resolution, resolution).to(next(self.parameters()).device)
        torch.onnx.export(self, 
                         dummy_input, 
                         path,
                         input_names=['input'], 
                         output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                     'output': {0: 'batch_size', 2: 'height', 3: 'width'}},
                         opset_version=11)
    
def main():
    parser = argparse.ArgumentParser(description='Convert Litbox PTH checkpoint to ONNX')
    parser.add_argument('--model-path', help='Path to the .pth model file')
    parser.add_argument('--training-folder', help='Path to the folder of the training output')
    parser.add_argument('--output-path', required=True, help='Path to save the .onnx file')
    parser.add_argument('--upsample', type=int, default=1, help='Upsample factor used in the model')
    parser.add_argument('--unet-size', type=int, default=5, help='U-Net depth (unet_size)')
    parser.add_argument('--features', type=int, default=32, help='Initial features count')
    parser.add_argument('--channels', type=int, default=8, help='Number of input channels (radiance, variance, albedo, density)')
    parser.add_argument('--resolution', type=int, default=256, help='Dummy input resolution for export')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LitboxDenoiserNet(
        upsample_factor=args.upsample,
        initial_features=args.features,
        unet_size=args.unet_size
    ).to(device)
    
    print(f"Loading weights from {args.model_path}...")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    print(f"Exporting to {args.output_path}...")
    model.export_onnx(args.output_path, input_channels=args.channels, resolution=args.resolution)
    print("Done.")

if __name__ == "__main__":
    main()
