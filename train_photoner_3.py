import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import numpy as np
import os
import glob
import time
import argparse
from PIL import Image
import OpenEXR
import Imath
import array
import math
from typing import Tuple, Optional
import torch.nn.functional as F
import torchvision.transforms.functional

# Settings (overridable via command line arguments)
g_output_upsample = 1 # 4
g_checkpoint_interval = 300
g_test_ratio = 0.0
g_epochs = 100
g_crop_size = 256
g_batch_size = 4
g_learn_rate = 0.00001 # 0.001

# Settings (internal)
g_unet_size = 5
g_padding_mode = 'reflect'
g_initial_features = 16
g_normalize_input = True
g_use_adam_w = True
g_weight_decay = 0.01
g_epsilon = 1e-6  # For log space transformation

# TODO
g_gaussian_initialization = True


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Photoner Neural Network Training Script')
   # parser.add_argument('--help', action='help', help='Displays documentation regarding these arguments')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--input-location', required=True, help='Path to input images')
    parser.add_argument('--training-location', help='Path to training images')
    parser.add_argument('--output-folder', help='Output folder for evaluated images')
    parser.add_argument('--model-path', required=True, help='Path to save/load model')
    parser.add_argument('--checkpoint-interval', type=int, default=g_checkpoint_interval, help='Seconds between checkpoints')
    parser.add_argument('--checkpoint-folder', help='Folder to save checkpoint data')
    parser.add_argument('--checkpoint-tests', help='Path to test images for checkpoints')
    parser.add_argument('--test-ratio', type=float, default=g_test_ratio, help='Percentage of data for testing')
    parser.add_argument('--epochs', type=int, default=g_epochs, help='Number of epochs to train')
    parser.add_argument('--log-space', action='store_true', help='Transform EXR data to log space')
    parser.add_argument('--crop-size', type=int, default=g_crop_size, help='Resolution of training crops')
    parser.add_argument('--upsample', type=int, default=g_output_upsample, choices=[1, 2, 4, 8], help='Upsampling factor')
    parser.add_argument('--onnx-export', help='Path to export ONNX model')
    parser.add_argument('--batch-size', type=int, default=g_batch_size, help='Batch size for training and testing') 
    parser.add_argument('--learn-rate', type=float, default=g_learn_rate, help='Learning rate') 
    
    args = parser.parse_args()
    
    # Validation
    if not args.eval and not args.training_location:
        parser.error("--training-location is required in training mode")
    if args.eval and not args.output_folder:
        parser.error("--output-folder is required in eval mode")
    if args.checkpoint_interval and not args.checkpoint_folder:
        parser.error("--checkpoint-folder is required when using --checkpoint-interval")
        
    return args

class PhotonerDataset(Dataset):
    def __init__(self, input_paths: list, training_paths: Optional[list] = None, 
                 crop_size: int = 64, upsample: int = 1,
                 truth_transform=None):
        self.input_paths = input_paths
        self.training_paths = training_paths
        self.crop_size = crop_size
        self.upsample = upsample
        self.truth_transform = truth_transform

        if len(input_paths) > 0:
            test_path = self.input_paths[0]
            
            # Check for EXR files or something else (SRGB-based)
            if test_path.lower().endswith('.exr'):
                self.exr_source = True
            else:
                self.exr_source = False
        
    def __len__(self):
        return len(self.input_paths)
    
    def load_exr(self, path: str) -> torch.Tensor:
        file = OpenEXR.InputFile(path)
        dw = file.header()['dataWindow']
        size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
        
        # Read all channels
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = []
        for channel in ['R', 'G', 'B']:
            str_ch = file.channel(channel, FLOAT)
            ch = np.array(array.array('f', str_ch))
            ch = ch.reshape(size)
            channels.append(ch)
        
        # Stack channels and convert to tensor
        img = np.stack(channels, axis=0)
        tensor = torch.from_numpy(img).float()
        
        return tensor
    
    def load_srgb(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        tensor = transforms.ToTensor()(img)
        # Convert from sRGB to linear space
        tensor = tensor.pow(2.2)
        return tensor
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_path = self.input_paths[idx]
        
        # Load input image
        if input_path.lower().endswith('.exr'):
            input_tensor = self.load_exr(input_path)
        else:
            input_tensor = self.load_srgb(input_path)
            
        # Random crop if in training mode
        if self.training_paths is not None:
            h, w = input_tensor.shape[1:]
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"Image {input_path} is smaller than crop size {self.crop_size}")
            top = torch.randint(0, h - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, w - self.crop_size + 1, (1,)).item()
            input_tensor = input_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            
            # Load and crop training image
            training_path = self.training_paths[idx]
            if training_path.lower().endswith('.exr'):
                target_tensor = self.load_exr(training_path)
            else:
                target_tensor = self.load_srgb(training_path)
                
            target_size = self.crop_size * self.upsample
            target_top = top * self.upsample
            target_left = left * self.upsample
            if target_tensor.shape[1] < target_size or target_tensor.shape[2] < target_size:
                raise ValueError(f"Target image {training_path} is smaller than required crop size {target_size}")
            target_tensor = target_tensor[:, 
                                       target_top:target_top+target_size,
                                       target_left:target_left+target_size]
            
            if self.truth_transform:
                target_tensor = self.truth_transform(target_tensor)
            
            angles = [0, 90, 180, 270]
            chosen_angle = random.choice(angles)
            input_tensor = torchvision.transforms.functional.rotate(input_tensor, chosen_angle)
            target_tensor = torchvision.transforms.functional.rotate(target_tensor, chosen_angle)
            
            return input_tensor, target_tensor
        
        return input_tensor, None

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # Optional: 1x1 conv if in_channels != out_channels for shortcut
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode=g_padding_mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode=g_padding_mode),
            nn.BatchNorm2d(out_channels))
        
        self.final = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.primary(x)
        out += residual # Add residual connection
        out = self.final(out)
        return out

# Define Perceptual Lossadfg
class VGGPerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False, feature_layers=['relu3_3', 'relu4_3']):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pre-trained VGG16 model
        vgg16_features =models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device)
        
        # Mapping from common VGG layer names to their nn.Sequential indices
        # These indices are specific to torchvision.models.vgg16.features
        self.vgg_layer_indices = {
            'relu1_1': 0,  # conv1_1
            'relu1_2': 2,  # conv1_2
            'relu2_1': 5,  # conv2_1
            'relu2_2': 7,  # conv2_2
            'relu3_1': 10, # conv3_1
            'relu3_2': 12, # conv3_2
            'relu3_3': 14, # conv3_3
            'relu4_1': 17, # conv4_1
            'relu4_2': 19, # conv4_2
            'relu4_3': 21, # conv4_3
            'relu5_1': 24, # conv5_1
            'relu5_2': 26, # conv5_2
            'relu5_3': 28  # conv5_3
        }

        # Select only the VGG layers needed for feature extraction
        self.feature_layers_to_extract = sorted([self.vgg_layer_indices[name] for name in feature_layers])
        
        # Build the VGG model up to the maximum required layer index
        self.vgg_model = nn.Sequential()
        for i, layer in enumerate(vgg16_features):
            self.vgg_model.add_module(str(i), layer)
            if i >= max(self.feature_layers_to_extract):
                break
        
        # Freeze VGG parameters to prevent them from being updated during training
        if not requires_grad:
            for param in self.vgg_model.parameters():
                param.requires_grad = False
        
        # Set VGG to evaluation mode
        self.vgg_model.eval()

        # ImageNet normalization parameters (VGG expects 0-1 then normalized)
        # You might need to adjust this if your model outputs are in a different range (e.g., -1 to 1)
        # and adjust the input to VGG accordingly.
        self.normalize_input = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])

    def forward(self, input_image, target_image):
        # Normalize images for VGG input
        # Only normalize when using PNGs. Not EXR        
        # normalized_input = self.normalize_input(input_image)
        # normalized_target = self.normalize_input(target_image)
        normalized_input = input_image
        normalized_target = target_image

        features_input = []
        features_target = []
        
        x_in = normalized_input
        x_target = normalized_target
        
        for i, layer in enumerate(self.vgg_model):
            x_in = layer(x_in)
            x_target = layer(x_target)
            
            if i in self.feature_layers_to_extract:
                features_input.append(x_in)
                features_target.append(x_target)
        
        perceptual_loss = 0
        for f_in, f_target in zip(features_input, features_target):
            # Use MSE loss between feature maps
            perceptual_loss += F.mse_loss(f_in, f_target)
        
        return perceptual_loss

class PhotonerNet(nn.Module):
    def __init__(self, upsample_factor, use_sigmoid=False, use_log_space=True):
        super(PhotonerNet, self).__init__()
        self.previous_range = -1
        self.use_sigmoid = use_sigmoid
        self.use_log_space = use_log_space
        self.upsample_factor = upsample_factor
        self.unet_encoders = nn.ModuleList()
        self.unet_downsamplers = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()
        self.unet_skipconns = nn.ModuleList()
        pipeline_channels = 1
        
        #########################
        # Initial Feature Extraction
        self.conv_in, pipeline_channels = self.make_feature_extraction(pipeline_channels, g_initial_features)

        #########################
        # Encoder (Downsampling path)
        for i in range(g_unet_size):
            next_encoder, pipeline_channels = self.make_encoder(pipeline_channels)
            self.unet_encoders.append(next_encoder)
            self.unet_downsamplers.append(nn.MaxPool2d(2))

        ##########################
        # Bottleneck
        self.bottleneck, pipeline_channels = self.make_bottleneck(pipeline_channels)

        self.short_circuit = nn.Conv2d(pipeline_channels, 1, kernel_size=3, padding=1)

        ##########################
        # Decoder (Upsampling path) - Using PixelShuffle for efficient upsampling
        # The number of input channels to PixelShuffle must be factor^2 * C_out
        for i in range(g_unet_size):
            next_decoder, pipeline_channels =  self.make_decoder(pipeline_channels)
            self.unet_decoders.append(next_decoder)
            self.unet_skipconns.append(self.make_skip_connector(pipeline_channels))

        # Final Convolution
        # Adjust final upsampling if factor is 4x and you only have 2x layers
        # if upsample_factor == 4:
        #     pipeline_channels_next = pipeline_channels / 2
        #     self.final_upsample = nn.Sequential(
        #         nn.Conv2d(pipeline_channels, pipeline_channels_next * (2**2), kernel_size=3, padding=1),
        #         nn.PixelShuffle(2),
        #         ResidualBlock(2 * pipeline_channels_next, pipeline_channels_next) # Concatenate with initial conv_in output (64 channels)
        #     )
        #     pipeline_channels = pipeline_channels_next
            
        #     self.dec3 = nn.Sequential(
        #         ResidualBlock(64 + 64, 64),
        #         # ResidualBlock(64, 64)
        #     )        
        #     self.conv_out = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        # else: # upsample_factor == 2
        #     self.conv_out = nn.Conv2d(128, 1, kernel_size=3, padding=1) # No extra upsample needed if input 2x already
        self.conv_out = nn.Conv2d(pipeline_channels, 1, kernel_size=3, padding=1)
        
        if use_sigmoid:
            self.clamp_output = nn.Sigmoid() # Or adjust based on your image range [0,1] or [-1,1]

    def pre_transform(self, x):
        if self.use_log_space:
            x = torch.log2(x + g_epsilon)
            if g_normalize_input:
                if self.previous_range != -1:
                    raise Exception('Cannot pre_transform without first matching a post_transform call for the previous call to pre_transform')
                self.previous_range = input_channel.max()
                input_channel /= self.previous_range
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
        for i in range(g_unet_size):
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
        for i in range(g_unet_size):
            pipeline_state = self.unet_decoders[i](pipeline_state)
            pipeline_state = torch.cat([pipeline_state, unet_skip_sources[g_unet_size - 1 - i]], dim=1)
            pipeline_state = self.unet_skipconns[i](pipeline_state)

        # up1 = self.upsample1(f_bottle) # [B, 256, H/2, W/2] (output of pixelshuffle, before residual)
        # # Skip connection: f_enc2 is [B, 256, H/2, W/2]
        # up1 = torch.cat([up1, f_enc2], dim=1) # Concatenate features from encoder
        # up1 = self.skipconn1(up1)

        # # Upsample 2 (from H/2 to H)
        # up2 = self.upsample2(up1) # [B, 128, H, W] (output of pixelshuffle, before residual)
        # # Skip connection: f_enc1 is [B, 128, H, W]
        # up2 = torch.cat([up2, f_enc1], dim=1) # Concatenate features from encoder
        # up2 = self.skipconn2(up2)
        
        output = self.conv_out(pipeline_state)
        # output = self.short_circuit(short_circuit_output)

        # Handle 4x upsampling
        if self.upsample_factor == 4:
            # The previous upsamples brought it from H/4 to H. Now need H to 4H.
            # This logic needs adjustment. If upsample_factor=4, you'd need a direct 4x pixelshuffle or
            # more cascading 2x upsamples. Let's simplify for a direct 4x upsample at the end
            # or ensure the intermediate stages align correctly.

            # More robust way for 4x:
            # If input is HxW, first pixelshuffle takes it to 2Hx2W
            # Second pixelshuffle takes it to 4Hx4W
            # The current setup is for total 4x upsampling if input is H/4 x W/4 at bottleneck.
            # If input to model is HR (no upsampling for gap-filling, then final upsample 1x)
            # Or if input is LR for SR, then each stage upsamples.

            # Let's adjust for a common SR pipeline:
            # Input LR (H, W) -> Features -> Upsample H'xW' -> Output HR
            
            # For 2x or 4x upsampling, the input `x_lr` would be the *low-resolution* image.
            # The architecture above currently aims to bring it back to the *original resolution* of f_in.
            # We need to explicitly upsample for 2x or 4x.

            # Re-thinking upsampling:
            # If upsample_factor is 2x:
            # x_lr (H, W) -> enc1 (H, W) -> pool1 (H/2, W/2) -> enc2 (H/2, W/2) -> pool2 (H/4, W/4)
            # bottleneck (H/4, W/4)
            # upsample1 (H/4 to H/2) -> upsample2 (H/2 to H)
            # This structure means output is same resolution as input.
            
            # For upsampling:
            # The pools should be removed or designed to produce output *at target HR*.
            # For 2x or 4x, you don't typically downsample the input if it's already LR.
            # Instead, the network extracts features at LR, then upsamples.
            pass

        if self.upsample_factor == 2:
            # Assuming `x_lr` is the LR input, and we want 2x output.
            # The current `upsample2` gives 1x original resolution.
            # We need one more 2x upsample stage.
            # Let's assume the previous `upsample2` brought features to the *target 2x resolution*.
            # This requires careful channel adjustments.

            # Simplified example for 2x/4x using one PixelShuffle at the end:
            
            # Input (LR): [B, 3, H, W]
            # conv_in: [B, 64, H, W]
            # encoder_stages: process at (H,W)
            # Then, before final conv_out:
            
            # self.upconv = nn.Sequential(
            #     nn.Conv2d(128, 3 * (upsample_factor**2), kernel_size=3, padding=1),
            #     nn.PixelShuffle(upsample_factor)
            # )
            # output = self.upconv(last_feature_map)

            # Let's correct the conceptual model for upsampling:
            # It's better to build the network based on the desired output resolution.
            # If `x_lr` is the input (say, original resolution for gap-filling, or LR for SR)
            # And we want 2x/4x upsampled output.

            pass # Placeholder for corrected logic

        # Corrected structure for upsampling:
        # Instead of MaxPool and then upsampling to original,
        # often SR models have a feature extractor, then an upsampling module.
        # For joint SR+Inpainting, the encoder handles both degradation features and context.

        # Let's re-define the forward pass conceptually for a single pass:
        # 1. Feature extraction at input resolution (with gaps).
        # 2. Downsample (if needed) for context, process with residual blocks.
        # 3. Upsample back to desired output resolution (2x/4x) using PixelShuffle.
        # 4. Final reconstruction layer.

        # Simplified PhotonerNet for 2x/4x SR and Inpainting:
        # (This is a common pattern for fast SR models like FSRCNN or ESPCN-like)

        # Remove explicit pooling in the encoder if we want direct upsampling from LR.
        # Instead, use strides=1 in conv layers for feature extraction
        # And then a dedicated upsampling block.

        # If upsample_factor is 2 or 4, this means `x_lr` is the input.
        # The goal is to output `x_hr` which is `x_lr` upsampled by `upsample_factor`.

        # Redo the forward based on common SR architectures + inpainting aspects:
        # 1. Feature extraction from LR input (with gaps)
        # 2. Many residual blocks to learn mapping and fill in context
        # 3. Upsampling layer (PixelShuffle)
        # 4. Final convolution

        if self.use_sigmoid:
            return self.clamp_output(output) # Apply sigmoid to ensure [0,1] output if needed
        else:
            return output

    def post_transform(self, x):
        if self.use_log_space:
            if g_normalize_input:
                if self.previous_range == -1:
                    raise Exception('Cannot post_transform without a prior call to pre_transform')
                x *= self.previous_range
            x = torch.exp2(x) - g_epsilon
        return x
    
    def make_feature_extraction(self, channels_in, features):
        channels_out = features
        module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, padding_mode=g_padding_mode),
            nn.ReLU(inplace=True),
            ResidualBlock(channels_out, channels_out)
        )
        return module, channels_out
    
    def make_encoder(self, channels_in):
        channels_out = channels_in * 2
        module = ResidualBlock(channels_in, channels_out)
        return module, channels_out
    
    def make_bottleneck(self, channels_in):
        channels_out = channels_in * 2
        module = nn.Sequential(
            ResidualBlock(channels_in, channels_out),
            ResidualBlock(channels_out, channels_out)
        )
        return module, channels_out
    
    def make_decoder(self, channels_in):
        channels_out = channels_in // 2
        module = nn.Sequential(
            nn.Conv2d(channels_in, channels_out * (2*2), kernel_size=3, padding=1), # Output for 2x upsample
            nn.PixelShuffle(2), # Upsamples features by 2x
        )
        return module, channels_out
    
    def make_skip_connector(self, channels):
        return nn.Sequential(
            ResidualBlock(2 * channels, channels),
            ResidualBlock(channels, channels)
        )

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def create_window(self, window_size):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(1, 1, window_size, window_size).contiguous()
        return window

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        return gauss/gauss.sum()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

def select_largest_range_channel(img_batch):
    # img_batch: [batch, 3, H, W]
    ranges = img_batch.max(dim=2)[0].max(dim=2)[0] - img_batch.min(dim=2)[0].min(dim=2)[0]  # [batch, 3]
    channel_indices = ranges.argmax(dim=1)  # [batch]
    # Gather the selected channel for each image in the batch
    selected = torch.stack([img_batch[i, c, :, :] for i, c in enumerate(channel_indices)], dim=0)  # [batch, H, W]
    return selected.unsqueeze(1)  # [batch, 1, H, W]

def use_sigmoid_from_input(input_files):
    # Assumes all input files are the same type
    return not input_files[0].lower().endswith('.exr')

def train(args):
    # Create datasets
    input_files = sorted(glob.glob(args.input_location))
    training_files = sorted(glob.glob(args.training_location))
    
    if len(input_files) < len(training_files):
        raise ValueError("There are fewer input files than training files. Each training file must have a corresponding input file.")
    # Only use the first N input files, where N is the number of training files
    input_files = input_files[:len(training_files)]
    
    # Split into train and test sets
    split_idx = int(len(input_files) * (1 - args.test_ratio))
    train_input = input_files[:split_idx]
    train_target = training_files[:split_idx]
    test_input = input_files[split_idx:]
    test_target = training_files[split_idx:]

    truth_transform = transforms.Compose([
        # transforms.Resize((8,8))
    ])
    
    train_dataset = PhotonerDataset(train_input, train_target, 
                                  args.crop_size, args.upsample, truth_transform)
    test_dataset = PhotonerDataset(test_input, test_target,
                                 args.crop_size, args.upsample, truth_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and move to device
    use_sigmoid = use_sigmoid_from_input(input_files)
    model = PhotonerNet(upsample_factor=args.upsample, use_sigmoid=use_sigmoid, use_log_space=train_dataset.exr_source and args.log_space).to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    ssim_loss = SSIM()
    # vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
    # perceptual_loss_fn = VGGPerceptualLoss(feature_layers=['relu3_3', 'relu4_3']).to(device)    

    # def perceptual_loss(pred, target):
    #     # pred and target: [batch, 1, H, W]
    #     pred_vgg = pred.repeat(1, 3, 1, 1)     # [batch, 3, H, W]
    #     target_vgg = target.repeat(1, 3, 1, 1) # [batch, 3, H, W]
    #     with torch.no_grad():
    #         return perceptual_loss_fn(pred_vgg, target_vgg)
    
    # Optimizer
    if g_use_adam_w:
        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=g_weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
    
    # Training loop
    start_time = time.time()
    last_checkpoint = start_time
    last_print = start_time
    
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (input_img, target_img) in enumerate(train_loader):
            # Clone tensors to make them resizable
            input_img = input_img.clone().detach().to(device)
            target_img = target_img.clone().detach().to(device)
            
            # Select largest range channel for both input and target
            input_channel = select_largest_range_channel(input_img)
            target_channel = select_largest_range_channel(target_img)
            
            optimizer.zero_grad()

            input_channel = model.pre_transform(input_channel)
            output = model(input_channel)
            output = model.post_transform(output)

            # Calculate losses
            loss_mse = mse_loss(output, target_channel)
            loss_l1 = l1_loss(output, target_channel)
            # loss_ssim = 1 - ssim_loss(output, target_channel)
            # loss_perceptual = perceptual_loss(output, target_channel)
            
            # total_loss = loss_mse + 0.5 * loss_ssim + 0.1 * loss_perceptual
            total_loss = loss_l1 + 0.25 * loss_mse # 0.1 * loss_perceptual
            # total_loss = loss_l1
            
            # total_loss.requires_grad = True
            total_loss.backward()
            optimizer.step()
            
            # Console output every 5 seconds
            current_time = time.time()
            if current_time - last_print >= 5:
                elapsed = current_time - start_time
                print(f"{elapsed:.2f},{epoch},{epoch*len(train_dataset) + batch_idx*len(input_img)},{total_loss.item():.6f}")
                last_print = current_time
                
                # Checkpoint if needed
                if args.checkpoint_interval and current_time - last_checkpoint >= args.checkpoint_interval:
                    checkpoint_dir = os.path.join(args.checkpoint_folder, f"{int(elapsed)}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
                    
                    # Evaluate checkpoint tests if provided
                    if args.checkpoint_tests:
                        evaluate(model, args.checkpoint_tests, checkpoint_dir, args)
                        model.train()
                        
                    last_checkpoint = time.time()
    
    # Save final model
    torch.save(model.state_dict(), args.model_path)
    
    # Export to ONNX if requested
    if args.onnx_export:
        dummy_input = torch.randn(1, 1, args.crop_size, args.crop_size).to(device)
        torch.onnx.export(model, dummy_input, args.onnx_export,
                         input_names=['input'], output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})
        
def infer_large(model, img, tile=256, overlap=8):
    _, C, H, W = img.shape
    stride = tile - overlap
    out = torch.zeros_like(img)
    counts = torch.zeros_like(img)

    for y in range(0, H - overlap, stride):
        for x in range(0, W - overlap, stride):
            y1, y2 = y, y + tile
            x1, x2 = x, x + tile
            if y2 > H or x2 > W:
                continue
            tile_in = img[:, :, y1:y2, x1:x2]
            # tile_out = model(tile_in)
            tile_in = model.pre_transform(tile_in)
            tile_out = model.post_transform(model(tile_in))
            tile_out = transforms.Resize((tile,tile))(tile_out)

            # crop inner region to avoid boundary artefacts
            inner = overlap // 2
            out[:, :, y1+inner:y2-inner, x1+inner:x2-inner] += \
                tile_out[:, :, inner:-inner, inner:-inner]
            counts[:, :, y1+inner:y2-inner, x1+inner:x2-inner] += 1

    return out / counts.clamp(min=1)

def evaluate(model, input_pattern, output_folder, args):
    model.eval()
    input_files = sorted(glob.glob(input_pattern))
    
    os.makedirs(output_folder, exist_ok=True)
    
    with torch.no_grad():
        for input_path in input_files:
            dataset = PhotonerDataset([input_path], None, args.crop_size, 
                                    args.upsample)
            input_img = dataset[0][0].unsqueeze(0).to(device)
            
            # 
            # Process each color channel
            output_channels = []
            for c in range(3):
                # output = model(input_img[:, c:c+1])
                output = infer_large(model, input_img[:, c:c+1], 256, 1 << g_unet_size)
                output_channels.append(output)
                
            output_img = torch.cat(output_channels, dim=1)
                
            # Save output
            output_name = os.path.basename(input_path).rsplit('.', 1)[0] + '_eval.' + input_path.rsplit('.', 1)[1]
            output_path = os.path.join(output_folder, output_name)
            
            if output_path.lower().endswith('.exr'):
                # Save as EXR
                header = OpenEXR.Header(output_img.shape[2], output_img.shape[3])
                header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
                
                out = OpenEXR.OutputFile(output_path, header)
                R = output_img[0, 0].cpu().numpy().astype(np.float32).tobytes()
                G = output_img[0, 1].cpu().numpy().astype(np.float32).tobytes()
                B = output_img[0, 2].cpu().numpy().astype(np.float32).tobytes()
                out.writePixels({'R': R, 'G': G, 'B': B})
                out.close()
            else:
                # Save as PNG
                output_img = output_img.pow(1/2.2)  # Convert to sRGB
                output_img = output_img.clamp(0, 1)
                output_img = (output_img * 255).byte()
                output_img = output_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                Image.fromarray(output_img).save(output_path)
            # return # only process one for now

def main():
    args = parse_args()
    print(f"Using device: {device}")

    if args.eval:
        input_files = sorted(glob.glob(args.input_location))
        use_sigmoid = use_sigmoid_from_input(input_files)
        model = PhotonerNet(upsample_factor=args.upsample, use_sigmoid=use_sigmoid).to(device)
        model.load_state_dict(torch.load(args.model_path))
        evaluate(model, args.input_location, args.output_folder, args)
    else:
        train(args)

if __name__ == "__main__":
    main() 