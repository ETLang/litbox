import torch
import torch.nn as nn


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
        
        self.final = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.primary(x)
        out += residual # Add residual connection
        out = self.final(out)
        return out

class PhotonerNet(nn.Module):
    def __init__(self,
                 upsample_factor, 
                 use_sigmoid=False, 
                 use_log_space=True, 
                 normalize_input=True, 
                 initial_features=16, 
                 unet_size=3, 
                 epsilon=1e-6,
                 padding_mode='reflect'):
        super(PhotonerNet, self).__init__()        
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
        pipeline_channels = 1
        
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

        self.short_circuit = nn.Conv2d(pipeline_channels, 1, kernel_size=3, padding=1)

        ##########################
        # Decoder (Upsampling path) - Using PixelShuffle for efficient upsampling
        # The number of input channels to PixelShuffle must be factor^2 * C_out
        for i in range(self.unet_size):
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

    def pre_transform(self, x): # [B, 1, H, W]
        if self.use_log_space:
            x = torch.log2(x + self.epsilon)
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
        channels_out = channels_in * 2
        module = ResidualBlock(channels_in, channels_out, self.padding_mode)
        return module, channels_out
    
    def make_bottleneck(self, channels_in):
        channels_out = channels_in * 2
        module = nn.Sequential(
            ResidualBlock(channels_in, channels_out, self.padding_mode),
            ResidualBlock(channels_out, channels_out, self.padding_mode)
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
            ResidualBlock(2 * channels, channels, self.padding_mode),
            ResidualBlock(channels, channels, self.padding_mode)
        )
