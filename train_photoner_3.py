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
import torchvision

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
    parser.add_argument('--checkpoint-interval', type=int, help='Seconds between checkpoints')
    parser.add_argument('--checkpoint-folder', help='Folder to save checkpoint data')
    parser.add_argument('--checkpoint-tests', help='Path to test images for checkpoints')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Percentage of data for testing')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--log-space', action='store_true', help='Transform EXR data to log space')
    parser.add_argument('--crop-size', type=int, default=64, help='Resolution of training crops')
    parser.add_argument('--upsample', type=int, default=1, choices=[1, 2, 4, 8], help='Upsampling factor')
    parser.add_argument('--onnx-export', help='Path to export ONNX model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training and testing') 
    parser.add_argument('--learn-rate', type=float, default=0.001, help='Learning rate') 
    
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
                 crop_size: int = 64, log_space: bool = False, upsample: int = 1):
        self.input_paths = input_paths
        self.training_paths = training_paths
        self.crop_size = crop_size
        self.log_space = log_space
        self.upsample = upsample
        self.epsilon = 1e-6  # For log space transformation
        
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
        
        if self.log_space:
            tensor = torch.log2(tensor + self.epsilon)
            
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
            
            return input_tensor, target_tensor
        
        return input_tensor, None

class PhotonerNet(nn.Module):
    def __init__(self, upsample_factor=1, use_sigmoid=False):
        super().__init__()
        # Initial feature extraction
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GELU()
        )

        # Replace transformer blocks with additional convolutions (with residuals)
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
        )

        self.upsample_factor = upsample_factor
        if upsample_factor > 1:
            self.upsampling = nn.Sequential(
                nn.Conv2d(256, 256 * upsample_factor * upsample_factor, 1),
                nn.PixelShuffle(upsample_factor)
            )

        refinement_layers = [
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, 3, padding=1)
        ]
        if use_sigmoid:
            refinement_layers.append(nn.Sigmoid())
        self.refinement = nn.Sequential(*refinement_layers)

    def forward(self, x):
        features = self.conv_features(x)
        # Residual convolutional blocks
        residual = features
        features = self.conv_blocks(features)
        features = features + residual  # Residual connection

        if self.upsample_factor > 1:
            features = self.upsampling(features)
        output = self.refinement(features)
        return output

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
    
    train_dataset = PhotonerDataset(train_input, train_target, 
                                  args.crop_size, args.log_space, args.upsample)
    test_dataset = PhotonerDataset(test_input, test_target,
                                 args.crop_size, args.log_space, args.upsample)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model and move to device
    use_sigmoid = use_sigmoid_from_input(input_files)
    model = PhotonerNet(upsample_factor=args.upsample, use_sigmoid=use_sigmoid).to(device)
    
    # Loss functions
    l1_loss = nn.L1Loss()
    ssim_loss = SSIM()
    vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16].to(device).eval()
    
    def perceptual_loss(pred, target):
        # pred and target: [batch, 1, H, W]
        pred_vgg = pred.repeat(1, 3, 1, 1)     # [batch, 3, H, W]
        target_vgg = target.repeat(1, 3, 1, 1) # [batch, 3, H, W]
        with torch.no_grad():
            return F.mse_loss(vgg(pred_vgg), vgg(target_vgg))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    start_time = time.time()
    last_checkpoint = start_time
    last_print = start_time
    
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (input_img, target_img) in enumerate(train_loader):
            # Clone tensors to make them resizable
            input_img = input_img.clone().detach().to(device)
            target_img = target_img.clone().detach().to(device)
            
            # Select largest range channel for both input and target
            input_channel = select_largest_range_channel(input_img)
            target_channel = select_largest_range_channel(target_img)
            
            optimizer.zero_grad()
            output = model(input_channel)
            
            # Calculate losses
            loss_l1 = l1_loss(output, target_channel)
            loss_ssim = 1 - ssim_loss(output, target_channel)
            loss_perceptual = perceptual_loss(output, target_channel)
            
            # total_loss = loss_l1 + 0.5 * loss_ssim + 0.1 * loss_perceptual
            total_loss = loss_perceptual
            
            total_loss.requires_grad = True
            total_loss.backward()
            optimizer.step()
            
            # Console output every 5 seconds
            current_time = time.time()
            if current_time - last_print >= 5:
                elapsed = current_time - start_time
                print(f"{elapsed:.2f},{epoch},{batch_idx * len(input_img)},{total_loss.item():.6f}")
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
                        
                    last_checkpoint = current_time
    
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
            tile_out = model(tile_in)

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
                                    args.log_space, args.upsample)
            input_img = dataset[0][0].unsqueeze(0).to(device)
            
            # 
            # Process each color channel
            output_channels = []
            for c in range(3):
                output = infer_large(model, input_img[:, c:c+1])
                output_channels.append(output)
                
            output_img = torch.cat(output_channels, dim=1)
            
            # Convert back from log space if needed
            if args.log_space:
                output_img = torch.exp2(output_img) - dataset.epsilon
                
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