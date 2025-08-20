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
from photoner_loss import HdrLoss
from photoner_dataset import PhotonerDataset
from photoner_model import PhotonerNet

# Settings (overridable via command line arguments)
g_output_upsample = 1 # 4
g_checkpoint_interval = 900
g_test_ratio = 0.0
g_epochs = 100
g_crop_size = 256
g_batch_size = 4
g_learn_rate = 0.00001 # 0.001

# Settings (internal)
g_unet_size = 5
g_padding_mode = 'reflect'
g_initial_features = 32
g_normalize_input = False
g_use_adam_w = True
g_weight_decay = 0.01
g_epsilon = 1e-6 

# TODO
g_gaussian_initialization = True


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Photoner Neural Network Training Script')
   # parser.add_argument('--help', action='help', help='Displays documentation regarding these arguments')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--input-easy-location', help='Path to easy input images, for curriculum training')
    parser.add_argument('--input-medium-location', help='Path to input images, for curriculum training')
    parser.add_argument('--input-location', required=True, help='Path to input images')
    parser.add_argument('--training-location', help='Path to training images')
    parser.add_argument('--output-folder', help='Output folder for evaluated images')
    parser.add_argument('--model-path', required=True, help='Path to save/load model')
    parser.add_argument('--checkpoint-interval', type=int, default=g_checkpoint_interval, help='Seconds between checkpoints')
    parser.add_argument('--checkpoint-folder', help='Folder to save checkpoint data')
    parser.add_argument('--checkpoint-tests', help='Path to test images for checkpoints')
    parser.add_argument('--test-ratio', type=float, default=g_test_ratio, help='Percentage of data for testing')
    parser.add_argument('--epochs', type=int, default=g_epochs, help='Number of epochs to train, per curriculum stage')
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

def select_random_channel(img_batch, target_batch=None):
    # img_batch: [batch, 3, H, W]
    batch_size = img_batch.shape[0]
    # Generate random channel indices for each image in the batch
    channel_indices = torch.randint(0, 3, (batch_size,), device=img_batch.device)  # [batch]
    # Gather the selected channel for each image in the batch
    img_selected = torch.stack([img_batch[i, c, :, :] for i, c in enumerate(channel_indices)], dim=0)  # [batch, H, W]
    if target_batch is not None:
        target_selected = torch.stack([target_batch[i, c, :, :] for i, c in enumerate(channel_indices)], dim=0)
        return img_selected.unsqueeze(1), target_selected.unsqueeze(1) # [batch, 1, H, W]
    else:
        return img_selected.unsqueeze(1)  # [batch, 1, H, W]

def use_sigmoid_from_input(input_files):
    # Assumes all input files are the same type
    return not input_files[0].lower().endswith('.exr')

def train(args):
    # Create datasets
    input_sets = []
    training_files = sorted(glob.glob(args.training_location))
    if args.input_easy_location:
        input_easy_files = sorted(glob.glob(args.input_easy_location))
        if len(input_easy_files) < len(training_files):
            raise ValueError("There are fewer input files than training files. Each training file must have a corresponding input file.")
        input_easy_files = input_easy_files[:len(training_files)]
        input_sets.append(("Easy", input_easy_files))
    if args.input_medium_location:
        input_medium_files = sorted(glob.glob(args.input_medium_location))
        if len(input_medium_files) < len(training_files):
            raise ValueError("There are fewer input files than training files. Each training file must have a corresponding input file.")
        input_medium_files = input_medium_files[:len(training_files)]
        input_sets.append(("Medium", input_medium_files))
    input_final_files = sorted(glob.glob(args.input_location))
    if len(input_final_files) < len(training_files):
        raise ValueError("There are fewer input files than training files. Each training file must have a corresponding input file.")
    input_final_files = input_final_files[:len(training_files)]
    input_sets.append(("Final", input_final_files))

    for curriculum_name, input_files in input_sets:
        # Split into train and test sets
        split_idx = int(len(training_files) * (1 - args.test_ratio))
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
        model = PhotonerNet(
            upsample_factor=args.upsample, 
            use_sigmoid=use_sigmoid, 
            use_log_space=train_dataset.exr_source and args.log_space,
            normalize_input=g_normalize_input, 
            initial_features=g_initial_features,
            unet_size=g_unet_size, 
            epsilon=g_epsilon, 
            padding_mode=g_padding_mode).to(device)

        
        # Loss functions
        hdr_loss = HdrLoss()
        # mse_loss = nn.MSELoss()
        # l1_loss = nn.L1Loss()
        # ssim_loss = SSIM()
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
                # TODO: Check if this is necessary
                input_img = input_img.clone().detach().to(device)
                target_img = target_img.clone().detach().to(device)
                
                # Select random channel for both input and target (using same index)
                input_channel, target_channel = select_random_channel(input_img, target_img)
                
                optimizer.zero_grad()

                input_channel = model.pre_transform(input_channel)
                output = model(input_channel)
                output = model.post_transform(output)

                # Calculate losses
                loss = hdr_loss(output, target_channel)
                # loss_mse = mse_loss(output, target_channel)
                # loss_l1 = l1_loss(output, target_channel)
                # loss_ssim = 1 - ssim_loss(output, target_channel)
                # loss_perceptual = perceptual_loss(output, target_channel)
                
                # total_loss = loss_mse + 0.5 * loss_ssim + 0.1 * loss_perceptual
                # total_loss = loss_l1 + 0.25 * loss_mse
                # total_loss = 0.5 * loss_l1 + loss_mse
                # total_loss = loss_l1
                
                # total_loss.requires_grad = True
                # total_loss.backward()
                loss.backward()
                optimizer.step()
                
                # Console output every 5 seconds
                current_time = time.time()
                if current_time - last_print >= 5:
                    elapsed = current_time - start_time
                    print(f"{elapsed:.2f},{curriculum_name},{epoch},{epoch*len(train_dataset) + batch_idx*len(input_img)},{loss.item():.6f}")
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

            # Process each color channel independently
            channels_out = []
            for c in range(tile_in.shape[1]):
                channel_in = tile_in[:, c:c+1]  # Select single channel
                channel_in = model.pre_transform(channel_in)
                channel_out = model.post_transform(model(channel_in))
                channels_out.append(channel_out)
            
            # Recombine channels
            tile_out = torch.cat(channels_out, dim=1)
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
        model = PhotonerNet(
            upsample_factor=args.upsample, 
            use_sigmoid=use_sigmoid, 
            use_log_space=args.log_space, 
            normalize_input=g_normalize_input, 
            initial_features=g_initial_features,
            unet_size=g_unet_size, 
            epsilon=g_epsilon, 
            padding_mode=g_padding_mode).to(device)
        model.load_state_dict(torch.load(args.model_path))
        evaluate(model, args.input_location, args.output_folder, args)
    else:
        train(args)

if __name__ == "__main__":
    main() 