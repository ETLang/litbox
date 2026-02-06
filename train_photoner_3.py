import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
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
from photoner_display import PhotonerDisplay
from photoner_loss import HdrLoss
from photoner_dataset import PhotonerDataset
from photoner_model import PhotonerNet

# Settings (overridable via command line arguments)
g_output_upsample = 1 # 4
g_checkpoint_interval = 900
g_test_ratio = 0.0
g_epochs = 20
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
g_loss_dark_bias = 0.5 # 1.0 is OK
g_loss_bright_weight = 1.5
g_loss_gradient_weight = 0.4 # 0.1 is OK
g_loss_l1_weight = 0.2

# TODO
g_gaussian_initialization = True


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Photoner Neural Network Training Script')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--input-a-easy-location', help='Path to easy input imageset A, for curriculum training')
    parser.add_argument('--input-b-easy-location', help='Path to easy input imageset B, for curriculum training')
    parser.add_argument('--input-a-medium-location', help='Path to input imageset A, for curriculum training')
    parser.add_argument('--input-b-medium-location', help='Path to input imageset B, for curriculum training')
    parser.add_argument('--input-a-location', required=True, help='Path to input imageset A')
    parser.add_argument('--input-b-location', required=True, help='Path to input imageset B')
    parser.add_argument('--input-albedo-location', required=True, help='Path to albedo imageset')
    parser.add_argument('--input-transmissibility-location', required=True, help='Path to transmissibility imageset')
    parser.add_argument('--reference-location', help='Path to reference images for training')
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
    if not args.eval and not args.reference_location:
        parser.error("--reference-location is required in training mode")
    if args.eval and not args.output_folder:
        parser.error("--output-folder is required in eval mode")
    if args.checkpoint_interval and not args.checkpoint_folder:
        parser.error("--checkpoint-folder is required when using --checkpoint-interval")
    if args.input_a_easy_location and not args.input_b_easy_location:
        parser.error("Both --input-a-easy-location and --input-b-easy-location must be provided for easy curriculum training")
    if args.input_b_easy_location and not args.input_a_easy_location:
        parser.error("Both --input-a-easy-location and --input-b-easy-location must be provided for easy curriculum training")
    if args.input_a_medium_location and not args.input_b_medium_location:
        parser.error("Both --input-a-medium-location and --input-b-medium-location must be provided for medium curriculum training")
    if args.input_b_medium_location and not args.input_a_medium_location:
        parser.error("Both --input-a-medium-location and --input-b-medium-location must be provided for medium curriculum training")
        
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

def compute_mean_and_relative_variance(image_a, image_b):
    mean = (image_a + image_b) / 2.0
    relative_variance = (image_a - image_b) ** 2 / (mean ** 2 + 1e-5)
    # Originally I was planning on using a lower resolution variance map, but for now let's try full resolution
    # relative_variance = F.avg_pool2d(relative_variance, kernel_size=4, stride=4)
    relative_variance = relative_variance.mean(dim=1, keepdim=True)
    return mean, relative_variance

def train(args):
    display = PhotonerDisplay()

    # Create datasets
    input_sets = []
    reference_files = sorted(glob.glob(args.reference_location))
    albedo_files = sorted(glob.glob(args.input_albedo))
    if len(albedo_files) < len(reference_files):
        raise ValueError("There are fewer albedo files than reference files. Each reference file must have a corresponding albedo file.")
    albedo_files = albedo_files[:len(reference_files)]
    transmissibility_files = sorted(glob.glob(args.input_transmissibility))
    if len(transmissibility_files) < len(reference_files):
        raise ValueError("There are fewer transmissibility files than reference files. Each reference file must have a corresponding transmissibility file.")
    transmissibility_files = transmissibility_files[:len(reference_files)]

    if args.input_a_easy_location:
        input_a_easy_files = sorted(glob.glob(args.input_a_easy_location))
        input_b_easy_files = sorted(glob.glob(args.input_b_easy_location))
        if len(input_a_easy_files) < len(reference_files) or len(input_b_easy_files) < len(reference_files):
            raise ValueError("There are fewer input files than reference files. Each reference file must have a corresponding input file.")
        input_a_easy_files = input_a_easy_files[:len(reference_files)]
        input_b_easy_files = input_b_easy_files[:len(reference_files)]
        input_sets.append(("Easy", input_a_easy_files, input_b_easy_files))
    if args.input_a_medium_location:
        input_a_medium_files = sorted(glob.glob(args.input_a_medium_location))
        input_b_medium_files = sorted(glob.glob(args.input_b_medium_location))
        if len(input_a_medium_files) < len(reference_files) or len(input_b_medium_files) < len(reference_files):
            raise ValueError("There are fewer input files than reference files. Each reference file must have a corresponding input file.")
        input_a_medium_files = input_a_medium_files[:len(reference_files)]
        input_b_medium_files = input_b_medium_files[:len(reference_files)]
        input_sets.append(("Medium", input_a_medium_files, input_b_medium_files))
    input_a_final_files = sorted(glob.glob(args.input_a_location))
    input_b_final_files = sorted(glob.glob(args.input_b_location))
    if len(input_a_final_files) < len(reference_files) or len(input_b_final_files) < len(reference_files):
        raise ValueError("There are fewer input files than reference files. Each reference file must have a corresponding input file.")
    input_a_final_files = input_a_final_files[:len(reference_files)]
    input_b_final_files = input_b_final_files[:len(reference_files)]
    input_sets.append(("Final", input_a_final_files, input_b_final_files))

    # Initialize model
    use_sigmoid = use_sigmoid_from_input(input_a_final_files)
    model = PhotonerNet(
        upsample_factor=args.upsample, 
        use_sigmoid=use_sigmoid, 
        use_log_space=False, #train_dataset.exr_source and args.log_space,
        normalize_input=g_normalize_input, 
        initial_features=g_initial_features,
        unet_size=g_unet_size,
        epsilon=g_epsilon, 
        padding_mode=g_padding_mode).to(device)
    
    # Loss functions
    loss_fn = HdrLoss(g_loss_bright_weight, g_loss_gradient_weight, g_loss_l1_weight, g_loss_dark_bias)
    #loss_fn = nn.MSELoss()

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

    for curriculum_name, input_a_files, input_b_files in input_sets:
        # Split into train and test sets
        split_idx = int(len(reference_files) * (1 - args.test_ratio))
        train_a_input = input_a_files[:split_idx]
        train_b_input = input_b_files[:split_idx]
        train_albedo_input = albedo_files[:split_idx]
        train_transmissibility_input = transmissibility_files[:split_idx]
        train_reference = reference_files[:split_idx]
        test_a_input = input_a_files[split_idx:]
        test_b_input = input_b_files[split_idx:]
        test_albedo_input = albedo_files[split_idx:]
        test_transmissibility_input = transmissibility_files[split_idx:]
        test_reference = reference_files[split_idx:]

        truth_transform = transforms.Compose([
            # transforms.Resize((8,8))
        ])
        
        train_dataset = PhotonerDataset(train_a_input, train_b_input, train_albedo_input, train_transmissibility_input, train_reference, 
                                    args.crop_size, args.upsample, None) # truth_transform)
        test_dataset = PhotonerDataset(test_a_input, test_b_input, test_albedo_input, test_transmissibility_input, test_reference,
                                    args.crop_size, args.upsample, None) #truth_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        for epoch in range(args.epochs):
            for batch_idx, (input_a, input_b, albedo, transmissibility, reference) in enumerate(train_loader):
                # Clone tensors to make them resizable
                # TODO: Check if this is necessary
                input_a = input_a.clone().detach().to(device)
                input_b = input_b.clone().detach().to(device)
                albedo = albedo.clone().detach().to(device)
                transmissibility = transmissibility.clone().detach().to(device)
                reference = reference.clone().detach().to(device)
                    
                # TODO uncertain if it would be best to train with all three color channels, or treat them independently
                # Select random channel for both input and target (using same index)
                # input_channel, target_channel = select_random_channel(input_a, reference)
                
                optimizer.zero_grad()

                input_channel = model.pre_transform(input_channel)
                output = model(input_channel)
                output = model.post_transform(output)

                # Calculate losses
                loss = loss_fn(output, target_channel)
                loss.backward()
                optimizer.step()
                
                # Console output every 5 seconds
                current_time = time.time()
                if current_time - last_print >= 10:
                    elapsed = current_time - start_time
                    print(f"{elapsed:.2f},{curriculum_name},{epoch},{epoch*len(train_dataset) + batch_idx*len(input_img)},{loss.item():.6f}")
                    last_print = current_time

                    display.show(input_channel, output, target_channel)
                    
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
    
    display.shutdown()
    
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