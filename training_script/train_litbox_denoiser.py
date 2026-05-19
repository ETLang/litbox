import json

import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import glob
import time
import random
import argparse
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import OpenEXR
import Imath
from litbox_display import LitboxDenoiserDisplay
from litbox_loss import HdrLoss
from litbox_loss import RelativeCharbonnierLoss
from litbox_dataset import LitboxDenoiserDataset
from litbox_model import LitboxDenoiserNet
import data_processing
from preprocess_data import build_cache, load_cached_stats
from preprocess_data import create_litbox_cached_datasets
from curriculum_manager import CurriculumManager
from data_processing import augment_for_training
import wandb

# Settings (overridable via command line arguments)
g_output_upsample = 1 # 4
g_checkpoint_interval = 900
g_test_ratio = 0.0
g_epochs = 20
g_crop_size = 256
g_batch_size = 4
g_learn_rate_min = 1e-6
g_learn_rate_max = 0.0001
g_momentum_min = 0.85
g_momentum_max = 0.95

# Settings (internal)
g_unet_size = 5
g_padding_mode = 'replicate'
g_initial_features = 32
g_normalize_input = False
g_use_adam_w = True
g_use_sigmoid = False
g_weight_decay = 0.01
g_epsilon = 1e-6 
g_loss_dark_bias = 0.5 # 1.0 is OK
g_loss_bright_weight = 1.5
g_loss_gradient_weight = 0.4 # 0.1 is OK
g_loss_l1_weight = 0.2

# Cosine Annealing with Warmup settings
g_warmup_epochs = 2

# TODO
g_gaussian_initialization = True


# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between
    the initial LR set in the optimizer to a minimum LR, after a warmup period during which it increases linearly
    between 0 and the initial LR set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        min_lr_ratio = g_learn_rate_min / g_learn_rate_max
        return max(0.0, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def adjust_momentum(optimizer, current_step, num_warmup_steps, num_training_steps):
    """
    Adjusts momentum opposite to learning rate.
    High momentum for low LR, low momentum for high LR.
    """
    if current_step < num_warmup_steps:
        # During warmup, LR increases, so momentum decreases from max to base
        progress = float(current_step) / float(max(1, num_warmup_steps))
        momentum = g_momentum_max - (g_momentum_max - g_momentum_min) * progress
    else:
        # After warmup, LR decreases, so momentum increases from base to max (cosine schedule)
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        momentum = g_momentum_min + (g_momentum_max - g_momentum_min) * 0.5 * (1.0 - math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if 'betas' in param_group:
            param_group['betas'] = (momentum, param_group['betas'][1])

def parse_args():
    parser = argparse.ArgumentParser(description='Litbox Denoiser Training Script')
    data_processing.register_dataset_args(parser, False)
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--finalize-checkpoint', type=int, help='Checkpoint to finalize')
    parser.add_argument('--recompute-stats', action='store_true', help='Compute global normalization stats, even if they are cached.')
    parser.add_argument('--skip-cache-validation', action='store_true', help='Trust that the cache is populated, and skip validating it')
    parser.add_argument('--cache-location', help='Path to preprocessed feature cache', default='./training_cache')
    parser.add_argument('--output-folder', required=True, help='Output folder for evaluated images or training results/checkpoints')
    parser.add_argument('--model-path', help='Path to model to use for eval')
    parser.add_argument('--checkpoint-interval', type=int, default=g_checkpoint_interval, help='Seconds between checkpoints')
    parser.add_argument('--checkpoint-tests', help='Path to test images for checkpoints')
    parser.add_argument('--test-ratio', type=float, default=g_test_ratio, help='Percentage of data for testing')
    parser.add_argument('--epochs', type=int, default=g_epochs, help='Number of epochs to train, per curriculum stage')
    parser.add_argument('--log-space', action='store_true', help='Transform EXR data to log space')
    parser.add_argument('--crop-size', type=int, default=g_crop_size, help='Resolution of training crops')
    parser.add_argument('--upsample', type=int, default=g_output_upsample, choices=[1, 2, 4, 8], help='Upsampling factor')
    parser.add_argument('--batch-size', type=int, default=g_batch_size, help='Batch size for training and testing') 
    parser.add_argument('--learn-rate-min', type=float, default=g_learn_rate_min, help='Minimum Learning rate') 
    parser.add_argument('--learn-rate-max', type=float, default=g_learn_rate_max, help='Maximum Learning rate') 
    parser.add_argument('--momentum-min', type=float, default=g_momentum_min, help='Minimum momentum') 
    parser.add_argument('--momentum-max', type=float, default=g_momentum_max, help='Maximum momentum') 
    
    args = parser.parse_args()
    
    # Validation
    is_training = not args.eval and not args.finalize_checkpoint
    if is_training and not args.reference_location:
        parser.error("--reference-location is required in training mode")
    if args.eval and not args.output_folder:
        parser.error("--output-folder is required in eval mode")
    if args.eval and not args.model_path:
        parser.error("--model-path is required in eval mode")
    data_processing.validate_dataset_args(args, parser)
        
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

def train(args, stats):
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="etlang-org",
        # Set the wandb project where this run will be logged.
        project="litbox-denoise",
        # Track hyperparameters and run metadata.
        config={
            "learn_rate_min": args.learn_rate_min,
            "learn_rate_max": args.learn_rate_max,
            "momentum_min": args.momentum_min,
            "momentum_max": args.momentum_max,
            "architecture": "UNet-5",
            "dataset": f"Litbox-{os.path.basename(os.path.dirname(args.input_a_location))}",
            "epochs": args.epochs,
            "upsample": args.upsample,
        },
    )

    # Save training configuration to JSON
    config_to_save = {
    "upsample": args.upsample,
    "epochs": args.epochs,
    "test_ratio": args.test_ratio,
    "crop_size": args.crop_size,
    "batch_size": args.batch_size,
    "learn_rate_min": args.learn_rate_min,
    "learn_rate_max": args.learn_rate_max,
    "momentum_min": args.momentum_min,
    "momentum_max": args.momentum_max,
    "unet_size": g_unet_size,
    "initial_features": g_initial_features,
    "input_a_location": args.input_a_location,
    "input_b_location": args.input_b_location,
    "input_albedo_location": args.input_albedo_location,
    "input_transmissibility_location": args.input_transmissibility_location,
    "reference_location": args.reference_location,
    "input_a_easy_location": args.input_a_easy_location,
    "input_b_easy_location": args.input_b_easy_location,
    }
    with open(os.path.join(args.output_folder, 'args.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)

    display = LitboxDenoiserDisplay()
    training_datasets, validation_datasets = create_litbox_cached_datasets(args, device)
    curriculum_manager = CurriculumManager(patience=3, min_delta=0.01)

    # Initialize model
    model = LitboxDenoiserNet(
        upsample_factor=args.upsample, 
        use_sigmoid=g_use_sigmoid, 
        use_log_space=False, #train_dataset.exr_source and args.log_space,
        normalize_input=g_normalize_input, 
        initial_features=g_initial_features,
        unet_size=g_unet_size,
        epsilon=g_epsilon, 
        padding_mode=g_padding_mode).to(device)

    # Optimizer
    if g_use_adam_w:
        weight_decay = g_weight_decay
    else:
        weight_decay = 0
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate_max, weight_decay=weight_decay)

    # Calculate total training steps for scheduler
    # Assuming all curriculum datasets have the same length
    num_samples = len(training_datasets[0])
    steps_per_epoch = (num_samples + args.batch_size - 1) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * g_warmup_epochs

    # LR Scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Training loop
    start_time = time.time()
    last_checkpoint = start_time
    last_print = start_time
    
    model.train()
    
    stop_training = False
    def on_key(event):
        nonlocal stop_training
        if event.key == 'q':
            stop_training = True
    display.fig.canvas.mpl_connect('key_press_event', on_key)

    # TODO: Easy may still be too hard. Add 64spp and 256spp
    curriculum_weights = [1, 0, 0] # [Easy, Medium, Hard]
    curriculum_spp = [16, 4, 1]
    mean = torch.tensor(stats['final_mean']).to(device)
    stddev = torch.tensor(stats['final_stddev']).to(device)

    # Loss functions
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = HdrLoss(g_loss_bright_weight, g_loss_gradient_weight, g_loss_l1_weight, g_loss_dark_bias)
    loss_fn = RelativeCharbonnierLoss(mean=mean, stddev=stddev)

    for epoch in range(args.epochs):
        if stop_training:
            break
        curriculum = random.choices(training_datasets, curriculum_weights[:len(training_datasets)], k=1)[0]

        loader = DataLoader(curriculum, batch_size=args.batch_size, shuffle=True, num_workers=2)

        for batch_idx, features in enumerate(loader):
            radiance = features['radiance']
            variance = features['variance']
            albedo = features['albedo']
            density = features['density']
            reference = features['reference']

            radiance, variance, albedo, density, reference = augment_for_training(
                radiance_stddev=stddev,
                radiance=radiance,
                variance=variance,
                albedo=albedo,
                transmissibility=density,
                reference=reference,
                upsample_factor=args.upsample,
                crop_size=args.crop_size
            )

            input_tensor = torch.cat([radiance, variance, albedo, density], dim=1)
            if ~torch.isfinite(input_tensor).all():
                print("oops input_tensor")

            output = model(input_tensor)
            # output = model.post_transform(output)

            if ~torch.isfinite(output).all():
                print("oops output has bad numbers")

            if ~torch.isfinite(reference).all():
                print("oops reference has bad numbers")

            # Calculate losses 
            loss = loss_fn(output, reference)
            
            # Zero gradients, backward pass, and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update LR and momentum
            lr_scheduler.step()
            adjust_momentum(optimizer, lr_scheduler.last_epoch, warmup_steps, total_steps)
            
            # Log to wandb
            if batch_idx % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                    "momentum": optimizer.param_groups[0]['betas'][0],
                    "epoch": epoch,
                    "curriculum": curriculum.name
                })
            
            # Console output every 5 seconds
            current_time = time.time()
            if current_time - last_print >= 10:
                elapsed = current_time - start_time
                print(f"{elapsed:.2f},{curriculum.name},{epoch},{epoch*len(curriculum) + batch_idx*len(radiance)},{loss.item():.6f}")
                last_print = current_time

                display.show(radiance, output, reference)
                
                # Checkpoint if needed
                if args.checkpoint_interval and current_time - last_checkpoint >= args.checkpoint_interval:
                    checkpoint_dir = os.path.join(args.output_folder, f"{int(elapsed)}")
                    os.makedirs(checkpoint_dir)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
                    
                    # Evaluate checkpoint tests if provided
                    if args.checkpoint_tests:
                        evaluate(model, args.checkpoint_tests, checkpoint_dir, args)
                        model.train()
                        
                    last_checkpoint = time.time()

            if stop_training:
                print("Stop key 'q' detected. Finishing training...")
                break

        # Validation stage
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for val_dataset in validation_datasets:
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
                for val_features in val_loader:
                    v_radiance = val_features['radiance']
                    v_variance = val_features['variance']
                    v_albedo = val_features['albedo']
                    v_density = val_features['density']
                    v_reference = val_features['reference']
                    
                    v_input = torch.cat([v_radiance, v_variance, v_albedo, v_density], dim=1)
                    v_output = model(v_input)
                    
                    v_loss = loss_fn(v_output, v_reference)
                    val_loss_total += v_loss.item()
                    val_steps += 1
        
        avg_val_loss = val_loss_total / max(val_steps, 1)
        wandb.log({ "val_loss": avg_val_loss })
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.6f}")
        
        model.train()

    plt.close(display.fig)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_folder, "final.pth"))

    # Export to ONNX
    # The model expects 8 input channels (radiance, variance, albedo, density)
    model.export_onnx(os.path.join(args.output_folder, "final.onnx"), input_channels=8, resolution=args.crop_size)
        
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

def evaluate(model, input_pattern, output_folder, args, stats):
    model.eval()
    input_files = sorted(glob.glob(input_pattern))
    
    with torch.no_grad():
        for input_path in input_files:
            dataset = LitboxDenoiserDataset([input_path], None, args.crop_size, 
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
    
    if args.finalize_checkpoint:
        config_path = os.path.join(args.output_folder, 'args.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved_args = json.load(f)
            # Override relevant model architecture params from saved config
            args.upsample = saved_args.get('upsample', args.upsample)
            args.crop_size = saved_args.get('crop_size', args.crop_size)
            global g_unet_size, g_initial_features
            g_unet_size = saved_args.get('unet_size', g_unet_size)
            g_initial_features = saved_args.get('initial_features', g_initial_features)

        checkpoint_path = os.path.join(args.output_folder, str(args.finalize_checkpoint), "model.pth")
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint model not found at {checkpoint_path}")
            return

        final_pth = os.path.join(args.output_folder, "final.pth")
        shutil.copy2(checkpoint_path, final_pth)

        model = LitboxDenoiserNet(
            upsample_factor=args.upsample, use_sigmoid=g_use_sigmoid, use_log_space=False,
            normalize_input=g_normalize_input, initial_features=g_initial_features,
            unet_size=g_unet_size, epsilon=g_epsilon, padding_mode=g_padding_mode).to(device)
        model.load_state_dict(torch.load(final_pth, map_location=device))
        model.export_onnx(os.path.join(args.output_folder, "final.onnx"), input_channels=8, resolution=args.crop_size)
        return

    if not args.skip_cache_validation:
        build_cache(args)
    stats = load_cached_stats(args)

    if args.eval:
        input_files = sorted(glob.glob(args.input_location))
        use_sigmoid = g_use_sigmoid
        model = LitboxDenoiserNet(
            upsample_factor=args.upsample, 
            use_sigmoid=use_sigmoid, 
            use_log_space=args.log_space, 
            normalize_input=g_normalize_input, 
            initial_features=g_initial_features,
            unet_size=g_unet_size, 
            epsilon=g_epsilon, 
            padding_mode=g_padding_mode).to(device)
        model.load_state_dict(torch.load(args.model_path))
        evaluate(model, args.input_location, args.output_folder, args, stats)
    else:
        if os.path.exists(args.output_folder):
            shutil.rmtree(args.output_folder)
        os.makedirs(args.output_folder, exist_ok=True)

        with open(os.path.join(args.output_folder, 'stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        train(args, stats)

if __name__ == "__main__":
    main() 
    exit(0)