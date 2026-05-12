import array
import glob
import random
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
from compute_runner import ComputeShaderRunner
import torchvision.transforms.functional as TF
from dataclasses import dataclass
from typing import List
import OpenEXR
import Imath
from PIL import Image

g_sigma_spatial = 1.2
g_sigma_albedo = 0.05
g_sigma_luminance_tight = 0.05
g_sigma_luminance_loose = 2.5
g_k_luminance = 2.0

compute_runner = ComputeShaderRunner()

def register_dataset_args(parser, require_ref_location):
    parser.add_argument('--input-a-easy-location', help='Path to easy input imageset A, for curriculum training')
    parser.add_argument('--input-b-easy-location', help='Path to easy input imageset B, for curriculum training')
    parser.add_argument('--input-a-medium-location', help='Path to input imageset A, for curriculum training')
    parser.add_argument('--input-b-medium-location', help='Path to input imageset B, for curriculum training')
    parser.add_argument('--input-a-location', help='Path to input imageset A')
    parser.add_argument('--input-b-location', help='Path to input imageset B')
    parser.add_argument('--input-albedo-location', help='Path to albedo imageset')
    parser.add_argument('--input-transmissibility-location', help='Path to transmissibility imageset')
    parser.add_argument('--reference-location', required=require_ref_location, help='Path to reference images for training')

def validate_dataset_args(args, parser):
    if args.input_a_easy_location and not args.input_b_easy_location:
        parser.error("Both --input-a-easy-location and --input-b-easy-location must be provided for easy curriculum training")
    if args.input_b_easy_location and not args.input_a_easy_location:
        parser.error("Both --input-a-easy-location and --input-b-easy-location must be provided for easy curriculum training")
    if args.input_a_medium_location and not args.input_b_medium_location:
        parser.error("Both --input-a-medium-location and --input-b-medium-location must be provided for medium curriculum training")
    if args.input_b_medium_location and not args.input_a_medium_location:
        parser.error("Both --input-a-medium-location and --input-b-medium-location must be provided for medium curriculum training")

def load_exr(path: str) -> torch.Tensor:
    file = OpenEXR.InputFile(path)
    dw = file.header()['dataWindow']
    size = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    # Read all channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = []

    if len(file.header()['channels']) == 1:
        channel_selection = [ 'Y' ]
    else:
        channel_selection = [ 'R', 'G', 'B' ]

    for channel in channel_selection:
        str_ch = file.channel(channel, FLOAT)
        ch = np.array(array.array('f', str_ch))
        ch = ch.reshape(size)
        channels.append(ch)
    
    # Stack channels and convert to tensor
    img = np.stack(channels, axis=0)
    tensor = torch.from_numpy(img).float()
    
    return tensor

def save_exr(path: str, tensor: torch.Tensor):
    img_data = tensor.detach().cpu().numpy().astype(np.float32)

    if img_data.ndim == 2:  # Single channel
        img_data = img_data[np.newaxis, :, :]

    channels, height, width = img_data.shape
    header = OpenEXR.Header(width, height)
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    if channels == 3:
        channel_names = ['R', 'G', 'B']
    elif channels == 1:
        channel_names = ['Y'] # Standard for luminance/single channel
    else:
        channel_names = [f'C{i}' for i in range(channels)]

    header['channels'] = {name: Imath.Channel(pixel_type) for name in channel_names}
    exr_file = OpenEXR.OutputFile(path, header)

    channel_data = {}
    for i, name in enumerate(channel_names):
        channel_data[name] = img_data[i].tobytes()
    exr_file.writePixels(channel_data)
    exr_file.close()

def load_srgb(path: str) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    tensor = transforms.ToTensor()(img)
    # Convert from sRGB to linear space
    tensor = tensor.pow(2.2)
    return tensor

def save_srgb(path: str, tensor: torch.Tensor):
    # TODO I dunno what this even does...
    tensor = tensor.pow(1/2.2)
    tensor = (tensor * 255).byte()
    img = transforms.ToPILImage()(tensor)
    img.save(path)

def load_image(path: str) -> torch.Tensor:
    if path.lower().endswith('.exr'):
        return load_exr(path)
    else:
        return load_srgb(path)
    
def save_image(path: str, tensor: torch.Tensor):
    if path.lower().endswith('.exr'):
        save_exr(path, tensor)
    else:
        save_srgb(path, tensor)
    
def get_dataset_files(reference_location,
                      **feature_locations):
    reference_files = sorted(glob.glob(reference_location))
    count = len(reference_files)

    def get_constrained(path, type_str):
        if path is None:
            return None
        files = sorted(glob.glob(path))
        if len(files) < count:
            raise ValueError("There are fewer " + type_str + " files than reference files. Each reference file must have a corresponding " + type_str + " file.")
        return files[:count]

    output = {}
    output['reference'] = reference_files

    for name, location in feature_locations.items():
        if location is not None:
            output[name] = get_constrained(location, name)
    return output

def compute_mean_and_relative_variance(image_a, image_b, albedo):
    mean = (image_a + image_b) / 2.0
    relative_variance = compute_runner.compute_variance(image_a, image_b, albedo)
    return mean, relative_variance

def preprocess_radiance(radiance_a, radiance_b, albedo, mean: torch.Tensor = None, stddev: torch.Tensor = None):
    (radiance, rel_variance) = compute_mean_and_relative_variance(radiance_a, radiance_b, albedo)
    radiance = torch.log10(radiance + 1e-6)
    rel_variance = torch.log10(rel_variance + 1e-6)

    if mean is not None and stddev is not None:
        mean_view = mean.view(1,3,1,1)
        stddev_view = stddev.view(1,3,1,1)
        radiance = (radiance - mean_view) / (stddev_view + 1e-6)
        rel_variance = (rel_variance - mean_view) / (stddev_view + 1e-6)
    return radiance, rel_variance

def preprocess_transmissibility(transmissibility, mean: torch.Tensor = None, stddev: torch.Tensor = None):
    density = -torch.log10(1 - torch.clamp(transmissibility, min=0, max=1-1e-6))

    if mean is not None and stddev is not None:
        mean_view = mean.view(1,1,1,1)
        stddev_view = stddev.view(1,1,1,1)
        density = (density - mean_view) / (stddev_view + 1e-6)
    return density

def preprocess_reference(reference, mean: torch.Tensor = None, stddev: torch.Tensor = None):
    reference = torch.log10(reference + 1e-6)

    if mean is not None and stddev is not None:
        mean_view = mean.view(1,3,1,1)
        stddev_view = stddev.view(1,3,1,1)
        reference = (reference - mean_view) / (stddev_view + 1e-6)
    return reference

def preprocess_albedo(albedo):
    return albedo  # Nothing to do for now

def postprocess_inference(inferred, mean, stddev):
    mean_view = mean.view(1,3,1,1)
    stddev_view = stddev.view(1,3,1,1)
    return torch.exp10(inferred * (stddev_view + 1e-6) + mean_view)

# Performs strided, jittered subsampling to create 'honest' low-res input.
def jittered_subsample(image, upsample_factor):
    _, _, h, w = image.shape
    
    new_h, new_w = h // upsample_factor, w // upsample_factor
    
    offset_y = torch.randint(0, upsample_factor, (1,)).item()
    offset_x = torch.randint(0, upsample_factor, (1,)).item()
    
    low_res = image[:, :, offset_y::upsample_factor, offset_x::upsample_factor]
    
    # 3. Crop to ensure dimensions match exactly (safety for non-multiples)
    return low_res[:, :, :new_h, :new_w]

def augment_for_training(radiance, variance, albedo, transmissibility, reference, upsample_factor: int = 1, crop_size: int = 64):
    height, width = radiance.shape[2:]

    # Random crop
    if height < crop_size or width < crop_size:
        raise ValueError(f"Imageset is smaller than crop size {crop_size}")
    top = torch.randint(0, height - crop_size + 1, (1,)).item()
    left = torch.randint(0, width - crop_size + 1, (1,)).item()

    radiance = radiance[:, top:top+crop_size, left:left+crop_size]
    variance = variance[:, top:top+crop_size, left:left+crop_size]
    albedo = albedo[:, top:top+crop_size, left:left+crop_size]
    transmissibility = transmissibility[:, top:top+crop_size, left:left+crop_size]
    reference = reference[:, top:top+crop_size, left:left+crop_size]

    # Squish inputs to demonstrate upsampling for training
    if upsample_factor > 1:
        # TODO: input tensors might not need the jittered subsampling. area resizing could work better due to how they are generated.
        radiance = jittered_subsample(radiance, upsample_factor)
        variance = jittered_subsample(variance, upsample_factor)
        albedo = jittered_subsample(albedo, upsample_factor)
        transmissibility = jittered_subsample(transmissibility, upsample_factor)
        transmissibility = transmissibility * upsample_factor  #transmissibility should already be in -log10 space
    
    # Random rotation to remove alignment bias
    angles = [0, 90, 180, 270]
    chosen_angle = random.choice(angles)
    radiance = TF.rotate(radiance, chosen_angle)
    variance = TF.rotate(variance, chosen_angle)
    albedo_tensor = TF.rotate(albedo_tensor, chosen_angle)
    transmissibility = TF.rotate(transmissibility, chosen_angle)
    reference = TF.rotate(reference, chosen_angle)

def check_tensor_fidelity(original_tensor, reprocessed_tensor):
    # Ensure both are on CPU and NumPy
    orig = original_tensor.detach().cpu().numpy()
    rel = reprocessed_tensor.detach().cpu().numpy()

    # 1. Max Absolute Error: The largest difference in a single pixel
    max_diff = np.max(np.abs(orig - rel))
    
    # 2. Mean Squared Error: General drift across the whole image
    mse = np.mean((orig - rel) ** 2)
    
    # 3. Bit-Perfect Check: Are they exactly identical?
    is_identical = np.array_equal(orig, rel)

    print(f"--- Fidelity Report ---")
    print(f"Exactly Identical: {is_identical}")
    print(f"Max Absolute Diff: {max_diff:.10f}")
    print(f"Mean Squared Error: {mse:.10e}")
    
    if not is_identical:
        # Check if the error is just floating point jitter (1e-7 is standard for float32)
        if np.allclose(orig, rel, atol=1e-7):
            print("Status: Minor floating-point jitter detected (Safe for AI).")
        else:
            print("Status: WARNING! Significant data loss detected.")