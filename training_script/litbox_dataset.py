import array
import random
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import OpenEXR
import Imath
from data_processing import load_image

class LitboxDenoiserDataset(Dataset):
    def __init__(self,
                 input_a_paths: list, input_b_paths: list,
                 albedo_paths: list, density_paths: list, reference_paths: Optional[list] = None,
                 do_augment: bool = False,
                 crop_size: int = 64, upsample: int = 1, 
                 truth_transform=None):
        self.input_a_paths = input_a_paths
        self.input_b_paths = input_b_paths
        self.albedo_paths = albedo_paths
        self.density_paths = density_paths
        self.reference_paths = reference_paths
        self.crop_size = crop_size
        self.upsample = upsample
        self.truth_transform = truth_transform

        if len(self.input_a_paths) > 0:
            test_path = self.input_a_paths[0]
            
            # Check for EXR files or something else (SRGB-based)
            if test_path.lower().endswith('.exr'):
                self.exr_source = True
            else:
                self.exr_source = False
        
    def __len__(self):
        return len(self.input_a_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_a_path = self.input_a_paths[idx]
        input_b_path = self.input_b_paths[idx]
        albedo_path = self.albedo_paths[idx]
        density_path = self.density_paths[idx]
        
        # Load input image
        input_a_tensor = load_image(input_a_path)
        input_b_tensor = load_image(input_b_path)
        albedo_tensor = load_image(albedo_path)[:3]
        density_tensor = load_image(density_path)[:1]
        reference_tensor = None
        
        # Verify dimensions
        h_w_a = input_a_tensor.shape[1:]
        h_w_b = input_b_tensor.shape[1:]
        h_w_albedo = albedo_tensor.shape[1:]
        h_w_density = density_tensor.shape[1:]
        if not (h_w_a == h_w_b == h_w_albedo == h_w_density):
            raise ValueError(f"Height and width mismatch among input images at index {idx}")
        
        if self.reference_paths is not None:
            reference_path = self.reference_paths[idx]
            reference_tensor = load_image(reference_path)
            if not (reference_tensor.shape[1:] == h_w_a):
                raise ValueError(f"Height and width mismatch between input and reference images at index {idx}")
            
        return input_a_tensor, input_b_tensor, albedo_tensor, density_tensor, reference_tensor

# Initialize LitboxDataset with a dictionary of paths. Retrieved items will match the input dictionary keys.
class LitboxDataset(Dataset):
    def __init__(self, path_set: dict, device: torch.device = None):
        self.path_set = path_set
        self.device = device


    def __len__(self):
        return min(len(paths[0]) for paths in self.path_set.values())
    
    def __getitem__(self, idx: int) -> dict:
        output = {}

        for key, paths in self.path_set.items():
            if self.device is None:
                output[key] = load_image(paths[0][idx])[0:paths[1], :, :]
            else:
                output[key] = load_image(paths[0][idx])[0:paths[1], :, :].clone().detach().to(self.device)

        # Verify all images have matching dimensions
        if len(output) > 1:
            first_shape = output[next(iter(output))].shape[1:]
            for key, tensor in output.items():
                if tensor.shape[1:] != first_shape:
                    raise ValueError(f"Dimension mismatch at index {idx}: key '{key}' has shape {tensor.shape[1:]}, expected {first_shape}")
        
        return output
