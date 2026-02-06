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

class PhotonerDataset(Dataset):
    def __init__(self,
                 input_a_paths: list, input_b_paths: list,
                 albedo_paths: list, transmissibility_paths: list, reference_paths: Optional[list] = None, 
                 crop_size: int = 64, upsample: int = 1,
                 truth_transform=None):
        self.input_a_paths = input_a_paths
        self.input_b_paths = input_b_paths
        self.albedo_paths = albedo_paths
        self.transmissibility_paths = transmissibility_paths
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
    
    def load_image(self, path: str) -> torch.Tensor:
        if path.lower().endswith('.exr'):
            return self.load_exr(path)
        else:
            return self.load_srgb(path)
        
    # Performs strided, jittered subsampling to create 'honest' low-res input.
    def jittered_subsample(image, upsample_factor):
        _, h, w = image.shape
        
        new_h, new_w = h // upsample_factor, w // upsample_factor
        
        offset_y = torch.randint(0, upsample_factor, (1,)).item()
        offset_x = torch.randint(0, upsample_factor, (1,)).item()
        
        low_res = image[:, offset_y::upsample_factor, offset_x::upsample_factor]
        
        # 3. Crop to ensure dimensions match exactly (safety for non-multiples)
        return low_res[:, :new_h, :new_w]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        input_a_path = self.input_a_paths[idx]
        input_b_path = self.input_b_paths[idx]
        albedo_path = self.albedo_paths[idx]
        transmissibility_path = self.transmissibility_paths[idx]
        
        # Load input image
        input_a_tensor = self.load_image(input_a_path)
        input_b_tensor = self.load_image(input_b_path)
        albedo_tensor = self.load_image(albedo_path)
        transmissibility_tensor = self.load_image(transmissibility_path)
        reference_tensor = None

        # Verify dimensions
        h_w_a = input_a_tensor.shape[1:]
        h_w_b = input_b_tensor.shape[1:]
        h_w_albedo = albedo_tensor.shape[1:]
        h_w_trans = transmissibility_tensor.shape[1:]
        if not (h_w_a == h_w_b == h_w_albedo == h_w_trans):
            raise ValueError(f"Height and width mismatch among input images at index {idx}")
        
        training_mode = self.reference_paths is not None

        if training_mode:
            # Load reference image
            reference_path = self.reference_paths[idx]
            reference_tensor = self.load_image(reference_path)
            if not (reference_tensor.shape[1:] == h_w_a):
                raise ValueError(f"Height and width mismatch between input and reference images at index {idx}")

            # Random crop
            if h_w_a[0] < self.crop_size or h_w_a[1] < self.crop_size:
                raise ValueError(f"Imageset {idx} is smaller than crop size {self.crop_size}")
            top = torch.randint(0, h_w_a[0] - self.crop_size + 1, (1,)).item()
            left = torch.randint(0, h_w_a[1] - self.crop_size + 1, (1,)).item()

            input_a_tensor = input_a_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            input_b_tensor = input_b_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            albedo_tensor = albedo_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            transmissibility_tensor = transmissibility_tensor[:, top:top+self.crop_size, left:left+self.crop_size]
            reference_tensor = reference_tensor[:, top:top+self.crop_size, left:left+self.crop_size]

            # Squish inputs to demonstrate upsampling for training
            if self.upsample > 1:
                # TODO: input tensors might not need the jittered subsampling. area resizing could work better due to how they are generated.
                input_a_tensor = self.jittered_subsample(input_a_tensor, self.upsample)
                input_b_tensor = self.jittered_subsample(input_b_tensor, self.upsample)
                albedo_tensor = self.jittered_subsample(albedo_tensor, self.upsample)
                transmissibility_tensor = self.jittered_subsample(transmissibility_tensor, self.upsample)
                transmissibility_tensor = transmissibility_tensor ** self.upsample
            
            # Random rotation to remove alignment bias
            angles = [0, 90, 180, 270]
            chosen_angle = random.choice(angles)
            input_a_tensor = TF.rotate(input_a_tensor, chosen_angle)
            input_b_tensor = TF.rotate(input_b_tensor, chosen_angle)
            albedo_tensor = TF.rotate(albedo_tensor, chosen_angle)
            transmissibility_tensor = TF.rotate(transmissibility_tensor, chosen_angle)
            reference_tensor = TF.rotate(reference_tensor, chosen_angle)

            # Customizable transformer for reference images
            if self.truth_transform:
                reference_tensor = self.truth_transform(reference_tensor)
            
        return input_a_tensor, input_b_tensor, albedo_tensor, transmissibility_tensor, reference_tensor
