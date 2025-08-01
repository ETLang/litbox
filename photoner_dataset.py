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
            input_tensor = TF.rotate(input_tensor, chosen_angle)
            target_tensor = TF.rotate(target_tensor, chosen_angle)
            
            return input_tensor, target_tensor
        
        return input_tensor, None
