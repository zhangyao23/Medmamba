import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from monai.transforms import Compose


class Volume3DMILDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        transform: Optional[Compose] = None,
        max_slices: int = 64,
        skip_empty_slices: bool = True,
        empty_threshold: float = 100.0,
    ):
        with open(json_path, 'r') as f:
            self.entries = json.load(f)
        
        self.transform = transform
        self.max_slices = max_slices
        self.skip_empty_slices = skip_empty_slices
        self.empty_threshold = empty_threshold
        
        logging.info(f"Loaded {len(self.entries)} samples from {json_path}")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        
        volume_path = entry['image']
        label = entry['label']
        
        volume = nib.load(volume_path).get_fdata()
        
        if volume.ndim == 3:
            H, W, D = volume.shape
        elif volume.ndim == 4:
            H, W, D, C = volume.shape
            if C == 1:
                volume = volume[:, :, :, 0]
            else:
                volume = volume.mean(axis=-1)
        
        slices = []
        z_indices = []
        
        for z in range(D):
            slice_2d = volume[:, :, z]
            
            if self.skip_empty_slices:
                non_background = np.sum(slice_2d > -500)
                if non_background < self.empty_threshold:
                    continue
            
            if slice_2d.max() > 0:
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            
            slice_2d = slice_2d.astype(np.float32)
            
            slices.append(slice_2d)
            z_indices.append(z)
        
        if len(slices) == 0:
            logging.warning(f"No valid slices found in {volume_path}, using center slice")
            center_z = D // 2
            slice_2d = volume[:, :, center_z].astype(np.float32)
            if slice_2d.max() > 0:
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
            slices = [slice_2d]
            z_indices = [center_z]
        
        if len(slices) > self.max_slices:
            indices = np.linspace(0, len(slices) - 1, self.max_slices, dtype=int)
            slices = [slices[i] for i in indices]
            z_indices = [z_indices[i] for i in indices]
        
        slices_tensor = []
        for s in slices:
            s_tensor = torch.from_numpy(s).unsqueeze(0)
            
            if s_tensor.shape[1:] != (224, 224):
                s_tensor = torch.nn.functional.interpolate(
                    s_tensor.unsqueeze(0),
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            if s_tensor.shape[0] == 1:
                s_tensor = s_tensor.repeat(3, 1, 1)
            
            if self.transform:
                s_tensor = self.transform(s_tensor)
            
            slices_tensor.append(s_tensor)
        
        slices_tensor = torch.stack(slices_tensor)
        
        return {
            'slices': slices_tensor,
            'z_indices': torch.tensor(z_indices, dtype=torch.long),
            'bag_label': torch.tensor(label, dtype=torch.long),
            'volume_id': volume_path,
            'num_slices': torch.tensor(len(slices), dtype=torch.long)
        }
