import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from monai.transforms import Compose


CT_ORGAN_WINDOWS = {
    'Colon':    (-160, 240),
    'Kidney':   (-160, 240),
    'Liver':    (-160, 240),
    'Pancreas': (-160, 240),
    'Lung':     (-1100, 400),
}

MRI_ORGANS = {'Bladder', 'Breast', 'Cervix', 'Prostate', 'Uterus'}


def _detect_organ(volume_path: str) -> str:
    path_lower = volume_path.lower()
    for organ in list(CT_ORGAN_WINDOWS.keys()) + list(MRI_ORGANS):
        if organ.lower() in path_lower:
            return organ
    return ''


def _normalize_volume(volume: np.ndarray, volume_path: str,
                      min_hu: float, max_hu: float,
                      adaptive_norm: bool = True) -> np.ndarray:
    if not adaptive_norm:
        volume = np.clip(volume, min_hu, max_hu)
        return (volume - min_hu) / (max_hu - min_hu)

    organ = _detect_organ(volume_path)

    if organ in MRI_ORGANS:
        p_low = np.percentile(volume, 0.5)
        p_high = np.percentile(volume, 99.5)
        if p_high - p_low < 1e-6:
            p_high = p_low + 1.0
        volume = np.clip(volume, p_low, p_high)
        return (volume - p_low) / (p_high - p_low)

    if organ in CT_ORGAN_WINDOWS:
        win_min, win_max = CT_ORGAN_WINDOWS[organ]
    else:
        win_min, win_max = min_hu, max_hu

    volume = np.clip(volume, win_min, win_max)
    return (volume - win_min) / (win_max - win_min)


class VolumetricMILDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        stride: Tuple[int, int, int] = (16, 16, 16),
        transform: Optional[Compose] = None,
        min_hu: float = -1024.0,
        max_hu: float = 3071.0,
        max_patches: int = 128,
        healthy_only: bool = False,
        augment: bool = False,
        adaptive_norm: bool = True,
    ):
        with open(json_path, 'r') as f:
            self.entries = json.load(f)
        if healthy_only:
            self.entries = [entry for entry in self.entries if entry.get('label', 1) == 0]
        
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.min_hu = min_hu
        self.max_hu = max_hu
        self.max_patches = max_patches
        self.healthy_only = healthy_only
        self.augment = augment
        self.adaptive_norm = adaptive_norm
        
        logging.info(f"Loaded {len(self.entries)} volumes from {json_path}")
        logging.info(f"Patch size: {patch_size}, Stride: {stride}, Max patches: {max_patches}")
        logging.info(f"Adaptive normalization: {adaptive_norm}")
        if healthy_only:
            logging.info("Dataset mode: healthy_only=True")
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        
        volume_path = entry['image']
        label = entry['label']
        
        img = nib.load(volume_path)
        img = nib.as_closest_canonical(img)
        volume = img.get_fdata()
        
        if volume.ndim == 4:
            if volume.shape[-1] == 1:
                volume = volume[:, :, :, 0]
            else:
                volume = volume.mean(axis=-1)
        
        volume = _normalize_volume(volume, volume_path,
                                    self.min_hu, self.max_hu,
                                    self.adaptive_norm)
        
        H, W, D = volume.shape
        
        patches, coords = self._extract_patches(volume)
        
        if len(patches) == 0:
            logging.warning(f"No patches extracted from {volume_path}, creating dummy patch")
            patches = [np.zeros(self.patch_size, dtype=np.float32)]
            coords = [(0, 0, 0)]
        
        if len(patches) > self.max_patches:
            if self.augment:
                indices = np.random.choice(len(patches), self.max_patches, replace=False)
                indices.sort()
                patches = [patches[i] for i in indices]
                coords = [coords[i] for i in indices]
            else:
                patches = patches[:self.max_patches]
                coords = coords[:self.max_patches]
        
        # Ensure pure numpy array (sometimes mmap arrays or other types cause issues with torch.from_numpy)
        try:
            patches_tensor = torch.stack([torch.from_numpy(p).float() for p in patches], dim=0)
        except TypeError:
            # Fallback for "expected np.ndarray (got numpy.ndarray)" which usually happens with certain numpy versions or memory maps
            patches_tensor = torch.stack([torch.tensor(p).float() for p in patches], dim=0)

        flip_flags = torch.zeros(3, dtype=torch.bool)
        if self.augment:
            patches_tensor, flip_flags = self._augment_patches(patches_tensor)

        patches_tensor = patches_tensor.unsqueeze(1)
        
        coords_tensor = torch.tensor(coords, dtype=torch.long)
        
        return {
            'patches': patches_tensor,
            'coords': coords_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'volume_shape': torch.tensor([D, H, W], dtype=torch.long),
            'volume_path': volume_path,
            'flip_flags': flip_flags,
        }
    
    def _augment_patches(self, patches: torch.Tensor):
        N, D, H, W = patches.shape
        flip_d = np.random.rand() < 0.5
        flip_h = np.random.rand() < 0.5
        flip_w = np.random.rand() < 0.5
        if flip_d:
            patches = torch.flip(patches, dims=[1])
        if flip_h:
            patches = torch.flip(patches, dims=[2])
        if flip_w:
            patches = torch.flip(patches, dims=[3])
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            patches = patches * scale
            patches = torch.clamp(patches, 0.0, 1.0)
        if np.random.rand() < 0.2:
            noise = torch.randn_like(patches) * 0.02
            patches = patches + noise
            patches = torch.clamp(patches, 0.0, 1.0)
        flip_flags = torch.tensor([flip_d, flip_h, flip_w], dtype=torch.bool)
        return patches, flip_flags

    def _extract_patches(
        self,
        volume: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
        H, W, D = volume.shape
        patch_d, patch_h, patch_w = self.patch_size
        stride_d, stride_h, stride_w = self.stride
        
        patches = []
        coords = []
        
        for z in range(0, max(1, D - patch_d + 1), stride_d):
            z_end = min(z + patch_d, D)
            z_start = max(0, z_end - patch_d)
            
            for y in range(0, max(1, H - patch_h + 1), stride_h):
                y_end = min(y + patch_h, H)
                y_start = max(0, y_end - patch_h)
                
                for x in range(0, max(1, W - patch_w + 1), stride_w):
                    x_end = min(x + patch_w, W)
                    x_start = max(0, x_end - patch_w)
                    
                    patch = volume[y_start:y_end, x_start:x_end, z_start:z_end]
                    
                    if patch.shape != (patch_h, patch_w, patch_d):
                        patch_padded = np.zeros((patch_h, patch_w, patch_d), dtype=np.float32)
                        patch_padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                        patch = patch_padded
                    
                    patch = np.transpose(patch, (2, 0, 1))
                    
                    patches.append(patch)
                    coords.append((z_start, y_start, x_start))
        
        return patches, coords
