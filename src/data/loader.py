from typing import Dict, List, Optional
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import Volume3DMILDataset


def collate_3d_mil(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_slices = max(b['slices'].shape[0] for b in batch)
    
    padded_slices = []
    masks = []
    z_indices_list = []
    labels = []
    volume_ids = []
    num_slices_list = []
    
    for b in batch:
        n_slices = b['slices'].shape[0]
        pad_size = max_slices - n_slices
        
        padded = F.pad(b['slices'], (0, 0, 0, 0, 0, 0, 0, pad_size))
        mask = torch.cat([
            torch.ones(n_slices),
            torch.zeros(pad_size)
        ])
        
        z_padded = F.pad(b['z_indices'], (0, pad_size), value=0)
        
        padded_slices.append(padded)
        masks.append(mask)
        z_indices_list.append(z_padded)
        labels.append(b['bag_label'])
        volume_ids.append(b['volume_id'])
        num_slices_list.append(b['num_slices'])
    
    return {
        'slices': torch.stack(padded_slices),
        'masks': torch.stack(masks),
        'z_indices': torch.stack(z_indices_list),
        'labels': torch.stack(labels),
        'volume_ids': volume_ids,
        'num_slices': torch.stack(num_slices_list)
    }


def create_mil_dataloader(
    json_path: str,
    batch_size: int,
    transforms: Optional = None,
    num_workers: int = 4,
    shuffle: bool = True,
    max_slices: int = 64,
    balanced_sampling: bool = False,
    **kwargs
) -> DataLoader:
    dataset = Volume3DMILDataset(
        json_path=json_path,
        transform=transforms,
        max_slices=max_slices,
        **kwargs
    )
    
    sampler = None
    if balanced_sampling and shuffle:
        with open(json_path) as f:
            entries = json.load(f)
        labels = [entry['label'] for entry in entries]
        
        class_counts = [labels.count(0), labels.count(1)]
        class_weights = [1.0 / count for count in class_counts]
        sample_weights = [class_weights[label] for label in labels]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_3d_mil,
        pin_memory=True
    )
    
    return dataloader
