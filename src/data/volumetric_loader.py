from typing import Dict, Iterator, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler, Sampler
import torch.distributed as dist

from .volumetric_dataset import VolumetricMILDataset
from .balanced_batch_sampler import BalancedBatchSampler
from .phase_aware_sampler import PhaseAwareBatchSampler


class DistributedBatchSampler(Sampler[List[int]]):
    def __init__(self, base_batch_sampler: Sampler[List[int]], num_replicas: int, rank: int):
        self.base_batch_sampler = base_batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = list(self.base_batch_sampler)
        usable_total = (len(all_batches) // self.num_replicas) * self.num_replicas
        if usable_total == 0:
            return
        for batch_idx in range(self.rank, usable_total, self.num_replicas):
            yield all_batches[batch_idx]

    def __len__(self) -> int:
        base_len = len(self.base_batch_sampler)
        usable_total = (base_len // self.num_replicas) * self.num_replicas
        return usable_total // self.num_replicas

    def set_epoch(self, epoch: int):
        if hasattr(self.base_batch_sampler, 'set_epoch'):
            self.base_batch_sampler.set_epoch(epoch)

    def set_phase(self, phase):
        if hasattr(self.base_batch_sampler, 'set_phase'):
            self.base_batch_sampler.set_phase(phase)

    def set_phase2_normal_ratio(self, ratio: float):
        if hasattr(self.base_batch_sampler, 'set_phase2_normal_ratio'):
            self.base_batch_sampler.set_phase2_normal_ratio(ratio)


def collate_volumetric_mil(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_patches = max(b['patches'].shape[0] for b in batch)
    
    padded_patches = []
    masks = []
    coords_list = []
    labels = []
    volume_shapes = []
    volume_paths = []
    num_patches_list = []
    
    flip_flags_list = []

    for b in batch:
        n_patches = b['patches'].shape[0]
        pad_size = max_patches - n_patches
        
        if pad_size > 0:
            padded = F.pad(b['patches'], (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_size))
        else:
            padded = b['patches']
        
        mask = torch.cat([
            torch.ones(n_patches),
            torch.zeros(pad_size)
        ])
        
        coords_padded = F.pad(b['coords'], (0, 0, 0, pad_size), value=0)
        
        padded_patches.append(padded)
        masks.append(mask)
        coords_list.append(coords_padded)
        labels.append(b['label'])
        volume_shapes.append(b['volume_shape'])
        volume_paths.append(b['volume_path'])
        num_patches_list.append(torch.tensor(n_patches))
        flip_flags_list.append(b.get('flip_flags', torch.zeros(3, dtype=torch.bool)))
    
    return {
        'patches': torch.stack(padded_patches),
        'masks': torch.stack(masks),
        'coords': torch.stack(coords_list),
        'labels': torch.stack(labels),
        'volume_shapes': torch.stack(volume_shapes),
        'volume_paths': volume_paths,
        'num_patches': torch.stack(num_patches_list),
        'flip_flags': torch.stack(flip_flags_list),
    }


def create_volumetric_dataloader(
    json_path: str,
    batch_size: int,
    patch_size: tuple = (32, 32, 32),
    stride: tuple = (16, 16, 16),
    transforms: Optional = None,
    num_workers: int = 4,
    shuffle: bool = True,
    balanced_sampling: bool = True,
    positive_ratio: float = 0.2,
    max_patches: int = 128,
    use_ddp: bool = False,
    use_phase_aware: bool = False,
    is_phase1: bool = True,
    phase1_normal_ratio: float = 0.8,
    healthy_only: bool = False,
    **kwargs
) -> DataLoader:
    dataset = VolumetricMILDataset(
        json_path=json_path,
        patch_size=patch_size,
        stride=stride,
        transform=transforms,
        max_patches=max_patches,
        healthy_only=healthy_only,
        **kwargs
    )
    
    sampler = None
    batch_sampler = None
    
    if use_ddp and use_phase_aware and shuffle:
        labels = [entry['label'] for entry in dataset.entries]
        base_batch_sampler = PhaseAwareBatchSampler(
            labels=labels,
            batch_size=batch_size,
            phase1_normal_ratio=phase1_normal_ratio,
            is_phase1=is_phase1
        )
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        batch_sampler = DistributedBatchSampler(base_batch_sampler, world_size, rank)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_volumetric_mil,
            pin_memory=True
        )
    elif use_ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_volumetric_mil,
            pin_memory=True
        )
    elif (balanced_sampling or use_phase_aware) and shuffle:
        labels = [entry['label'] for entry in dataset.entries]
        
        if use_phase_aware:
            batch_sampler = PhaseAwareBatchSampler(
                labels=labels,
                batch_size=batch_size,
                phase1_normal_ratio=phase1_normal_ratio,
                is_phase1=is_phase1
            )
        else:
            batch_sampler = BalancedBatchSampler(labels=labels, batch_size=batch_size)
        
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_volumetric_mil,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_volumetric_mil,
            pin_memory=True
        )
    
    return dataloader
