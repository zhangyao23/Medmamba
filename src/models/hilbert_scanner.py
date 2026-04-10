import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class HilbertCurveSpatialScanner(nn.Module):
    def __init__(self, max_order: int = 8):
        super().__init__()
        self.max_order = max_order
    
    def hilbert_index_3d(self, x: int, y: int, z: int, order: int) -> int:
        index = 0
        s = 1 << (order - 1)
        
        for i in range(order - 1, -1, -1):
            rx = 1 if (x & s) else 0
            ry = 1 if (y & s) else 0
            rz = 1 if (z & s) else 0
            
            index = (index << 3) | (rx << 2) | (ry << 1) | rz
            
            x, y, z = self.rotate(s, x, y, z, rx, ry, rz)
            s >>= 1
        
        return index
    
    def rotate(self, n: int, x: int, y: int, z: int, rx: int, ry: int, rz: int) -> Tuple[int, int, int]:
        if rz == 0:
            if ry == 0:
                x, y = y, x
            if rx == 1:
                x = n - 1 - x
                z = n - 1 - z
        
        return x, y, z
    
    def compute_hilbert_order(self, max_coord: int) -> int:
        order = 1
        while (1 << order) < max_coord:
            order += 1
        return min(order, self.max_order)
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = features.shape
        device = features.device
        
        sorted_features = torch.zeros_like(features)
        sorted_coords = torch.zeros_like(coords)
        perm_indices = torch.zeros(B, N, dtype=torch.long, device=device)
        
        for b in range(B):
            coords_np = coords[b].cpu().numpy().astype(np.int32)
            
            max_coord = coords_np.max() + 1
            order = self.compute_hilbert_order(max_coord)
            
            hilbert_indices = []
            for i in range(N):
                x, y, z = coords_np[i]
                idx = self.hilbert_index_3d(int(x), int(y), int(z), order)
                hilbert_indices.append(idx)
            
            hilbert_indices = np.array(hilbert_indices)
            sort_order = np.argsort(hilbert_indices)
            sort_order_tensor = torch.from_numpy(sort_order).to(device)
            
            sorted_features[b] = features[b, sort_order_tensor]
            sorted_coords[b] = coords[b, sort_order_tensor]
            perm_indices[b] = sort_order_tensor
        
        return sorted_features, sorted_coords, perm_indices
