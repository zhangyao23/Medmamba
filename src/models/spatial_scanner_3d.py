import torch
import torch.nn as nn
from typing import Optional, Tuple


def invert_permutation(sort_perm: torch.Tensor) -> torch.Tensor:
    return sort_perm.argsort(dim=1)


def reorder_sequence(
    sequence: Optional[torch.Tensor],
    sort_perm: torch.Tensor
) -> Optional[torch.Tensor]:
    if sequence is None:
        return None

    if sequence.dim() < 2:
        raise ValueError("sequence must have at least 2 dimensions")

    index = sort_perm
    while index.dim() < sequence.dim():
        index = index.unsqueeze(-1)
    index = index.expand_as(sequence)
    return torch.gather(sequence, 1, index)


def restore_sequence_order(
    sequence: Optional[torch.Tensor],
    sort_perm: torch.Tensor
) -> Optional[torch.Tensor]:
    if sequence is None:
        return None
    return reorder_sequence(sequence, invert_permutation(sort_perm))


class ZOrderSpatialScanner(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = features.shape
        
        sorted_features_list = []
        sorted_coords_list = []
        sort_indices_list = []
        
        for b in range(B):
            feat_b = features[b]
            coord_b = coords[b]
            
            z_order_indices = self._compute_z_order_indices(coord_b)
            
            sort_indices = torch.argsort(z_order_indices)
            
            sorted_feat = feat_b[sort_indices]
            sorted_coord = coord_b[sort_indices]
            
            sorted_features_list.append(sorted_feat)
            sorted_coords_list.append(sorted_coord)
            sort_indices_list.append(sort_indices)
        
        sorted_features = torch.stack(sorted_features_list, dim=0)
        sorted_coords = torch.stack(sorted_coords_list, dim=0)
        perm_indices = torch.stack(sort_indices_list, dim=0)
        
        return sorted_features, sorted_coords, perm_indices
    
    def _compute_z_order_indices(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords[:, 0]
        y = coords[:, 1]
        x = coords[:, 2]
        
        max_val = max(z.max().item(), y.max().item(), x.max().item())
        num_bits = max_val.bit_length()
        
        z_order = torch.zeros_like(z, dtype=torch.long)
        
        for i in range(num_bits):
            z_bit = (z >> i) & 1
            y_bit = (y >> i) & 1
            x_bit = (x >> i) & 1
            
            z_order |= (z_bit << (3 * i + 2))
            z_order |= (y_bit << (3 * i + 1))
            z_order |= (x_bit << (3 * i))
        
        return z_order


class HilbertSpatialScanner(nn.Module):
    def __init__(self, order: int = 5):
        super().__init__()
        self.order = order
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = features.shape
        
        sorted_features_list = []
        sorted_coords_list = []
        sort_indices_list = []
        
        for b in range(B):
            feat_b = features[b]
            coord_b = coords[b]
            
            hilbert_indices = self._compute_hilbert_indices(coord_b)
            
            sort_indices = torch.argsort(hilbert_indices)
            
            sorted_feat = feat_b[sort_indices]
            sorted_coord = coord_b[sort_indices]
            
            sorted_features_list.append(sorted_feat)
            sorted_coords_list.append(sorted_coord)
            sort_indices_list.append(sort_indices)
        
        sorted_features = torch.stack(sorted_features_list, dim=0)
        sorted_coords = torch.stack(sorted_coords_list, dim=0)
        perm_indices = torch.stack(sort_indices_list, dim=0)
        
        return sorted_features, sorted_coords, perm_indices
    
    def _compute_hilbert_indices(self, coords: torch.Tensor) -> torch.Tensor:
        hilbert_indices = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        
        for i in range(coords.shape[0]):
            z, y, x = coords[i].tolist()
            hilbert_indices[i] = self._xyz_to_hilbert(x, y, z, self.order)
        
        return hilbert_indices
    
    def _xyz_to_hilbert(self, x: int, y: int, z: int, order: int) -> int:
        hilbert_index = 0
        
        for i in range(order - 1, -1, -1):
            rx = (x >> i) & 1
            ry = (y >> i) & 1
            rz = (z >> i) & 1
            
            hilbert_index = (hilbert_index << 3) | (rx << 2) | (ry << 1) | rz
        
        return hilbert_index
