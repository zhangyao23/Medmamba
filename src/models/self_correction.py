import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SelfCorrectionModule(nn.Module):
    def __init__(
        self,
        entropy_threshold: float = 0.7,
        neighbor_radius: int = 2,
        spatial_weight: float = 0.3
    ):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.neighbor_radius = neighbor_radius
        self.spatial_weight = spatial_weight
    
    def compute_prediction_uncertainty(
        self,
        attention: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        uncertainty = 1.0 - attention
        
        if mask is not None:
            uncertainty = uncertainty * mask
        
        return uncertainty
    
    def get_spatial_neighbors(
        self,
        coords: torch.Tensor,
        radius: int = 2
    ) -> torch.Tensor:
        B, N, _ = coords.shape
        
        coords_expanded = coords.unsqueeze(2)
        coords_tiled = coords.unsqueeze(1)
        
        distances = torch.cdist(coords_expanded.float(), coords_tiled.float(), p=2)
        distances = distances.squeeze(2)
        
        neighbor_mask = (distances <= radius) & (distances > 0)
        
        return neighbor_mask
    
    def correct_with_neighbors(
        self,
        codes: torch.Tensor,
        coords: torch.Tensor,
        attention: torch.Tensor,
        uncertainty: torch.Tensor,
        codebook_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = codes.shape
        device = codes.device
        
        uncertain_mask = uncertainty > self.entropy_threshold
        
        if not uncertain_mask.any():
            return codes, attention
        
        corrected_codes = codes.clone()
        corrected_attention = attention.clone()
        
        for b in range(B):
            if not uncertain_mask[b].any():
                continue
            
            neighbor_mask = self.get_spatial_neighbors(
                coords[b:b+1],
                radius=self.neighbor_radius
            )
            
            uncertain_indices = torch.where(uncertain_mask[b])[0]
            
            for idx in uncertain_indices:
                neighbors = neighbor_mask[0, idx]
                
                if mask is not None:
                    neighbors = neighbors & mask[b].bool()
                
                if not neighbors.any():
                    continue
                
                neighbor_codes = codes[b, neighbors]
                
                mode_code = torch.mode(neighbor_codes)[0]
                
                corrected_codes[b, idx] = mode_code
                
                neighbor_attention = attention[b, neighbors]
                corrected_attention[b, idx] = neighbor_attention.mean()
        
        return corrected_codes, corrected_attention
    
    def forward(
        self,
        codes: torch.Tensor,
        coords: torch.Tensor,
        attention: torch.Tensor,
        codebook_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = self.compute_prediction_uncertainty(attention, mask)
        
        corrected_codes, corrected_attention = self.correct_with_neighbors(
            codes,
            coords,
            attention,
            uncertainty,
            codebook_embeddings,
            mask
        )
        
        correction_rate = (corrected_codes != codes).float().mean()
        
        return corrected_codes, corrected_attention, correction_rate
