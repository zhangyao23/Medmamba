import torch
import torch.nn as nn
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
        attention_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = torch.sigmoid(attention_logits)
        entropy = -(probs * torch.log(probs.clamp(min=1e-8)) +
                    (1.0 - probs) * torch.log((1.0 - probs).clamp(min=1e-8)))
        uncertainty = entropy / torch.log(torch.tensor(2.0, device=attention_logits.device))

        if mask is not None:
            uncertainty = uncertainty * mask
        
        return uncertainty

    def normalize_attention(
        self,
        attention_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask == 0, -1e4)

        attention = torch.softmax(attention_logits, dim=1)
        if mask is None:
            return attention

        mask_f = mask.float()
        attention = attention * mask_f
        denom = attention.sum(dim=1, keepdim=True)
        return torch.where(
            denom > 0,
            attention / denom.clamp(min=1e-8),
            torch.zeros_like(attention),
        )
    
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
        attention_logits: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N = codes.shape
        
        uncertain_mask = uncertainty > self.entropy_threshold
        
        if not uncertain_mask.any():
            return codes, self.normalize_attention(attention_logits, mask=mask)
        
        corrected_codes = codes.clone()
        corrected_logits = attention_logits.clone()
        
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

                neighbor_logit = attention_logits[b, neighbors].mean()
                corrected_logits[b, idx] = (
                    (1.0 - self.spatial_weight) * attention_logits[b, idx] +
                    self.spatial_weight * neighbor_logit
                )

        corrected_attention = self.normalize_attention(corrected_logits, mask=mask)
        return corrected_codes, corrected_attention
    
    def forward(
        self,
        codes: torch.Tensor,
        coords: torch.Tensor,
        attention_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uncertainty = self.compute_prediction_uncertainty(attention_logits, mask)
        
        corrected_codes, corrected_attention = self.correct_with_neighbors(
            codes,
            coords,
            attention_logits,
            uncertainty,
            mask
        )

        if mask is not None:
            correction_rate = (
                ((corrected_codes != codes).float() * mask.float()).sum() /
                mask.float().sum().clamp(min=1.0)
            )
        else:
            correction_rate = (corrected_codes != codes).float().mean()
        
        return corrected_codes, corrected_attention, correction_rate
