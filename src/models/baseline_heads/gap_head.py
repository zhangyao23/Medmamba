import torch
import torch.nn as nn
from typing import Tuple, Optional


class GAPHead(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            bag_feature = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            bag_feature = h.mean(dim=1)

        logits = self.classifier(bag_feature)
        attention = torch.zeros(h.shape[0], h.shape[1], device=h.device)
        return logits, attention
