import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class DSMILHead(nn.Module):
    """DSMIL (Li et al., 2021) - Dual-Stream MIL with instance and bag classifiers."""

    def __init__(self, input_dim: int = 512, num_classes: int = 2):
        super().__init__()
        self.i_classifier = nn.Sequential(
            nn.Linear(input_dim, num_classes),
        )
        self.b_classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes),
        )
        self.query_proj = nn.Linear(input_dim, 128)
        self.key_proj = nn.Linear(input_dim, 128)

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape

        instance_logits = self.i_classifier(h)

        instance_scores = instance_logits[:, :, 1] - instance_logits[:, :, 0]
        if mask is not None:
            instance_scores = instance_scores.masked_fill(mask == 0, -1e4)

        topk_k = max(1, min(N // 8, 8))
        _, topk_indices = torch.topk(instance_scores, topk_k, dim=1)

        topk_features = torch.gather(
            h, 1, topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        critical = topk_features.mean(dim=1)

        query = self.query_proj(critical).unsqueeze(1)
        keys = self.key_proj(h)
        att_logits = (query * keys).sum(dim=-1) / (128 ** 0.5)

        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -1e4)

        attention = torch.softmax(att_logits, dim=1)
        bag_feature = (attention.unsqueeze(-1) * h).sum(dim=1)

        bag_logits = self.b_classifier(bag_feature)

        instance_max_logits = instance_logits[
            torch.arange(B, device=h.device),
            topk_indices[:, 0]
        ]
        logits = 0.5 * bag_logits + 0.5 * instance_max_logits

        return logits, attention
