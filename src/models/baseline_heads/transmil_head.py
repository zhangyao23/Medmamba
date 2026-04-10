import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PPEG(nn.Module):
    """Pyramid Position Encoding Generator from TransMIL."""

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv1d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv1d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x_t = x.transpose(1, 2)
        x_out = self.proj(x_t) + self.proj1(x_t) + self.proj2(x_t)
        return x + x_out.transpose(1, 2)


class TransMILHead(nn.Module):
    """TransMIL (Shao et al., 2021) - Transformer MIL with PPEG."""

    def __init__(self, input_dim: int = 512, num_classes: int = 2, num_layers: int = 2, nhead: int = 8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))
        self.ppeg = PPEG(dim=input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=input_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = h.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, h], dim=1)

        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            key_padding_mask = (full_mask == 0)
        else:
            key_padding_mask = None

        x = self.ppeg(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.classifier(cls_out)

        instance_features = x[:, 1:]
        att_logits = (instance_features * cls_out.unsqueeze(1)).sum(dim=-1)
        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -1e4)
        attention = torch.softmax(att_logits, dim=1)

        return logits, attention
