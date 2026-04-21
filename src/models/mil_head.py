import torch
import torch.nn as nn
from typing import Tuple, Optional


class AttentionMILHead(nn.Module):
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Linear(input_dim, num_classes)

    def compute_attention_logits(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        att_logits = self.attention(h).squeeze(-1)
        if mask is not None:
            att_logits = att_logits.masked_fill(mask == 0, -1e4)
        return att_logits

    def normalize_attention(
        self,
        attention_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention = torch.softmax(attention_logits, dim=1)
        if mask is None:
            return attention

        mask_f = mask.float()
        attention = attention * mask_f
        denom = attention.sum(dim=1, keepdim=True)
        attention = torch.where(
            denom > 0,
            attention / denom.clamp(min=1e-8),
            torch.zeros_like(attention),
        )
        return attention

    def classify_with_attention(
        self,
        h: torch.Tensor,
        attention: torch.Tensor
    ) -> torch.Tensor:
        bag_feature = (attention.unsqueeze(-1) * h).sum(dim=1)
        return self.classifier(bag_feature)
    
    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.compute_attention_logits(h, mask=mask)
        attention = self.normalize_attention(attention_logits, mask=mask)
        logits = self.classify_with_attention(h, attention)
        return logits, attention
