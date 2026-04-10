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
    
    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self.attention(h).squeeze(-1)
        
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e4)
        
        A = torch.softmax(A, dim=1)
        
        bag_feature = (A.unsqueeze(-1) * h).sum(dim=1)
        logits = self.classifier(bag_feature)
        
        return logits, A
