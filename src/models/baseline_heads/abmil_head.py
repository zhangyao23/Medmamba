import torch
import torch.nn as nn
from typing import Tuple, Optional


class ABMILHead(nn.Module):
    """Attention-Based MIL (Ilse et al., 2018) with gated attention."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A_V = self.attention_V(h)
        A_U = self.attention_U(h)
        A = self.attention_w(A_V * A_U).squeeze(-1)

        if mask is not None:
            A = A.masked_fill(mask == 0, -1e4)

        A = torch.softmax(A, dim=1)
        bag_feature = (A.unsqueeze(-1) * h).sum(dim=1)
        logits = self.classifier(bag_feature)
        return logits, A
