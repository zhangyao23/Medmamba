import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VectorQuantizer3D(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 2048,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        self.register_buffer(
            'normal_prototype_mask',
            torch.zeros(num_embeddings, dtype=torch.bool)
        )
        self.register_buffer(
            'code_usage_count',
            torch.zeros(num_embeddings)
        )
        
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
            self.register_buffer('ema_w', self.embedding.weight.data.clone())
            self.ema_decay = ema_decay
            self.epsilon = epsilon
    
    def forward(
        self,
        z: torch.Tensor,
        is_negative_bag: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, D = z.shape
        z_flat = z.reshape(-1, D)
        
        distances = torch.cdist(z_flat, self.embedding.weight)
        encoding_indices = distances.argmin(dim=1)
        
        quantized = self.embedding(encoding_indices)
        
        if self.training:
            if self.use_ema:
                self._ema_update(z_flat, encoding_indices)
                vq_loss = self.commitment_cost * F.mse_loss(quantized.detach(), z_flat)
            else:
                e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
                q_latent_loss = F.mse_loss(quantized, z_flat.detach())
                vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
            
            if is_negative_bag is not None:
                self._update_normal_prototypes(encoding_indices, is_negative_bag, B, N)
        else:
            vq_loss = torch.tensor(0.0, device=z.device)
        
        quantized = z_flat + (quantized - z_flat).detach()
        
        quantized = quantized.view(B, N, D)
        encoding_indices = encoding_indices.view(B, N)
        
        return quantized, encoding_indices, vq_loss
    
    def _ema_update(self, z_flat: torch.Tensor, encoding_indices: torch.Tensor):
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                (1 - self.ema_decay) * encodings.sum(0)
        
        n = self.ema_cluster_size.sum()
        self.ema_cluster_size = (
            (self.ema_cluster_size + self.epsilon) /
            (n + self.num_embeddings * self.epsilon) * n
        )
        
        dw = encodings.t() @ z_flat
        self.ema_w = self.ema_w * self.ema_decay + (1 - self.ema_decay) * dw
        
        self.embedding.weight.data = self.ema_w / self.ema_cluster_size.unsqueeze(1)
    
    def _update_normal_prototypes(
        self,
        encoding_indices: torch.Tensor,
        is_negative_bag: torch.Tensor,
        B: int,
        N: int
    ):
        indices_reshaped = encoding_indices.view(B, N)
        neg_indices = indices_reshaped[is_negative_bag].flatten()
        
        if len(neg_indices) > 0:
            self.normal_prototype_mask[neg_indices] = True
            self.code_usage_count[encoding_indices] += 1
    
    def get_normal_prototype_penalty(
        self,
        encoding_indices: torch.Tensor,
        bag_labels: torch.Tensor
    ) -> torch.Tensor:
        pos_mask = (bag_labels == 1)
        if not pos_mask.any():
            return torch.tensor(0.0, device=encoding_indices.device)
        
        pos_indices = encoding_indices[pos_mask]
        is_normal_code = self.normal_prototype_mask[pos_indices]
        
        normal_ratio = is_normal_code.float().mean()
        return normal_ratio
