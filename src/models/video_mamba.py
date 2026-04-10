import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not available, using LSTM fallback")


class VideoMamba3D(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 4,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        if MAMBA_AVAILABLE:
            self.forward_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(num_layers)
            ])
            
            if bidirectional:
                self.backward_layers = nn.ModuleList([
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand
                    )
                    for _ in range(num_layers)
                ])
                self.fusion = nn.Linear(d_model * 2, d_model)
        else:
            self.forward_layers = nn.ModuleList([
                nn.LSTM(d_model, d_model, batch_first=True)
                for _ in range(num_layers)
            ])
            
            if bidirectional:
                self.backward_layers = nn.ModuleList([
                    nn.LSTM(d_model, d_model, batch_first=True)
                    for _ in range(num_layers)
                ])
                self.fusion = nn.Linear(d_model * 2, d_model)
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        
        if bidirectional:
            self.backward_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        mask_expand = mask.unsqueeze(-1) if mask is not None else None

        h_fwd = x
        for i, layer in enumerate(self.forward_layers):
            if MAMBA_AVAILABLE:
                h_fwd = layer(h_fwd) + h_fwd
            else:
                h_fwd_out, _ = layer(h_fwd)
                h_fwd = h_fwd_out + h_fwd
            h_fwd = self.layer_norms[i](h_fwd)
            h_fwd = self.dropout(h_fwd)
            if mask_expand is not None:
                h_fwd = h_fwd * mask_expand
        
        if not self.bidirectional:
            return h_fwd
        
        mask_flip = torch.flip(mask_expand, dims=[1]) if mask_expand is not None else None
        h_bwd = torch.flip(x, dims=[1])
        for i, layer in enumerate(self.backward_layers):
            if MAMBA_AVAILABLE:
                h_bwd = layer(h_bwd) + h_bwd
            else:
                h_bwd_out, _ = layer(h_bwd)
                h_bwd = h_bwd_out + h_bwd
            h_bwd = self.backward_norms[i](h_bwd)
            h_bwd = self.dropout(h_bwd)
            if mask_flip is not None:
                h_bwd = h_bwd * mask_flip
        h_bwd = torch.flip(h_bwd, dims=[1])
        
        h = self.fusion(torch.cat([h_fwd, h_bwd], dim=-1))
        
        if mask_expand is not None:
            h = h * mask_expand
        
        return h
