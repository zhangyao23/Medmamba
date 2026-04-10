import sys
sys.path.append('/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba')

import torch
from src.models.video_mamba import VideoMamba3D

mamba = VideoMamba3D(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2,
    num_layers=4,
    bidirectional=True
)

total_params = sum(p.numel() for p in mamba.parameters())
trainable_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Size: {total_params / 1e6:.2f}M")

print("\nBreakdown:")
print(f"Forward layers: {sum(p.numel() for layer in mamba.forward_layers for p in layer.parameters()):,}")
if mamba.bidirectional:
    print(f"Backward layers: {sum(p.numel() for layer in mamba.backward_layers for p in layer.parameters()):,}")
    print(f"Fusion layer: {sum(p.numel() for p in mamba.fusion.parameters()):,}")
print(f"Layer norms: {sum(p.numel() for layer in mamba.layer_norms for p in layer.parameters()):,}")
