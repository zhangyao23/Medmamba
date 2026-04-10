import sys
sys.path.append('/mnt/nas/share/home/liuke/prjs/uter/model_with_mamba')

import torch
from monai.networks.nets import resnet18
from src.models.vector_quantizer_3d import PartitionedVectorQuantizer
from src.models.video_mamba import VideoMamba3D
from src.models.mil_head import AttentionMILHead

print("=" * 50)
print("3D MIL Model Parameter Count")
print("=" * 50)

print("\n1. Feature Extractor (ResNet-18 3D):")
feature_extractor = resnet18(spatial_dims=3, n_input_channels=1, num_classes=512)
fe_params = sum(p.numel() for p in feature_extractor.parameters())
print(f"   Parameters: {fe_params:,} ({fe_params/1e6:.2f}M)")

print("\n2. Vector Quantizer (Codebook):")
vq = PartitionedVectorQuantizer(
    num_embeddings=100,
    embedding_dim=512,
    healthy_ratio=0.8,
    commitment_cost=0.25
)
vq_params = sum(p.numel() for p in vq.parameters())
print(f"   Parameters: {vq_params:,} ({vq_params/1e6:.2f}M)")

print("\n3. VideoMamba3D:")
mamba = VideoMamba3D(
    d_model=512,
    d_state=16,
    d_conv=4,
    expand=2,
    num_layers=4,
    bidirectional=True
)
mamba_params = sum(p.numel() for p in mamba.parameters())
print(f"   Parameters: {mamba_params:,} ({mamba_params/1e6:.2f}M)")

print("\n4. MIL Head (Attention):")
mil_head = AttentionMILHead(input_dim=512, hidden_dim=256, num_classes=2)
mil_params = sum(p.numel() for p in mil_head.parameters())
print(f"   Parameters: {mil_params:,} ({mil_params/1e6:.2f}M)")

total = fe_params + vq_params + mamba_params + mil_params
print("\n" + "=" * 50)
print(f"TOTAL: {total:,} ({total/1e6:.2f}M)")
print("=" * 50)

print(f"\nPercentage breakdown:")
print(f"  ResNet-18:    {fe_params/total*100:.1f}%")
print(f"  Codebook:     {vq_params/total*100:.1f}%")
print(f"  Mamba:        {mamba_params/total*100:.1f}%")
print(f"  MIL Head:     {mil_params/total*100:.1f}%")
