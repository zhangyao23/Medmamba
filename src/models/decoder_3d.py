import torch
import torch.nn as nn


class VolumetricDecoder3D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patch_size: tuple = (32, 32, 32),
        output_channels: int = 1,
        hidden_dims=None
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 1024]

        self.patch_size = tuple(patch_size)
        self.output_channels = int(output_channels)
        out_dim = self.output_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, quantized: torch.Tensor) -> torch.Tensor:
        bsz, n_patches, feat_dim = quantized.shape
        decoded = self.decoder(quantized.reshape(bsz * n_patches, feat_dim))
        decoded = decoded.view(
            bsz,
            n_patches,
            self.output_channels,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2]
        )
        return torch.sigmoid(decoded)
