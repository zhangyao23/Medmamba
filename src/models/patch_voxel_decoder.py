import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ResBlock3d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels), channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SkipFusion(nn.Module):
    def __init__(self, decoder_ch: int, skip_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(decoder_ch + skip_ch, decoder_ch, kernel_size=1),
            nn.GroupNorm(min(32, decoder_ch), decoder_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_feat, skip_feat):
        if skip_feat.shape[2:] != decoder_feat.shape[2:]:
            skip_feat = F.interpolate(
                skip_feat, size=decoder_feat.shape[2:],
                mode='trilinear', align_corners=False)
        return self.conv(torch.cat([decoder_feat, skip_feat], dim=1))


class UNetUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            ResBlock3d(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SpatialUNetDecoder(nn.Module):
    def __init__(self, patch_size: Tuple[int, int, int] = (32, 32, 32),
                 context_dim: int = 0):
        super().__init__()
        self.patch_size = patch_size
        self.context_dim = context_dim

        if context_dim > 0:
            self.film_scale = nn.Linear(context_dim, 512)
            self.film_bias = nn.Linear(context_dim, 512)
            nn.init.zeros_(self.film_scale.weight)
            nn.init.zeros_(self.film_scale.bias)
            nn.init.zeros_(self.film_bias.weight)
            nn.init.zeros_(self.film_bias.bias)

        self.up4 = UNetUpBlock(512, 256, 256)
        self.up3 = UNetUpBlock(256, 128, 128)
        self.up2 = UNetUpBlock(128, 64, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, h1, h2, h3, h4, context=None):
        if self.context_dim > 0 and context is not None:
            scale = self.film_scale(context).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            bias = self.film_bias(context).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            h4 = h4 * (1.0 + scale) + bias
        x = self.up4(h4, h3)
        x = self.up3(x, h2)
        x = self.up2(x, h1)
        x = self.final(x)
        pd, ph, pw = self.patch_size
        return x[:, 0, :pd, :ph, :pw]


class PatchVoxelDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        hidden_dims: List[int] = [256, 128, 64],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        pd, ph, pw = patch_size

        n_upsample = len(hidden_dims)
        init_d = pd // (2 ** n_upsample)
        init_h = ph // (2 ** n_upsample)
        init_w = pw // (2 ** n_upsample)
        assert init_d >= 1 and init_h >= 1 and init_w >= 1, (
            f"patch_size {patch_size} too small for {n_upsample} upsample layers"
        )

        self.init_spatial = (init_d, init_h, init_w)
        first_ch = hidden_dims[0]
        self.fc = nn.Linear(input_dim, first_ch * init_d * init_h * init_w)
        self.fc_norm = nn.GroupNorm(min(32, first_ch), first_ch)
        self.fc_relu = nn.ReLU(inplace=True)

        self.up_blocks = nn.ModuleList()
        in_ch = first_ch
        for out_ch in hidden_dims[1:]:
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.ReLU(inplace=True),
                ResBlock3d(out_ch),
            ))
            in_ch = out_ch

        self.final = nn.Sequential(
            nn.ConvTranspose3d(in_ch, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, patch_features: torch.Tensor) -> torch.Tensor:
        B, N, D = patch_features.shape
        x = self.fc(patch_features)
        ch = x.shape[-1] // (self.init_spatial[0] * self.init_spatial[1] * self.init_spatial[2])
        x = x.view(B * N, ch, *self.init_spatial)
        x = self.fc_relu(self.fc_norm(x))

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.final(x)
        pd, ph, pw = self.patch_size
        x = x[:, 0, :pd, :ph, :pw]
        x = x.view(B, N, pd, ph, pw)
        return x
