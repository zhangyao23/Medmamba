import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet18, resnet34
from typing import Dict, Literal, Tuple


def _convert_bn_to_gn(module, num_groups=32):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            num_ch = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(num_groups, num_ch),
                num_channels=num_ch,
                eps=child.eps,
                affine=child.affine
            )
            setattr(module, name, gn)
        else:
            _convert_bn_to_gn(child, num_groups)


class VolumetricFeatureExtractor(nn.Module):
    def __init__(
        self,
        arch: Literal['resnet18', 'resnet34'] = 'resnet18',
        spatial_dims: int = 3,
        n_input_channels: int = 1,
        pretrained: bool = False,
        frozen: bool = False
    ):
        super().__init__()
        
        self.arch = arch
        self.frozen = frozen
        self.spatial_dims = spatial_dims
        
        if arch == 'resnet18':
            self.encoder = resnet18(
                spatial_dims=spatial_dims,
                n_input_channels=n_input_channels,
                num_classes=512,
                feed_forward=False
            )
            self.output_dim = 512
        elif arch == 'resnet34':
            self.encoder = resnet34(
                spatial_dims=spatial_dims,
                n_input_channels=n_input_channels,
                num_classes=512,
                feed_forward=False
            )
            self.output_dim = 512
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Choose 'resnet18' or 'resnet34'")
        
        _convert_bn_to_gn(self.encoder, num_groups=32)
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        
        print(f"3D Feature Extractor: {arch}, spatial_dims={spatial_dims}, output_dim={self.output_dim}, norm=GroupNorm32")
    
    def forward(self, patches: torch.Tensor, mini_batch_size: int = 8) -> torch.Tensor:
        if patches.dim() == 6:
            B, N, C, D, H, W = patches.shape
        else:
            raise ValueError(f"Expected 6D tensor, got {patches.dim()}D")
        
        if self.training and self.frozen:
            self.encoder.eval()
        
        all_features = []
        encoder_device = next(self.encoder.parameters()).device
        
        for b in range(B):
            sample_patches = patches[b]
            sample_features = []
            
            for i in range(0, N, mini_batch_size):
                end_idx = min(i + mini_batch_size, N)
                batch = sample_patches[i:end_idx].to(encoder_device, non_blocking=True)
                
                with torch.set_grad_enabled(not self.frozen):
                    feat = self.encoder(batch)
                
                if feat.dim() == 5:
                    feat = torch.nn.functional.adaptive_avg_pool3d(feat, (1, 1, 1))
                    feat = feat.view(feat.size(0), -1)
                elif feat.dim() == 2:
                    pass
                else:
                    feat = feat.view(feat.size(0), -1)
                
                sample_features.append(feat)
                
                del batch, feat
            
            sample_features = torch.cat(sample_features, dim=0)
            all_features.append(sample_features)
            
            del sample_features
        
        features = torch.stack(all_features, dim=0)
        
        return features
    
    def forward_multiscale(self, patches: torch.Tensor, mini_batch_size: int = 8,
                           return_spatial: bool = False):
        if patches.dim() == 6:
            B, N, C, D, H, W = patches.shape
        else:
            raise ValueError(f"Expected 6D tensor, got {patches.dim()}D")

        if self.training and self.frozen:
            self.encoder.eval()

        enc = self.encoder
        encoder_device = next(enc.parameters()).device

        all_l2 = []
        all_l3 = []
        all_l4 = []
        all_sp_l1 = [] if return_spatial else None
        all_sp_l2 = [] if return_spatial else None
        all_sp_l3 = [] if return_spatial else None
        all_sp_l4 = [] if return_spatial else None

        for b in range(B):
            sample_patches = patches[b]
            s_l2, s_l3, s_l4 = [], [], []
            sp_l1, sp_l2, sp_l3, sp_l4 = ([], [], [], []) if return_spatial else (None, None, None, None)

            for i in range(0, N, mini_batch_size):
                end_idx = min(i + mini_batch_size, N)
                batch = sample_patches[i:end_idx].to(encoder_device, non_blocking=True)

                with torch.set_grad_enabled(not self.frozen):
                    _act = getattr(enc, 'act', None) or enc.relu
                    h = _act(enc.bn1(enc.conv1(batch)))
                    h = enc.maxpool(h)
                    h1 = enc.layer1(h)
                    h2 = enc.layer2(h1)
                    h3 = enc.layer3(h2)
                    h4 = enc.layer4(h3)

                    f2 = F.adaptive_avg_pool3d(h2, (1, 1, 1)).view(h2.size(0), -1)
                    f3 = F.adaptive_avg_pool3d(h3, (1, 1, 1)).view(h3.size(0), -1)
                    f4 = F.adaptive_avg_pool3d(h4, (1, 1, 1)).view(h4.size(0), -1)

                s_l2.append(f2)
                s_l3.append(f3)
                s_l4.append(f4)

                if return_spatial:
                    sp_l1.append(h1)
                    sp_l2.append(h2)
                    sp_l3.append(h3)
                    sp_l4.append(h4)

                del batch, h, h1, h2, h3, h4, f2, f3, f4

            all_l2.append(torch.cat(s_l2, dim=0))
            all_l3.append(torch.cat(s_l3, dim=0))
            all_l4.append(torch.cat(s_l4, dim=0))
            if return_spatial:
                all_sp_l1.append(torch.cat(sp_l1, dim=0))
                all_sp_l2.append(torch.cat(sp_l2, dim=0))
                all_sp_l3.append(torch.cat(sp_l3, dim=0))
                all_sp_l4.append(torch.cat(sp_l4, dim=0))

            del s_l2, s_l3, s_l4

        multi_features = {
            'layer2': torch.stack(all_l2, dim=0),
            'layer3': torch.stack(all_l3, dim=0),
            'layer4': torch.stack(all_l4, dim=0),
        }
        features = multi_features['layer4']

        if return_spatial:
            spatial_features = {
                'layer1': torch.stack(all_sp_l1, dim=0),
                'layer2': torch.stack(all_sp_l2, dim=0),
                'layer3': torch.stack(all_sp_l3, dim=0),
                'layer4': torch.stack(all_sp_l4, dim=0),
            }
            return multi_features, features, spatial_features

        return multi_features, features

    def get_multiscale_dims(self) -> Dict[str, int]:
        return {'layer2': 128, 'layer3': 256, 'layer4': 512}

    def get_feature_dim(self) -> int:
        return self.output_dim
