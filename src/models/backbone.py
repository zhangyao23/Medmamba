import torch
import torch.nn as nn
import timm


class PerSliceBackbone(nn.Module):
    def __init__(
        self, 
        arch: str = 'resnet50',
        pretrained: bool = True,
        frozen: bool = True
    ):
        super().__init__()
        
        self.frozen = frozen
        
        self.encoder = timm.create_model(
            arch,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.encoder(dummy_input)
            if dummy_output.dim() == 4:
                self.output_dim = dummy_output.shape[1]
                self.use_pooling = True
            else:
                self.output_dim = dummy_output.shape[1]
                self.use_pooling = False
        
        print(f"Backbone {arch}: output_dim={self.output_dim}, use_pooling={self.use_pooling}")
    
    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = slices.shape
        
        flat = slices.view(B * N, C, H, W)
        
        if self.training and self.frozen:
            self.encoder.eval()
        
        with torch.set_grad_enabled(not self.frozen):
            features = self.encoder(flat)
        
        if self.use_pooling and features.dim() == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        features = features.view(B, N, -1)
        
        return features

    def forward_feature_map(self, slices: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = slices.shape
        
        flat = slices.view(B * N, C, H, W)
        
        if self.training and self.frozen:
            self.encoder.eval()
        
        with torch.set_grad_enabled(not self.frozen):
            features = self.encoder(flat)
        
        if features.dim() == 2:
            features = features.view(B, N, features.size(1), 1, 1)
        else:
            features = features.view(B, N, features.size(1), features.size(2), features.size(3))
        
        return features
