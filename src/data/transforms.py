from typing import Iterable, List, Sequence

import torch


class Compose:
    def __init__(self, transforms: Iterable):
        self.transforms = list(transforms)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            tensor = transform(tensor)
        return tensor


class ClampIntensity:
    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clamp(self.min_value, self.max_value)


class RandomIntensityScaleShift:
    def __init__(
        self,
        scale_range: Sequence[float] = (0.9, 1.1),
        shift_range: Sequence[float] = (-0.05, 0.05),
        p: float = 0.3,
    ):
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor

        scale = torch.empty(
            1, device=tensor.device, dtype=tensor.dtype
        ).uniform_(self.scale_range[0], self.scale_range[1])
        shift = torch.empty(
            1, device=tensor.device, dtype=tensor.dtype
        ).uniform_(self.shift_range[0], self.shift_range[1])
        return tensor * scale + shift


class RandomGaussianNoise:
    def __init__(self, std: float = 0.02, p: float = 0.2):
        self.std = std
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() >= self.p:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std


class PerSampleStandardize:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() < 3:
            return tensor

        if tensor.dim() in (3, 4):
            reduce_dims = tuple(range(1, tensor.dim()))
        else:
            reduce_dims = tuple(range(2, tensor.dim()))

        mean = tensor.mean(dim=reduce_dims, keepdim=True)
        std = tensor.std(dim=reduce_dims, keepdim=True, unbiased=False)
        return (tensor - mean) / std.clamp(min=self.eps)


def get_train_transforms(
    standardize: bool = False,
    intensity_jitter: bool = False,
    gaussian_noise: bool = False,
    clamp_range: Sequence[float] = (0.0, 1.0),
) -> Compose:
    transforms: List = []
    if intensity_jitter:
        transforms.append(RandomIntensityScaleShift())
    if gaussian_noise:
        transforms.append(RandomGaussianNoise())
    transforms.append(ClampIntensity(*clamp_range))
    if standardize:
        transforms.append(PerSampleStandardize())
    return Compose(transforms)


def get_val_transforms(
    standardize: bool = False,
    clamp_range: Sequence[float] = (0.0, 1.0),
) -> Compose:
    transforms: List = [ClampIntensity(*clamp_range)]
    if standardize:
        transforms.append(PerSampleStandardize())
    return Compose(transforms)
