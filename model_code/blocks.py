"""
Reusable building blocks for the CBAM U-Net:
  - ConvBlock:     double conv (+norm +activation), the core feature extractor
  - Downsample:    spatial halving (maxpool or strided conv)
  - EncoderBlock:  ConvBlock (+ optional CBAM) producing a skip feature map
  - Upsample:      spatial doubling (bilinear/nearest interpolation or transposed conv)
  - DecoderBlock:  upsample + size-aligned skip concat + ConvBlock (+ optional CBAM)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cbam import CBAM


def _activation_layer(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _safe_group_count(num_channels: int, requested_groups: int) -> int:
    """
    Largest divisor of `num_channels` that is <= requested_groups (and >= 1).

    GroupNorm requires num_groups to evenly divide num_channels, and
    channel counts vary by stage in this model (base_channels *
    channel_multiplier ** stage), so a single fixed group count isn't
    guaranteed to divide every stage. Rather than erroring out on every
    awkward (channels, requested_groups) combination, each norm layer
    silently falls back to the nearest valid group count <= requested.
    """
    groups = min(requested_groups, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


def _norm_layer(norm: str, num_channels: int, group_norm_groups: int = 8) -> Optional[nn.Module]:
    """Build the requested normalization layer, or None for norm='none'."""
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "group":
        groups = _safe_group_count(num_channels, group_norm_groups)
        return nn.GroupNorm(num_groups=groups, num_channels=num_channels)
    if norm == "none":
        return None
    raise ValueError(f"Unsupported norm: {norm!r}. Expected 'batch', 'group', or 'none'.")


class ConvBlock(nn.Module):
    """
    Two (Conv2d -> [Norm] -> Activation) layers.

    This is the workhorse feature extractor used at every stage of the
    encoder and decoder. Padding is set to preserve H, W so that spatial
    changes only ever happen in explicit Downsample/Upsample steps —
    this keeps skip-connection alignment predictable.

    `norm` supports "batch" (BatchNorm2d), "group" (GroupNorm — useful
    when batch size is small, since it normalizes within each sample
    rather than across the batch), or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
        group_norm_groups: int = 8,
    ):
        super().__init__()
        padding = kernel_size // 2
        use_bias = norm == "none"  # BN/GN make conv bias redundant

        layers = []
        for i, (c_in, c_out) in enumerate([(in_channels, out_channels), (out_channels, out_channels)]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=use_bias))
            norm_layer = _norm_layer(norm, c_out, group_norm_groups)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(_activation_layer(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample(nn.Module):
    """Halves H and W (rounding down), via maxpool or a stride-2 conv."""

    def __init__(self, channels: int, mode: str = "maxpool"):
        super().__init__()
        self.mode = mode
        if mode == "maxpool":
            self.op = nn.MaxPool2d(kernel_size=2, stride=2)
        elif mode == "stride":
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported downsample mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class EncoderBlock(nn.Module):
    """
    ConvBlock (+ optional CBAM) at the current resolution. Returns the
    pre-downsample feature map (used as a skip connection) — downsampling
    itself is applied by the caller between stages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
        group_norm_groups: int = 8,
        use_cbam: bool = True,
        cbam_reduction_ratio: int = 8,
        cbam_spatial_kernel_size: int = 7,
    ):
        super().__init__()
        self.conv_block = ConvBlock(
            in_channels, out_channels, kernel_size, norm, activation, dropout, group_norm_groups
        )
        self.cbam: Optional[CBAM] = (
            CBAM(out_channels, cbam_reduction_ratio, cbam_spatial_kernel_size) if use_cbam else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


def _align_and_concat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """
    Robustly align `upsampled` to `skip`'s spatial size before concatenation.

    Odd input sizes (e.g. 301x481) mean the encoder's downsample/upsample
    round-trip won't always land back on the exact original resolution.
    Rather than assuming exact divisibility, we center-crop or pad the
    upsampled tensor to match the skip tensor's H, W exactly.
    """
    diff_h = skip.size(2) - upsampled.size(2)
    diff_w = skip.size(3) - upsampled.size(3)

    if diff_h != 0 or diff_w != 0:
        # Pad if smaller, crop if larger (pad handles negative via crop first).
        pad_top = max(diff_h, 0) // 2
        pad_bottom = max(diff_h, 0) - pad_top
        pad_left = max(diff_w, 0) // 2
        pad_right = max(diff_w, 0) - pad_left
        if pad_top or pad_bottom or pad_left or pad_right:
            upsampled = F.pad(upsampled, [pad_left, pad_right, pad_top, pad_bottom])

        crop_h = max(-diff_h, 0)
        crop_w = max(-diff_w, 0)
        if crop_h or crop_w:
            top = crop_h // 2
            left = crop_w // 2
            upsampled = upsampled[:, :, top:top + skip.size(2), left:left + skip.size(3)]

    return torch.cat([upsampled, skip], dim=1)


class Upsample(nn.Module):
    """Doubles H and W, via interpolation or a transposed conv."""

    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        if mode == "transpose":
            self.op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            # interpolation keeps channel count; a 1x1 conv projects to out_channels
            align_corners = False if mode == "bilinear" else None
            self.interp_mode = mode
            self.align_corners = align_corners
            self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            self.op = None

    def forward(self, x: torch.Tensor, target_size: Optional[torch.Size] = None) -> torch.Tensor:
        if self.op is not None:
            return self.op(x)
        scale_factor = 2 if target_size is None else None
        size = target_size if target_size is not None else None
        kwargs = {"align_corners": self.align_corners} if self.interp_mode == "bilinear" else {}
        x = F.interpolate(
            x, size=size, scale_factor=scale_factor, mode=self.interp_mode, **kwargs
        )
        return self.project(x)


class DecoderBlock(nn.Module):
    """
    Upsamples the incoming feature map, concatenates it with the matching
    encoder skip connection (with robust size alignment), then applies a
    ConvBlock (+ optional CBAM) to fuse the combined features.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
        group_norm_groups: int = 8,
        use_cbam: bool = True,
        cbam_reduction_ratio: int = 8,
        cbam_spatial_kernel_size: int = 7,
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.upsample = Upsample(in_channels, in_channels, mode=upsample_mode)
        self.conv_block = ConvBlock(
            in_channels + skip_channels, out_channels, kernel_size, norm, activation, dropout, group_norm_groups
        )
        self.cbam: Optional[CBAM] = (
            CBAM(out_channels, cbam_reduction_ratio, cbam_spatial_kernel_size) if use_cbam else None
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x, target_size=skip.shape[2:])
        x = _align_and_concat(x, skip)
        x = self.conv_block(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x
