"""
CBAM: Convolutional Block Attention Module.

Implements the two sequential sub-modules from Woo et al. (2018):
  1. Channel Attention  — "what" is meaningful, via shared MLP over
     avg- and max-pooled channel descriptors.
  2. Spatial Attention   — "where" is meaningful, via a conv over
     channel-pooled (avg + max) spatial maps.

CBAM(x) = SpatialAttention(ChannelAttention(x) * x) * (ChannelAttention(x) * x)

This module is attached inside encoder/decoder/bottleneck blocks
(see blocks.py) rather than used as a standalone model.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeezes spatial dims via global avg-pool and max-pool, passes both
    descriptors through a shared 2-layer MLP (implemented as 1x1 convs),
    sums the results, and applies a sigmoid gate.
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 8):
        super().__init__()
        hidden = max(num_channels // reduction_ratio, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(num_channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_channels, kernel_size=1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        gate = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return x * gate


class SpatialAttention(nn.Module):
    """
    Pools across the channel dimension (avg + max), concatenates into a
    2-channel map, and applies a single conv + sigmoid to produce a
    per-pixel spatial gate.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"SpatialAttention kernel_size must be odd and at least 3 (got {kernel_size}). "
                "An odd kernel with symmetric padding keeps the attention map's spatial size "
                "identical to its input."
            )
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        gate = self.sigmoid(self.conv(pooled))          # [B, 1, H, W]
        return x * gate


class CBAM(nn.Module):
    """Sequential channel-then-spatial attention, applied in-place on a feature map."""

    def __init__(self, num_channels: int, reduction_ratio: int = 8, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
