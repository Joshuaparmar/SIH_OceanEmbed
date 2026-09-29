"""
OceanEmbed CBAM U-Net.

CNN encoder-decoder (U-Net style) with CBAM attention, reconstructing
subsurface ocean temperature at D depths from C surface variables.

    Input:  [B, C, H, W]
    Output: [B, D, H, W]

C and D are read from ModelConfig — nothing about the number of
variables or depths is hardcoded. The encoder bottleneck is the
compact learned embedding referenced in the project name ("OceanEmbed").
"""

from typing import List

import torch
import torch.nn as nn

from configs.model_config import ModelConfig
from models.blocks import ConvBlock, Downsample, EncoderBlock, DecoderBlock
from models.cbam import CBAM


class CBAMUNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        channels = config.stage_channels()  # e.g. (base, base*m, base*m^2, ...)

        # 1. Input convolution block: C -> base_channels
        self.input_conv = ConvBlock(
            config.in_channels,
            channels[0],
            kernel_size=config.conv_kernel_size,
            norm=config.norm,
            activation=config.activation,
            dropout=config.dropout,
            group_norm_groups=config.group_norm_groups,
        )

        # 2 & 3. Encoder blocks + downsampling between stages.
        # channels = (ch0, ch1, ..., chN) where N = num_stages.
        # Stage i: channels[i] -> channels[i+1] (channel growth happens in the
        # conv block); the skip is taken at full resolution *after* growth,
        # then Downsample halves H, W only (channel count unchanged).
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage in range(config.num_stages):
            in_c, out_c = channels[stage], channels[stage + 1]
            self.encoder_blocks.append(
                EncoderBlock(
                    in_c, out_c,
                    kernel_size=config.conv_kernel_size,
                    norm=config.norm,
                    activation=config.activation,
                    dropout=config.dropout,
                    group_norm_groups=config.group_norm_groups,
                    use_cbam=config.use_cbam,
                    cbam_reduction_ratio=config.cbam_reduction_ratio,
                    cbam_spatial_kernel_size=config.cbam_spatial_kernel_size,
                )
            )
            self.downsamples.append(Downsample(out_c, mode=config.downsample_mode))

        # 4. Bottleneck (deepest, compact learned embedding) + CBAM.
        # Input is channels[N] (the encoder's final channel count); stays at
        # channels[N] since the encoder already reached max depth.
        bottleneck_c = channels[config.num_stages]
        self.bottleneck_conv = ConvBlock(
            bottleneck_c, bottleneck_c,
            kernel_size=config.conv_kernel_size,
            norm=config.norm,
            activation=config.activation,
            dropout=config.dropout,
            group_norm_groups=config.group_norm_groups,
        )
        self.bottleneck_cbam = (
            CBAM(bottleneck_c, config.cbam_reduction_ratio, config.cbam_spatial_kernel_size)
            if config.use_cbam else None
        )

        # 5/6. Decoder blocks with skip connections (mirrors encoder, deepest to shallowest).
        # Stage i (reversed): incoming features at channels[i+1], skip at
        # channels[i+1] (from encoder stage i), fused down to channels[i].
        self.decoder_blocks = nn.ModuleList()
        prev_c = bottleneck_c
        for stage in reversed(range(config.num_stages)):
            skip_c = channels[stage + 1]
            out_c = channels[stage]
            self.decoder_blocks.append(
                DecoderBlock(
                    prev_c, skip_c, out_c,
                    kernel_size=config.conv_kernel_size,
                    norm=config.norm,
                    activation=config.activation,
                    dropout=config.dropout,
                    group_norm_groups=config.group_norm_groups,
                    use_cbam=config.use_cbam,
                    cbam_reduction_ratio=config.cbam_reduction_ratio,
                    cbam_spatial_kernel_size=config.cbam_spatial_kernel_size,
                    upsample_mode=config.upsample_mode,
                )
            )
            prev_c = out_c

        # 7. Final 1x1 conv output head -> D depths. No activation: this is
        #    continuous regression, not classification, so we leave raw
        #    linear output (sigmoid/softmax would incorrectly bound the range).
        self.output_head = nn.Conv2d(channels[0], config.out_depths, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass. Returns only the prediction [B, D, H, W]."""
        out, _embedding = self.forward_with_embedding(x)
        return out

    def forward_with_embedding(self, x: torch.Tensor):
        """
        Same computation as forward(), but also returns the bottleneck
        embedding tensor.

        The embedding is the encoder bottleneck's output *after* its CBAM
        attention (when enabled) — the "compact learned embedding"
        referenced in the project name. It is a **spatial feature map**,
        not a single flattened vector:

            embedding shape: [B, C_bottleneck, H_bottleneck, W_bottleneck]

        where `C_bottleneck` is the deepest stage's channel count
        (`config.stage_channels()[-1]`) and `H_bottleneck`, `W_bottleneck`
        are the input's H, W each roughly halved `config.num_stages` times
        (exact values depend on rounding for non-power-of-2 input sizes).

        The encoder bottleneck acts as a compact spatial embedding that
        captures latent surface-ocean features at a lower spatial
        resolution — it is not pooled or flattened into a single vector,
        so spatial structure within the embedding is preserved.

        The embedding is not detached from the autograd graph, so
        gradients flow through it normally during training; callers that
        want a detached copy (e.g. for embedding analysis/visualization)
        should call `.detach()` themselves.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input of shape [B, C, H, W], got {tuple(x.shape)}")
        if x.size(1) != self.config.in_channels:
            raise ValueError(
                f"Expected {self.config.in_channels} input channels, got {x.size(1)}"
            )

        x = self.input_conv(x)

        skips: List[torch.Tensor] = []
        for encoder_block, downsample in zip(self.encoder_blocks, self.downsamples):
            x = encoder_block(x)
            skips.append(x)          # save pre-downsample feature map for the decoder
            x = downsample(x)

        x = self.bottleneck_conv(x)
        if self.bottleneck_cbam is not None:
            x = self.bottleneck_cbam(x)
        embedding = x  # compact spatial embedding (OceanEmbed bottleneck) — not detached

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip)

        out = self.output_head(x)  # [B, D, H, W], no final activation
        return out, embedding

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Readable architecture printout: per-module parameter counts + shapes config."""
        lines = []
        lines.append("=" * 70)
        lines.append("OceanEmbed CBAM U-Net — Architecture Summary")
        lines.append("=" * 70)
        lines.append(f"Input channels (C):     {self.config.in_channels}")
        lines.append(f"Output depths (D):      {self.config.out_depths}")
        lines.append(f"Encoder stages:         {self.config.num_stages}")
        lines.append(f"Stage channels:         {self.config.stage_channels()}")
        lines.append(f"CBAM enabled:           {self.config.use_cbam}")
        lines.append(f"Downsample / Upsample:  {self.config.downsample_mode} / {self.config.upsample_mode}")
        lines.append("-" * 70)

        def block_params(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        lines.append(f"{'Module':<28}{'Params':>15}")
        lines.append("-" * 70)
        lines.append(f"{'input_conv':<28}{block_params(self.input_conv):>15,}")
        for i, (enc, down) in enumerate(zip(self.encoder_blocks, self.downsamples)):
            lines.append(f"{'encoder_block_' + str(i):<28}{block_params(enc):>15,}")
            lines.append(f"{'downsample_' + str(i):<28}{block_params(down):>15,}")
        lines.append(f"{'bottleneck_conv':<28}{block_params(self.bottleneck_conv):>15,}")
        if self.bottleneck_cbam is not None:
            lines.append(f"{'bottleneck_cbam':<28}{block_params(self.bottleneck_cbam):>15,}")
        for i, dec in enumerate(self.decoder_blocks):
            lines.append(f"{'decoder_block_' + str(i):<28}{block_params(dec):>15,}")
        lines.append(f"{'output_head':<28}{block_params(self.output_head):>15,}")
        lines.append("-" * 70)
        lines.append(f"{'TOTAL':<28}{self.count_parameters():>15,}")
        lines.append("=" * 70)
        return "\n".join(lines)
