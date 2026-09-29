"""
Configuration for the OceanEmbed CBAM U-Net model.

Nothing here is hardcoded to a specific dataset. `in_channels`,
`out_depths`, and the spatial size are all left as free parameters —
the preprocessing pipeline (developed separately) determines the real
values at data-loading time. This file only defines *architecture*
knobs, with safe defaults for smoke-testing.

`in_channels=5` / `out_depths=13` in the defaults below are only
placeholders left over from an earlier working spec. The project's
current intended input set (SST, SSS, SSH/SLA, current-u/v, wind-u/v,
land mask) would put `in_channels` around 8, but the preprocessing
pipeline has not finalized the channel arrangement — this dataclass
must be constructed with the real `in_channels` once that is known.
Nothing in the model architecture assumes any specific value.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class ModelConfig:
    # --- Data-dependent shape parameters (set these from the actual
    #     preprocessing pipeline output — do NOT assume fixed values) ---
    in_channels: int = 5          # C: number of surface input variables
    out_depths: int = 13          # D: number of subsurface target depths

    # Optional metadata only — purely descriptive, never consumed by the
    # architecture. Lets the config carry the physical depth (in meters)
    # that each output channel corresponds to, so downstream training/
    # evaluation code has one place to read it from instead of guessing.
    # If provided, its length must equal out_depths. Leave as None until
    # the preprocessing pipeline confirms the final depth list.
    target_depths: Optional[Sequence[float]] = None

    # --- Architecture knobs ---
    base_channels: int = 32       # channels after the input conv block
    num_stages: int = 3           # number of encoder downsampling stages
    channel_multiplier: int = 2   # channel growth factor per stage
    max_channels: int = 256       # cap on channel growth for deep configs

    # CBAM
    use_cbam: bool = True
    cbam_reduction_ratio: int = 8
    cbam_spatial_kernel_size: int = 7

    # Conv block internals
    conv_kernel_size: int = 3
    norm: str = "batch"           # "batch" | "group" | "none"
    group_norm_groups: int = 8    # only used when norm == "group"
    activation: str = "relu"      # "relu" | "leaky_relu"
    dropout: float = 0.0          # optional dropout inside conv blocks

    # Down/up-sampling
    downsample_mode: str = "maxpool"   # "maxpool" | "stride"
    upsample_mode: str = "bilinear"    # "bilinear" | "nearest" | "transpose"

    def stage_channels(self) -> Tuple[int, ...]:
        """Channel count at the input conv and each encoder stage."""
        channels = [self.base_channels]
        for _ in range(self.num_stages):
            next_c = min(channels[-1] * self.channel_multiplier, self.max_channels)
            channels.append(next_c)
        return tuple(channels)  # length = num_stages + 1 (includes bottleneck)

    def safe_group_count(self, num_channels: int) -> int:
        """
        Largest divisor of `num_channels` that is <= group_norm_groups
        (and >= 1). GroupNorm requires num_groups to evenly divide the
        channel count, and channel counts here vary by stage (base_channels
        * channel_multiplier ** stage), so a single fixed group count is
        not guaranteed to divide every stage's channel count. Rather than
        raising at model-construction time for every awkward combination,
        each norm layer resolves its own safe group count from this helper.
        """
        groups = min(self.group_norm_groups, num_channels)
        while groups > 1 and num_channels % groups != 0:
            groups -= 1
        return max(groups, 1)

    def __post_init__(self):
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.out_depths < 1:
            raise ValueError("out_depths must be >= 1")
        if self.target_depths is not None and len(self.target_depths) != self.out_depths:
            raise ValueError(
                f"target_depths has {len(self.target_depths)} entries but out_depths={self.out_depths}; "
                "they must match 1:1 (one physical depth per output channel)."
            )
        if self.base_channels < 1:
            raise ValueError("base_channels must be >= 1")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1 for a U-Net-style model")
        if self.channel_multiplier < 1:
            raise ValueError("channel_multiplier must be >= 1")
        if self.max_channels < self.base_channels:
            raise ValueError("max_channels must be >= base_channels")

        if self.cbam_reduction_ratio < 1:
            raise ValueError("cbam_reduction_ratio must be >= 1")
        if self.cbam_spatial_kernel_size < 3 or self.cbam_spatial_kernel_size % 2 == 0:
            raise ValueError(
                "cbam_spatial_kernel_size must be odd and >= 3 "
                f"(got {self.cbam_spatial_kernel_size}); an odd kernel keeps the "
                "spatial-attention conv's output size aligned with its input."
            )

        if self.conv_kernel_size < 1 or self.conv_kernel_size % 2 == 0:
            raise ValueError(f"conv_kernel_size must be odd and >= 1 (got {self.conv_kernel_size})")

        if self.norm not in ("batch", "group", "none"):
            raise ValueError(f"Unsupported norm: {self.norm!r}. Expected 'batch', 'group', or 'none'.")
        if self.norm == "group" and self.group_norm_groups < 1:
            raise ValueError("group_norm_groups must be >= 1 when norm='group'")

        if self.activation not in ("relu", "leaky_relu"):
            raise ValueError(f"Unsupported activation: {self.activation!r}. Expected 'relu' or 'leaky_relu'.")

        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be within [0, 1] (got {self.dropout})")

        if self.downsample_mode not in ("maxpool", "stride"):
            raise ValueError(f"Unsupported downsample_mode: {self.downsample_mode!r}. Expected 'maxpool' or 'stride'.")
        if self.upsample_mode not in ("bilinear", "nearest", "transpose"):
            raise ValueError(
                f"Unsupported upsample_mode: {self.upsample_mode!r}. Expected 'bilinear', 'nearest', or 'transpose'."
            )
