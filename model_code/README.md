# OceanEmbed — CBAM U-Net Model Implementation

Satellite Embedding-Based Deep Learning Framework for Reconstruction of
Subsurface Ocean Temperature from Surface Ocean Observations (SIH 2026).

**Scope of this stage:** this delivers only the model architecture and
software-level validation (forward pass, backward pass, optimizer step,
embedding extraction). No training on real data, no dataset loading, and
no evaluation metrics are implemented — per the project brief, those
depend on the preprocessing pipeline being developed separately. No
scientific claims are made anywhere in this repo; every test tensor is
`torch.randn` noise, used purely to check shapes, gradient flow, and
numerical stability. **This is software validation, not scientific
validation** — the model has not been trained or evaluated on real ocean
data.

## 1. Architecture

CNN encoder-decoder / U-Net style network with CBAM (Convolutional Block
Attention Module) attention integrated into every encoder, bottleneck,
and decoder stage. Unchanged from the initial implementation in this
revision — only configuration, validation, normalization options, and
test coverage were extended.

```
Input [B, C, H, W]
    │
    ▼
Input Conv Block (C -> base_channels)
    │
    ├──────────────────────────────────────────────┐  skip 0
    ▼                                               │
Encoder Block 0 (+CBAM) ── Downsample (÷2)          │
    │                                               │
    ├──────────────────────────────────────────────┼──┐ skip 1
    ▼                                               │  │
Encoder Block 1 (+CBAM) ── Downsample (÷2)          │  │
    │                                               │  │
    ├──────────────────────────────────────────────┼──┼──┐ skip 2
    ▼                                               │  │  │
Encoder Block 2 (+CBAM) ── Downsample (÷2)          │  │  │
    │                                               │  │  │
    ▼                                               │  │  │
Bottleneck Conv (+CBAM)  ◄── compact spatial embedding  │  │
    │                                               │  │  │
    ▼                                               │  │  │
Decoder Block 0 (upsample + concat skip 2 + conv +CBAM) ◄┘  │
    ▼                                               │
Decoder Block 1 (upsample + concat skip 1 + conv +CBAM) ◄┘
    ▼
Decoder Block 2 (upsample + concat skip 0 + conv +CBAM)
    ▼
1x1 Conv output head (-> D channels, no activation)
    │
    ▼
Output [B, D, H, W]
```

- **Input conv block**: maps the `C` surface variables to `base_channels`.
- **Encoder blocks**: each is a double-conv block (Conv→Norm→Activation
  ×2) with CBAM attention, followed by a separate `Downsample` (max-pool
  or stride-2 conv) that halves H, W. Channel count grows by
  `channel_multiplier` at each stage (capped at `max_channels`).
- **Bottleneck**: the deepest, lowest-resolution feature map — the
  compact learned embedding referenced in the project name. A CBAM
  module is applied here too. See "Spatial embedding" below.
- **Decoder blocks**: each upsamples (bilinear/nearest interpolation or
  transposed conv), concatenates the corresponding encoder skip
  connection (with automatic size alignment), then applies a conv block
  + CBAM to fuse the combined features.
- **Output head**: a single 1×1 convolution to `D` depth channels, with
  **no final activation** (sigmoid/softmax would be wrong here — this is
  continuous temperature regression, not classification). Confirmed by
  code review this revision: no sigmoid/softmax/clamping exists anywhere
  on the output path.

### CBAM (Convolutional Block Attention Module)

Implemented in `models/cbam.py` with two sequential sub-modules, applied
inside each encoder/bottleneck/decoder block (not as a separate,
disconnected model):

1. **Channel attention** — global average- and max-pooling per channel,
   passed through a shared 2-layer MLP (implemented as 1×1 convs), summed,
   and passed through a sigmoid to produce a per-channel gate.
2. **Spatial attention** — average- and max-pooling *across* channels,
   concatenated into a 2-channel map, passed through a single conv +
   sigmoid to produce a per-pixel gate. The kernel size must be odd and
   `>= 3`; this revision replaced the previous Python `assert` with an
   explicit `ValueError` that names the constraint.

`CBAM(x)`: channel attention is applied first, then spatial attention is
applied to the channel-refined result — matching the original CBAM paper
(Woo et al., 2018).

### Robust skip-connection alignment

`H` and `W` are not guaranteed to be divisible by `2^num_stages` (e.g.
the brief mentions 301×481 as a real but non-power-of-2 example), so the
decoder does not assume the upsampled tensor exactly matches its skip
connection's spatial size. Each decoder block upsamples toward the
skip's target size, then center-pads or center-crops as needed before
concatenating along the channel dimension
(`models/blocks.py::_align_and_concat`). Re-verified this revision with
both the 301×481 stress test and an additional odd/asymmetric case
(67×53 with 4 stages).

### Spatial embedding (OceanEmbed bottleneck)

`CBAMUNet.forward_with_embedding(x)` returns `(prediction, embedding)`.
The embedding is the bottleneck's output *after* its CBAM attention
(when enabled):

```
embedding shape: [B, C_bottleneck, H_bottleneck, W_bottleneck]
```

The encoder bottleneck acts as a compact spatial embedding that captures
latent surface-ocean features at a lower spatial resolution. It is
**not** pooled or flattened into a single vector — spatial structure
within the embedding is preserved, which is why it is described as a
spatial feature map rather than an embedding vector. `forward()` calls
`forward_with_embedding()` internally and simply discards the embedding,
so the two paths always stay in sync. The embedding is never detached
from the autograd graph — gradients flow through it normally during
training; callers who want a detached copy for offline analysis should
call `.detach()` themselves.

## 2. Input / Output Tensor Format

| | Shape | Notes |
|---|---|---|
| Input | `[B, C, H, W]` | `C` = number of surface variables (configurable) |
| Output | `[B, D, H, W]` | `D` = number of subsurface depths (configurable) |

`C`, `D`, `H`, and `W` are **never hardcoded** anywhere in the model
code — re-checked this revision against the full file set (see
"Hidden-assumptions review" below). `C` and `D` come from `ModelConfig`;
`H` and `W` are inferred at runtime from the input tensor and preserved
end-to-end.

### Example input variables

The project's currently intended surface input set:

- SST — Sea Surface Temperature
- SSS — Sea Surface Salinity
- SSH/SLA — Sea Surface Height or Sea Level Anomaly (which one depends on
  the finalized dataset — these are **not** treated as interchangeable
  by this model; whichever variable the preprocessing pipeline produces
  is simply one of the `C` input channels, and its physical meaning must
  be tracked in the preprocessing/documentation, not in this model)
- Current eastward component (current-u)
- Current northward component (current-v)
- Wind eastward component (wind-u)
- Wind northward component (wind-v)
- Land-sea mask

This set would currently suggest `[B, 8, H, W]`, but **the exact number
and order of channels must match whatever the preprocessing pipeline
finalizes** — this model places no constraint on which variables occupy
which channel index, only on the total count `C`. `in_channels=5` /
`out_depths=13` remain the defaults in `ModelConfig` only because they
match an earlier working test spec; construct `ModelConfig(in_channels=8, ...)`
(or whatever `C` turns out to be) once the pipeline is finalized.

## 3. Configurable Parameters

All defined in `configs/model_config.py` (`ModelConfig` dataclass), with
validation in `__post_init__` (see "Configuration validation" below):

| Parameter | Meaning |
|---|---|
| `in_channels` (C) | number of surface input variables |
| `out_depths` (D) | number of subsurface target depths |
| `target_depths` | optional metadata: physical depth (m) per output channel — see below |
| `base_channels` | channel count after the input conv |
| `num_stages` | number of encoder downsampling stages |
| `channel_multiplier` | channel growth factor per stage |
| `max_channels` | cap on channel growth |
| `use_cbam` | toggle CBAM attention on/off |
| `cbam_reduction_ratio` | channel-attention MLP bottleneck ratio |
| `cbam_spatial_kernel_size` | spatial-attention conv kernel size (odd, >= 3) |
| `conv_kernel_size` | kernel size used in all conv blocks (odd, >= 1) |
| `norm` | `"batch"`, `"group"`, or `"none"` |
| `group_norm_groups` | requested GroupNorm group count (only used when `norm="group"`) |
| `activation` | `"relu"` or `"leaky_relu"` |
| `dropout` | dropout probability inside conv blocks, in `[0, 1]` |
| `downsample_mode` | `"maxpool"` or `"stride"` |
| `upsample_mode` | `"bilinear"`, `"nearest"`, or `"transpose"` |

None of these have dataset-specific defaults baked into the architecture
logic itself — `in_channels=5` / `out_depths=13` are only placeholders,
trivially overridden by constructing a different `ModelConfig`.

### Configuration validation

`ModelConfig.__post_init__` now raises explicit, descriptive
`ValueError`s (not bare `assert`s) for invalid settings, including:
`in_channels`/`out_depths`/`base_channels` `>= 1`; `num_stages`/
`channel_multiplier` `>= 1`; `max_channels >= base_channels`;
`cbam_reduction_ratio >= 1`; `cbam_spatial_kernel_size` odd and `>= 3`;
`conv_kernel_size` odd; `dropout` within `[0, 1]`; `norm`,
`downsample_mode`, `upsample_mode` restricted to their supported string
values; `group_norm_groups >= 1` when `norm="group"`; and, if
`target_depths` is provided, its length must equal `out_depths`.

### Normalization options, including GroupNorm

`ConvBlock` (`models/blocks.py`) now supports three normalization modes:

- `norm="batch"` (default, unchanged) — `nn.BatchNorm2d`. Normalizes
  across the batch dimension; statistics are more stable with larger
  batch sizes but can become noisy/unstable with very small batches.
- `norm="group"` (**new**) — `nn.GroupNorm`. Normalizes within each
  sample (across channel groups), independent of batch size — useful
  because the real training dataset may be small and batch size may be
  limited, where BatchNorm's running statistics tend to be less
  reliable. This is presented as **an option to evaluate**, not a
  claim that it is automatically superior to BatchNorm — the right
  choice depends on the eventual batch size and dataset characteristics.
- `norm="none"` (unchanged) — no normalization layer.

**GroupNorm channel-compatibility handling:** GroupNorm requires
`num_groups` to evenly divide the layer's channel count. Because this
model's channel count grows per stage (`base_channels * channel_multiplier ** stage`,
capped at `max_channels`), a single fixed `group_norm_groups` is not
guaranteed to divide every stage's channel count. Rather than raising a
configuration error for every such mismatch, each `GroupNorm` layer
independently resolves a **safe group count** via
`ModelConfig.safe_group_count(num_channels)` / the internal
`_safe_group_count` helper in `blocks.py`: the largest divisor of that
layer's channel count that is `<= group_norm_groups`. This keeps
GroupNorm usable across arbitrary `base_channels`/`channel_multiplier`
combinations without the caller having to hand-tune the group count per
stage. Existing BatchNorm and no-normalization modes are unchanged and
were re-tested (Case 1 and Case 6 in the test suite) to confirm nothing
broke.

## 4. Target-depth channel mapping

`out_depths` (`D`) is only a **count** — the model has no notion of what
physical depth each output channel represents; it just produces `D`
regression channels. The mapping from output-channel index to a physical
depth in meters must be maintained by the training/evaluation pipeline,
**not** by this model's architecture.

For the current working specification, the intended target depths are:

```
10, 20, 50, 75, 100, 125, 150, 200, 300, 500, 700, 800, 1000  (meters)
```

These are **not assumed to be finalized** — they depend on the
preprocessing configuration. If useful, `ModelConfig.target_depths` can
optionally carry this list as metadata (validated to have exactly
`out_depths` entries if provided), purely for bookkeeping — it is never
read by the model's forward pass or architecture logic.

## 5. Repository Structure

```
models/
    __init__.py
    cbam.py          # ChannelAttention, SpatialAttention, CBAM
    blocks.py         # ConvBlock, Downsample, Upsample, EncoderBlock, DecoderBlock
    cbam_unet.py       # CBAMUNet — full model
configs/
    model_config.py    # ModelConfig dataclass
scripts/
    test_model.py      # shape/sanity/backward-pass/embedding test suite
README.md
```

## 6. How to Run the Tests

```bash
cd oceanembed
python scripts/test_model.py
```

The script runs, in order:

1. **7 forward-pass shape/sanity cases** (the original 6 plus a new
   GroupNorm case): configurable `C` (5, 8, 9), configurable `D` (13,
   20), a non-square size (80×120), the 301×481 non-power-of-2-divisible
   stress test, a 4-stage stride/transpose variant, CBAM enabled and
   disabled, and GroupNorm at batch size 1. Each checks output shape
   equals `[B, D, H, W]` and `torch.isfinite` passes on the output.
2. **3 backward-pass / optimizer-step cases** (new): forward → MSE loss
   → `loss.backward()` → check every trainable parameter received a
   finite gradient → one `torch.optim.Adam` step, run on the primary
   spec, the 301×481-derived config, and the GroupNorm config.
3. **2 embedding-extraction cases** (new): calls
   `forward_with_embedding()` on the primary spec and on an odd-spatial-
   dimension input, and reports input/embedding/output shapes.
4. A full architecture summary (parameter counts per module) for the
   primary spec.

## 7. Test Results (software-level only — NOT scientific validation)

Environment: Python 3.12.3, PyTorch 2.14.0+cu130 (CPU execution used for
this test run).

Command:
```bash
python scripts/test_model.py
```

**Forward-pass cases — all 7 passed** (shape match, finite output):

| Case | Input | Output | Params |
|---|---|---|---|
| Current spec (C=5, D=13) | `[2, 5, 64, 64]` | `[2, 13, 64, 64]` | 3,454,927 |
| Alternate config (C=9, D=20) | `[1, 9, 96, 96]` | `[1, 20, 96, 96]` | 1,946,089 |
| Non-square size | `[2, 5, 80, 120]` | `[2, 13, 80, 120]` | 3,454,927 |
| Odd size 301×481 | `[1, 5, 301, 481]` | `[1, 13, 301, 481]` | 866,053 |
| Stride/transpose variant, 4 stages | `[2, 5, 67, 53]` | `[2, 13, 67, 53]` | 4,519,145 |
| CBAM disabled | `[2, 5, 64, 64]` | `[2, 13, 64, 64]` | 3,409,933 |
| **GroupNorm (new), B=1, C=8** | `[1, 8, 64, 64]` | `[1, 13, 64, 64]` | 1,945,698 |

**Backward-pass / optimizer-step cases — all 3 PASS:**

| Case | Loss (MSE, random data) | Grad check | Adam step |
|---|---|---|---|
| Current spec, `[2,5,64,64]` | 1.009605 (finite) | PASS (91/91 params) | PASS |
| 301×481-derived, `[1,5,61,97]` | 1.045344 (finite) | PASS (91/91 params) | PASS |
| GroupNorm, `[1,8,64,64]` | 1.031174 (finite) | PASS (91/91 params) | PASS |

All trainable parameters received finite gradients in every case (0
missing, 0 NaN/Inf) — this only confirms the computation graph is
correctly connected; it says nothing about whether the model would learn
anything useful from real data.

**Embedding-extraction cases — both PASS:**

| Case | Input shape | Embedding shape | Output shape |
|---|---|---|---|
| Current spec | `(2, 5, 64, 64)` | `(2, 256, 8, 8)` | `(2, 13, 64, 64)` |
| Odd spatial dims | `(1, 5, 301, 481)` | `(1, 128, 37, 60)` | `(1, 13, 301, 481)` |

The embedding is a spatial feature map in both cases (never flattened to
a vector), confirming `forward_with_embedding()` behaves correctly for
both regular and non-power-of-2-divisible input sizes.

Primary-spec (`C=5, D=13, num_stages=3, base_channels=32`) parameter
count: **3,454,927** (unchanged from the initial implementation — this
revision did not add parameters to the default configuration; GroupNorm
and the new metadata field are opt-in).

## 8. Hidden-assumptions review (this revision)

The full model was reviewed against the checklist below. No changes to
the core architecture were needed — only the items marked "fixed" were
touched:

- Hardcoded input channel counts: none found.
- Hardcoded output depth counts: none found.
- Hardcoded spatial dimensions: none found (H, W are read from the input
  tensor at runtime everywhere).
- Unsafe image-size assumptions: none beyond the already-handled
  non-divisible-by-`2^num_stages` case, which the pad/crop alignment
  logic covers.
- Skip-connection / decoder / CBAM channel counts: re-traced stage by
  stage; all consistent (`channels[i] -> channels[i+1]` in the encoder,
  mirrored in the decoder).
- Final activation: confirmed there is no sigmoid/softmax/clamp on the
  output path — the output head is a plain `nn.Conv2d(..., kernel_size=1)`.
- In-place ops / autograd hazards: `ReLU(inplace=True)` /
  `LeakyReLU(inplace=True)` are used inside `ConvBlock`, but each
  activation output is only consumed once downstream (by the next layer
  in the `Sequential`, or by CBAM's out-of-place multiplicative gating);
  no tensor that's still needed for backward is mutated in place. No bug
  found here.
- Unnecessary detaching: none found — `forward_with_embedding()` returns
  the embedding without `.detach()` (fixed/confirmed this revision, see
  §1 "Spatial embedding").
- **Fixed:** `SpatialAttention`'s kernel-size check used a bare `assert`
  (stripped under `python -O`); replaced with an explicit `ValueError`.
- **Fixed:** `ModelConfig` validated only a subset of fields; extended to
  cover all fields listed in §3.

## 9. Remaining integration requirements

- Confirm the final `in_channels` (`C`) once the preprocessing pipeline
  settles on the surface-variable set and channel ordering, and
  construct `ModelConfig(in_channels=C, ...)` accordingly.
- Confirm the final channel *ordering* (which index is SST, SSS, etc.) —
  this model is order-agnostic, but the preprocessing pipeline and any
  downstream analysis must agree on a fixed ordering.
- Confirm the final `out_depths` (`D`) and, if useful, populate
  `ModelConfig.target_depths` with the confirmed physical depths once
  finalized (currently only a placeholder list is documented in §4, not
  assumed by the code).
- Decide on a masking strategy for land pixels / missing data (e.g. via
  the land-sea mask channel) — not implemented in this model; NaNs over
  land must be handled upstream in preprocessing before tensors reach
  this model, since the model has no NaN-handling logic of its own.
- Decide on `norm="batch"` vs `"group"` once the real training batch
  size is known — this is a dataset/hardware decision to make later, not
  something this model can resolve on its own.
- Training loop, loss function choice (beyond the sanity-test MSE),
  learning-rate schedule, dataset loader, and evaluation metrics remain
  unimplemented, per the brief.
