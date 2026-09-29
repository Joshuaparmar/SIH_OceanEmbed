# SSS Preprocessing -- SIH26066 / OceanEmbed

**Scope: 2025 internal round only.** This is a standalone Sea Surface
Salinity (SSS) preprocessing pipeline, separate from and independent of
the existing temperature preprocessing pipeline. It does not use, read,
or modify anything in `temperature_preprocessing/`. The old training
plan (2021-2023 train / 2024 cross-validation / 2025 test) is **not**
relevant here -- this pipeline only ever processes the single 2025 raw
file.

## What this does

Reads the raw CNR SSS product (`sos`, on a ~0.125 deg native grid,
5.0625-29.9375N x 45.0625-104.9375E, 365 daily 2025 timesteps,
singleton surface depth), validates it, drops the depth dimension, and
regrids it onto OceanEmbed's official 0.25 deg / 101x241 project grid
(5-30N x 45-105E), using a **NaN-aware, normalized weighted bilinear
interpolation** that never fabricates data over land or satellite
coverage gaps.

## Files

```
preprocessing/sss_preprocess_trial1/
    config_sss.py       All tunable settings: paths, region, dates, thresholds.
    netcdf_utils.py      Detection, validation, regridding, I/O, reopen-verification.
    preprocess_sss.py    CLI entry point / orchestration.
    README.md            This file.
```

## Why this is NOT a copy of the temperature pipeline's regridder

The temperature pipeline regrids with plain `xarray.interp(method=
"linear")`. That is fine for temperature (NaN only over land, mostly-
valid ocean). It is **not** appropriate for SSS: ~50.6% of the raw file
is invalid (a spatially-fixed satellite-coverage/land pattern -- same
49.36% valid fraction at *every single timestep*, confirmed by the
per-timestep inspection this pipeline runs and logs, not assumed).
Plain linear interpolation on an array containing that much NaN lets a
single invalid neighbour in a target cell's 4-corner stencil silently
NaN the result (or, depending on the exact interpolator, silently
contaminate it) with no record of how much real support the answer
actually had.

Instead, `netcdf_utils.regrid_weighted_linear()` implements a masked /
normalized weighted interpolation:

1. `finite_mask = isfinite(source values)`
2. `values_filled = where(finite_mask, values, 0.0)`
3. Interpolate `values_filled` onto the target grid -> weighted numerator
4. Interpolate `finite_mask.astype(float)` onto the target grid -> weight
5. `result = numerator / weight`
6. Any target cell where `weight < min_valid_weight` (default **0.5**,
   see "Unresolved decisions" below) is forced to NaN, regardless of
   what step 5 computed
7. `sss_valid_mask = (weight >= min_valid_weight) & isfinite(result)`

Both interpolators (`scipy.interpolate.RegularGridInterpolator`) are
built with `bounds_error=False, fill_value=0.0`, i.e. **extrapolation is
disabled**: a target point outside the native grid's convex hull gets
weight 0 from both interpolators, so it becomes NaN via the same
divide-by-zero guard as an interior all-missing cell -- not a special
case, not a guessed edge value.

Any failure in this step (missing scipy, bad coordinates, a fully-
invalid result) is a **fatal `RuntimeError`/`ValueError`** -- the
pipeline never silently falls back to the native grid or a different
method.

## Boundary handling (5N/30N/45E/105E)

The native grid does **not** reach the requested region's exact edges:

| | native | target |
|---|---|---|
| latitude | 5.0625 -- 29.9375 | 5.00 -- 30.00 |
| longitude | 45.0625 -- 104.9375 | 45.00 -- 105.00 |

The gap on each side (0.0625 deg = half the native 0.125 deg spacing)
means the **outermost target row (lat=5.00 and lat=30.00) and outermost
target column (lon=45.00 and lon=105.00) are entirely outside the
native grid's convex hull**, and are therefore **entirely NaN /
invalid** after regridding -- confirmed by a synthetic-data test run of
this exact code (see "Testing performed" below). This is deliberate,
documented, conservative behavior (no-extrapolation policy), not a
defect. `config_sss.allow_edge_nearest_fill` exists as a placeholder
for a possible future nearest-neighbor edge-extension policy, but
defaults to **off** and is **not implemented** -- turning it on
currently has no effect. Do not enable a workaround for this without an
explicit project decision (see below).

## Missing-value / validity-mask meaning

`sss_valid_mask`:
- **1** = valid, finite, regridded SSS value with sufficient
  interpolation support (`weight >= min_valid_weight`)
- **0** = invalid -- missing source data, land, outside native grid
  coverage (no extrapolation), or insufficient interpolation support

Missing/invalid cells are **never** filled with zero, a mean, or any
fabricated value, at any stage. `fill_enabled` in the config exists
only as an explicit, off-by-default, documented escape hatch and is not
used by the official path.

## Unit handling

The source `sos` variable's `units` attribute is `".001"`. This
pipeline preserves that string **verbatim** in the output's `units`
attribute and in provenance metadata. It does **not** automatically
relabel it as PSU (or anything else) -- see "Unresolved decisions"
below.

## CLI

Dry run (validates + reports, writes nothing):

```powershell
python .\preprocessing\sss_preprocess_trial1\preprocess_sss.py `
  --input ".\data\raw\sss\test\2025\sss_2025.nc" `
  --output-dir ".\data\processed\sss\2025" `
  --dry-run
```

Actual preprocessing:

```powershell
python .\preprocessing\sss_preprocess_trial1\preprocess_sss.py `
  --input ".\data\raw\sss\test\2025\sss_2025.nc" `
  --output-dir ".\data\processed\sss\2025"
```

If `data\processed\sss\2025\sss_2025_0p25deg.nc` already exists, the
run refuses and exits with `FileExistsError` unless `--overwrite` is
also passed.

Other flags: `--region "lat_min,lat_max,lon_min,lon_max"`,
`--target-resolution <deg>`, `--date-range "YYYY-MM-DD,YYYY-MM-DD"`,
`--regrid-method weighted_linear` (only supported value),
`--min-valid-weight <0-1>`, `--allow-partial-time`, `--no-plot`,
`--no-npy`, `--log-level`.

## Output

```
data/processed/sss/2025/
    sss_2025_0p25deg.nc              # Dataset: sss (time,lat,lon) float32, sss_valid_mask (time,lat,lon) uint8
    sss_2025_0p25deg.npy             # optional companion, = sss.values
    sss_2025_0p25deg_valid_mask.npy  # optional companion, = sss_valid_mask.values
    sss_2025_0p25deg_metadata.json   # full provenance (see below)
    sss_diagnostic.png               # 3-panel: t=0 SSS map, t=0 validity mask, domain-mean valid fraction over time
```

Expected shape for the real 2025 file: `(365, 101, 241)`.

The metadata JSON records: project name/ID, source file path + `sos`
variable attrs/encoding, original dims, native and target resolution
and region, regridding method/support-threshold/extrapolation policy,
missing-data policy, processing timestamp, actual matched time range,
source units + interpretation note, library versions, pre- and
post-regrid missing-value stats, implausible-value check results, the
reopen-validation result (`validation_status`), and confirmation that
the raw input file's size/mtime were unchanged
(`raw_input_file_unchanged`).

## Unresolved decisions (need project sign-off)

These are explicitly flagged, not silently decided:

1. **`min_valid_weight = 0.5` (default).** How much real interpolation
   support a target cell needs to be kept as valid, vs. discarded, is a
   genuine trade-off between spatial coverage and conservativeness that
   this pipeline cannot resolve on its own. Override with
   `--min-valid-weight`.
2. **Unit interpretation of `".001"`.** Not auto-converted or relabeled
   to PSU. If the project confirms (from actual source documentation,
   not inference) that this is intended as practical-salinity-unit
   equivalent, that confirmation -- and any resulting relabeling -- needs
   to be made explicitly, not assumed by this pipeline.
3. **Edge-cell policy.** Currently: outermost row/column = NaN, always.
   No nearest-neighbor edge extension is implemented. If the project
   wants the 5.00N/30.00N/45.00E/105.00E edges populated via a
   documented, bounded extrapolation instead of left NaN, that's a new
   scientific decision requiring explicit approval before implementing.
4. **`.npy` companion outputs.** Written by default for parity with the
   temperature pipeline's output set, but no downstream SSS consumer/
   loader has been confirmed to actually need them alongside the
   self-describing NetCDF. Can be disabled with `--no-npy`.
5. **Output coordinate/variable naming** (`sss`, `sss_valid_mask`,
   `time`/`latitude`/`longitude`) is the obvious/conventional choice,
   not verified against a specific downstream OceanEmbed loader
   contract.
6. **Confirmed vs. assumed:** unlike temperature's config, `sss_2025.nc`'s
   file path, variable name (`sos`), date range (2025-01-01 --
   2025-12-31, 365 daily steps), fixed 49.36% valid fraction, and
   `_FillValue`/`scale_factor`/`add_offset` were all directly inspected
   in the brief. The pipeline still re-verifies every one of these
   against the real file at runtime rather than trusting the brief
   blindly (see `preprocess_sss.py` logging), and will raise a clear
   error if the real file doesn't actually match what was reported.

## Testing performed (IMPORTANT -- read before trusting "it works")

**This code has not yet been run against the real 2025 SSS file** --
the working environment for this task explicitly has no access to it
(`data/raw/sss/test/2025/sss_2025.nc` is not reachable from here).

What *has* been done: a synthetic NetCDF file was generated locally,
matching the real file's structure, dimension names, coordinate
spacing/range convention (`5.0625..29.9375` step `0.125`, etc.),
variable attrs (`long_name`, `standard_name`, `units=".001"`), encoding
(`_FillValue=-32767`, `scale_factor=1.0`, `add_offset=0.0`), and global
attrs (`Conventions`, `title`, `institution`, `product_version`,
`software_version`), but with only 5 timesteps (for speed) and a
synthetic ~49.36%-valid spatial mask (for a realistic missing-data
test, not the real land/coverage pattern). Against that synthetic file,
this exact code was run end-to-end:

- `--dry-run`: completed, produced a dry-run JSON report, wrote no
  processed output.
- Full run: completed, wrote `sss_2025_0p25deg.nc` + `.npy` companions
  + metadata JSON, **reopened the NetCDF from a fresh `xr.open_dataset`
  call and validated** dims/shape/dtype/mask-consistency/grid spacing
  (all passed), and confirmed the raw (synthetic) input file's
  size/mtime were unchanged after the run.
- Verified directly on the written output: the outermost lat/lon
  row/column are all-NaN (boundary/no-extrapolation behavior working as
  designed), an interior row has real (non-all-NaN) values, and
  `sss_valid_mask` is exactly `np.isfinite(sss)` everywhere.
- Verified the overwrite guard: a second run without `--overwrite`
  against an existing output correctly refuses with `FileExistsError`.

**This is a code-correctness / mechanics check on synthetic data, not
scientific validation on the real dataset.** In particular: the actual
~365-day run time/memory on the full 200x480 native grid, the true
spatial pattern of the 49.36% valid fraction (is it really a clean land
mask, or does it also include time-invariant sensor artifacts?), and
whether `min_valid_weight=0.5` produces a scientifically sensible
coverage fraction on the *real* coverage pattern, are all still open
until run against the real file. Please run the dry-run first, review
its report, then run for real and share the log/output back for a
second look before treating this as production-ready.
