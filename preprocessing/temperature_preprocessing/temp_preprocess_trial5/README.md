# Temperature target preprocessing (SIH26066 / OceanEmbed)

Preprocessing pipeline for the GLORYS `thetao` (potential temperature)
target. Reads the raw GLORYS NetCDF file(s), validates and reports
real structure, subsets to the configured region/time/depths, regrids
to the official 0.25 x 0.25 deg target grid, standardizes coordinate
names for Step 5, and writes a clean, model-ready subsurface
temperature target dataset.

**Status:** active dataset is **2025** (Jan-2026 prototype is
discontinued, see config). **The official output is regridded to
0.25 x 0.25 deg by default** -- native-resolution output is only ever
produced as an explicitly-requested, clearly-marked diagnostic run
(`--no-regrid`), never as "the" output.

This folder only touches the GLORYS temperature target. It does not
create or modify SST/SSS/SSH/currents/winds/ARGO preprocessing or any
model-training code.

## Run it

```bash
python preprocessing/temperature/preprocess_temperature.py --dry-run   # inspect/validate, no output written
python preprocessing/temperature/preprocess_temperature.py             # official run: regridded to 0.25 deg, fatal on regrid failure
python preprocessing/temperature/preprocess_temperature.py --no-regrid # NON-OFFICIAL native-grid diagnostic run
```

Input discovery, exit codes, and CLI flags are unchanged from the
previous pass except: `--regrid` is gone (regridding is now the
default); `--no-regrid` is the new explicit opt-out. Run `--help` for
everything.

## The official grid: 0.25 deg, explicit construction

`netcdf_utils.build_target_grid()` builds the target latitude/
longitude arrays **directly from `config.lat_min/lat_max/lon_min/
lon_max/target_resolution_deg`** -- never from the source file's own
subsetted extent, which could differ slightly from the requested
bounds. Convention: **cell-center, both endpoints included**:

```
n_points = round((max - min) / spacing) + 1
grid = np.linspace(min, max, n_points)
```

For the default bounds (5-30N, 45-105E, 0.25 deg) this gives exactly:

| | min | max | count | spacing |
|---|---|---|---|---|
| latitude | 5.0 | 30.0 | **101** | 0.25 deg |
| longitude | 45.0 | 105.0 | **241** | 0.25 deg |

`np.linspace` (not `np.arange`) on purpose: it hits the exact
endpoints with no floating-point drift from repeated addition, so the
count is exactly 101/241 every time, not "101 or 102 depending on
rounding."

## Regridding: method, extrapolation, and why failure is fatal

- **Method:** linear interpolation (`xarray.DataArray.interp`,
  scipy-backed), matching `config.regrid_method` (default `"linear"`)
  -- a reasonable default for a smooth scalar field like temperature;
  nearest-neighbor would be blocky, cubic can overshoot near
  land/NaN boundaries.
- **No extrapolation:** `kwargs={"bounds_error": False, "fill_value":
  np.nan}` is passed explicitly to the interpolator. A target cell
  outside the source data's actual coverage becomes NaN, never a
  guessed or edge-extended value. The fraction of NaN target cells
  after regridding is always computed and reported
  (`target_cells_invalid_fraction` in the metadata).
- **Source coordinates are validated immediately before interpolating**
  (numeric, 1-D, finite, no duplicates -- raised clearly if not). A
  descending source coordinate is sorted ascending first via
  `.sortby()` -- **recorded** in the metadata
  (`coordinates_sorted_ascending`), never silent.
- **Regridding failure is FATAL, by design, with no exceptions.**
  Missing scipy, non-monotonic/duplicate/non-finite source
  coordinates, an interpolation error, or an entirely-NaN result (the
  requested region doesn't overlap the source's real coverage at all)
  all raise and stop the pipeline before any output is written. There
  is deliberately no `try/except` anywhere around the regridding call
  that continues with the native grid on failure -- that pattern was
  in the previous version of this pipeline and has been removed.
- **`--no-regrid`** is the only way to get a native-grid output, and
  it is unambiguously marked as a diagnostic, not the official result
  (see below).

## Official vs. diagnostic output -- never confusable

| | Official (default) | Diagnostic (`--no-regrid`) |
|---|---|---|
| Directory | `data/processed/temperature/` | `data/processed/temperature/native_diagnostic/` |
| Grid | 0.25 x 0.25 deg, 101 x 241 | native GLORYS resolution |
| `is_official_output` (metadata + `.nc` global attr) | `true` | `false`, with an explicit warning string |
| Regridding failure | fatal, no output written | n/a (not regridded) |

The two never share a directory or filename, including for
`--dry-run` and failure reports, so a diagnostic run can never end up
at the path Step 5 (or anyone else) expects the official output at.

## Post-regrid validation, before anything is written

`netcdf_utils.verify_final_grid()` runs right before writing and
**recomputes** spacing/range/count from the actual resulting
coordinates -- it does not trust `config` or assume the regrid call
did what it was asked. It raises if:
- latitude or longitude spacing isn't ~0.25 deg (tolerance
  `config.final_grid_tolerance_deg`, default 1e-6 deg) -- official
  runs only,
- the latitude/longitude range doesn't match the requested bounds --
  official runs only,
- depth count isn't exactly 15, or depth isn't strictly ascending --
  **always**, both modes,
- any coordinate has duplicates or isn't monotonic -- **always**,
- unexpected dimensions are present -- **always**.

The same check runs again after writing, against the **reopened**
file (`verify_output_file`), so the on-disk result is what's actually
validated, not just the in-memory array before the write.

## Pipeline order (why implausible-value checking moved)

```
open -> detect variable/coords -> validate coordinates -> subset
region/time -> select 15 depths -> [depth-count / time-coverage
strict checks] -> units check -> PHYSICAL-VALUE SAFETY CHECK (fatal
backstop) -> REGRID (fatal on failure) -> recompute missing/implausible
stats on final array -> compute valid_mask -> standardize coordinate
names -> verify_final_grid (fatal) -> write -> reopen and verify
```

The physical-value safety check now runs **before** regridding, not
after (as in the previous pass) -- an unmasked `_FillValue`/
`missing_value` sentinel (e.g. `1e20`) would otherwise get smeared
into real neighbouring ocean cells by linear interpolation, corrupting
them before there's ever a chance to catch it. Missing-value and
implausible-value stats are still recomputed on the **final** (post-
regrid) array too, for the metadata that actually describes what's
shipped.

## Memory: lazy/chunked missing-value checks, time-batched regridding

Confirmed against a real 2025 file: the previous version crashed with
`numpy._core._exceptions._ArrayMemoryError` trying to allocate 8.85 GiB
(`365 x 15 x 301 x 721` float64) inside `inspect_missing_values(depth_da.values)`
-- `.values` forces a dask-backed array fully into memory as one numpy
array. Fixed in two places:

- **Missing-value / implausible-value checks** (`inspect_missing_values`,
  `flag_implausible_values`, `compute_valid_mask` in `netcdf_utils.py`)
  now take the `xr.DataArray` directly, never `.values`, and use lazy
  reductions (`np.isfinite(da).sum()`, `da.where(mask).min()`, etc. --
  `np.isfinite` on purpose, not `da.isnull()`, since `isnull()` alone
  doesn't count `+/-Inf` as invalid and would have silently changed
  validation behavior), `.compute()`-ing only the small scalar result.
  **Verified at the exact real problem size** (`365 x 15 x 301 x 721`,
  dask-chunked, 8.85 GiB if materialized) in this environment (which
  has less total memory, 3.9 GiB, than that array would need
  eagerly): both functions completed successfully, peak RSS ~2.5 GiB.
- **Regridding** (`regrid_to_target_grid`) now processes the time
  dimension in batches (`config.regrid_time_batch_size`, default 30
  days / `--regrid-batch-size`) rather than interpolating the full
  cube in one `.interp()` call, which needs the whole native-resolution
  array in memory to build the interpolator regardless of chunking
  upstream. Horizontal interpolation for one day never depends on any
  other day, so this changes peak memory, not the numerical result --
  **verified**: batched (3 unequal batches) and non-batched regridding
  of the same data produce bit-identical output
  (`np.testing.assert_array_equal`). Also verified at real spatial
  resolution (301x721 native, 15 depths, two 30-day batches): completed
  in ~12s, peak RSS ~2.7 GiB.

**Known remaining risk, disclosed rather than silently left:** past
this point (final mask, `save_outputs`, the diagnostic plot), the
array is small on the official regridded path (post-regrid:
`365 x 15 x 101 x 241` float32, ~530 MB -- safe to materialize) but
would NOT be small on a **`--no-regrid` diagnostic run at full native
resolution**, which still forces `.values` in a few remaining spots.
Not fixed in this pass -- out of scope, since the official (regridded)
path is the one actually used, and a full non-dry-run hasn't been
requested yet. Worth revisiting before ever running a full
`--no-regrid` pass on the complete year.

## Nominal 0 m surface request vs. GLORYS's real shallowest level

Confirmed against a real 2025 file (run independently, not in this
environment -- see below): GLORYS essentially never has an exact 0 m
depth level; the shallowest model layer is a small positive depth
(observed: ~0.494 m). A literal `requested < available.min()` check
would reject the nominal "surface" (0 m) request as `OUT_OF_RANGE`,
which would drop the target depth count below 15.

`config.allow_surface_depth_mapping` (default `True`) is a **targeted**
exception for this specific case only -- not a general tolerance
increase, not `require_all_requested_depths=False`. It fires only when
a requested depth is both `<= config.surface_depth_max_m` (default
1.0 m) AND shallower than every available depth; it maps to the single
shallowest available level, recorded with status `SURFACE_MAPPED` in
the depth report -- distinct from `EXACT`/`NEAREST`, and counted as a
successful match by `verify_depth_count`, so the strict 15-depth
requirement is still satisfied. Every other requested depth is
completely unaffected and still goes through ordinary exact/nearest
matching with `depth_max_tolerance_m`. Set `allow_surface_depth_mapping
= False` to get the original strict behavior back exactly (verified:
reproduces `OUT_OF_RANGE` for 0 m).

The mapping is recorded in three places: the depth report (and hence
`temperature_metadata.json`'s `requested_vs_actual_depths`), and two
new NetCDF global attrs written directly onto the output file itself
-- `depth_selection_report_json` (the full requested-vs-actual list)
and `surface_depth_mapping` (a human-readable one-line summary, or
`"not applicable"` when no mapping occurred).

## Outputs

Same file set as before (`temperature_target.nc` + `.npy` + mask
`.npy` + `temperature_metadata.json` + optional diagnostic PNG +
dry-run/error reports), now also:
- **Compressed**: `zlib` compression (`complevel=4` by default) on
  both data variables in the NetCDF -- a full 2025 x 15-depth x
  101x241 float32 array is roughly half a gigabyte uncompressed;
  compression is a standard, dependency-free xarray/netCDF4 feature.
- **New global attrs**: `is_official_output`, `regridding_method`,
  `extrapolation_policy`, `requested_lat_bounds`/`requested_lon_bounds`,
  `actual_lat_bounds`/`actual_lon_bounds`, `source_dataset`
  (`"GLORYS (Copernicus Marine)"`), `source_variable` (`"thetao"`).
- **New metadata fields**: `regridding` (source/target resolution,
  method, extrapolation policy, sorted-coordinate record, invalid-
  cell fraction), `final_grid` (the recomputed spacing/range/count),
  separate `temperature_stats_finite_only_pre_regrid` /
  `_final` and `missing_values_pre_regrid` / `_final`.

## Missing values, mask, and Step 5 compatibility

Unchanged from the previous pass and still correct after adding
regridding: NaN preserved by default (`fill_enabled=False`), a
`valid_mask` (1=valid/0=invalid) always saved alongside, `thetao` ->
`temperature_target` renamed on write, coordinates standardized to
`time`/`depth`/`latitude`/`longitude` to match
`oceanembed_step5/config.py`'s `CoordNames` defaults, dtype `float32`,
depth strictly ascending. Step 5 computes its own valid-data mask via
`np.isfinite()` on the array directly -- it doesn't read `valid_mask`
-- which is exactly why `fill_enabled` must stay `False`: a NaN
target cell after regridding (outside source coverage) must reach
Step 5 as NaN, not as a filled number that would be silently treated
as a real observation. The one known mismatch (Step 5's
`target_paths` default doesn't point at this pipeline's output path)
is unchanged and still not fixed here -- see the final report.

## Dependencies and memory

**New hard dependency: `scipy`**, for the interpolation the official
0.25 deg path now always performs (installed in this environment;
not auto-installed by the script itself, per the constraint against
modifying the environment without permission -- if it's missing, the
regridding call fails with a clear message naming it as a likely
cause, and that failure is fatal as designed, not silently skipped).
`dask` is also required (for `xr.open_mfdataset` and optional chunked
opening) and was already a dependency from the previous pass.

**Memory**: a full 2025 x 15-depth x 101x241 float32 array is
~530 MB uncompressed in memory (365 x 15 x 101 x 241 x 4 bytes).
`open_dataset()` opens with `chunks="auto"` (dask-backed, lazy) by
default so the raw source file(s) aren't necessarily pulled fully
into memory just to open and validate structure, mirroring
`oceanembed_step5/data_loader.py`'s own use of chunked xarray objects.
The regridding step itself still operates on the full subsetted
array in memory rather than in chunks -- deliberately not adding
chunked/`map_blocks`-based interpolation, which would be a much
larger, riskier change for a benefit that's currently unmeasured (no
real file to profile against). If ~500 MB-1 GB in-memory arrays turn
out to be a real problem in practice, the natural mitigation without a
rewrite is processing the year in time slices (e.g. loop over months,
regrid each, concatenate/write incrementally) -- documented here as a
recommendation, not implemented, since there's no evidence yet that
it's needed.

## What has and hasn't actually been run

Same honest baseline as every previous pass: **no real 2025 GLORYS
file was available in this environment** (checked again: no uploads,
no new repo commits, no matching file anywhere on disk). No
synthetic/fabricated NetCDF was created to get around that.

**Verified for real, this pass** (mechanics-level, plain/random
placeholder arrays -- never anything resembling a real ocean
dataset):
- `build_target_grid`: produces exactly 101 lat x 241 lon points,
  exact bounds, exact 0.25 deg spacing -- checked by assertion, not
  eyeballed.
- `regrid_to_target_grid` against a small placeholder source array
  (random noise, deliberately descending latitude, deliberately
  smaller than the full target region): confirmed the descending
  coordinate gets sorted and recorded, confirmed cells outside source
  coverage come back NaN (not extrapolated) while cells inside come
  back finite (real interpolation happened), confirmed a duplicate
  source coordinate raises `ValueError`, confirmed a target region
  entirely outside the source's coverage raises `RuntimeError`.
- `verify_final_grid`: a correct 101x241/15-depth grid passes cleanly;
  wrong spacing, wrong depth count, duplicate coordinates, and
  non-monotonic depth each independently raise with a specific
  message; native-diagnostic mode correctly skips spacing/range
  enforcement while still enforcing depth count and monotonicity.
- Config validation: new fields (`final_grid_tolerance_deg`,
  `netcdf_complevel`) reject bad values; defaults confirmed correct
  (`regrid_to_quarter_degree=True`, `native_diagnostic_subdir`,
  `use_dask_chunking=True`, `enable_netcdf_compression=True`).
- All 3 files compile cleanly after every edit. CLI (`--help`, every
  flag including the new `--no-regrid`) parses correctly.
- Both modes' missing-file and dry-run paths run end-to-end and
  correctly separate their output/report locations (`native_diagnostic/`
  vs. official) -- this specific check caught and fixed a real bug:
  the early failure-report paths weren't originally mode-aware.
- Confirmed by direct code inspection (grep for every write call in
  the codebase): every write targets only `out_dir`/temp paths derived
  from it -- never an input path. Raw input files are not modified.

**Not verified:** the actual detection/validation/subsetting/depth-
selection/regridding/output path against a real 2025 file, because
none exists here to run it against -- this includes the actual
interpolation numerics on real GLORYS-shaped data, actual memory
behavior at full scale, and actual Step 5 loading of a real output
file. The regridding function's *mechanics* were verified (see
above); a real GLORYS file is still the only way to verify the
*numerics and performance* on real ocean data.

**Fastest way to close the gap:** upload a real 2025 GLORYS file (or
a small crop) and run `--dry-run` first, then a full official run.
