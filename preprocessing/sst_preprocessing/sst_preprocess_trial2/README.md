# SST preprocessing (SIH26066 / OceanEmbed)

Preprocessing pipeline for the OSTIA L4 SST (`analysed_sst`) surface
input. Reads the raw OSTIA/CMEMS NetCDF file(s), validates and reports
real structure, subsets to the configured region/time, checks and (if
needed) explicitly converts units, regrids to the official
0.25 x 0.25 deg OceanEmbed common grid, computes statistics and
optional normalization, standardizes coordinate names, and writes a
clean, model-ready SST dataset.

This folder only touches SST. It does not create or modify the
`preprocessing/temp_preprocess_trial3/` (subsurface temperature)
pipeline, or SSS/SSH/currents/winds/land-mask preprocessing, or any
model-training/DataLoader code.

**Architecture note:** this is a fresh implementation, not a copy of
`temp_preprocess_trial3/`. It deliberately reuses that pipeline's
*design patterns* -- dataclass config with `*_candidates` discovery,
detect-don't-assume coordinate/variable handling, boolean-mask
subsetting instead of `slice()`, fatal-by-design regridding with no
silent native-grid fallback, atomic writes, reopen-and-verify output
checking -- for project-wide consistency. It does **not** import
anything from that folder. What's genuinely different here: no depth
dimension (SST is `(time, latitude, longitude)`, not `(time, depth,
latitude, longitude)`); explicit Kelvin&harr;Celsius unit handling;
land-mask-aware missing-value reporting; and a normalization/statistics
step (Section I of the brief), which the temperature pipeline doesn't
have.

## v0.2.0 upgrade (this round)

This pass reviewed the pipeline against `temp_preprocess_trial3` as an
engineering-discipline benchmark and made **focused, targeted patches**
-- not a rewrite -- to close specific gaps. Summary (full detail in
each relevant section below):

1. **Memory-safe diagnostics.** `inspect_missing_values()` and
   `flag_implausible_values()` now accept an `xr.DataArray` and use
   dask-lazy reductions (`.count()`, `.min()`, `.max()`, `.mean()`,
   `.std()`, boolean `.sum()`) instead of forcing `da.values` first.
   The call sites that mattered most were the pre-regrid checks, which
   previously materialized the **subsetted-but-not-yet-regridded**
   array -- at OSTIA's native ~0.05 deg resolution over the full
   5-30N/45-105E region for a full year, that's on the order of 2x10^8
   cells (roughly 800 MB-1 GB as float32), too large to comfortably
   hold as one in-memory numpy array just to count NaNs or check a
   value range.
2. **Explicit batched regridding.** `regrid_to_target_grid()` now
   optionally splits the time dimension into batches
   (`config.regrid_time_batch_size`), interpolating and materializing
   one batch at a time and concatenating the results, bounding peak
   memory to one batch's worth of native-resolution data rather than
   the whole array at once. Verified numerically identical to
   unbatched regridding on a test fixture (not a real file -- see
   Testing).
3. **Correct end-date handling.** Time selection changed from
   `time <= pd.Timestamp(end_date)` (which silently excludes any
   timestamp on `end_date` stamped later than midnight -- a real risk
   for an L4 product that might stamp daily values at noon or another
   fixed hour) to an end-EXCLUSIVE interval
   (`time < pd.Timestamp(end_date) + pd.Timedelta(days=1)`), matching
   exactly what the task brief specified. The temporal-coverage
   **completeness check** had the identical bug on the start side and
   was fixed the same way (both bounds now compared by calendar date,
   not raw timestamp) -- found and fixed during this round's fixture
   testing (see Testing).
4. **Strict official spatial coverage.** New
   `verify_official_spatial_coverage()`: an official (regridded) run
   now hard-fails if the source doesn't fully cover the requested
   region, independent of `allow_partial_spatial_overlap` (which only
   governs a lower-level, non-fatal-by-default check).  A
   `--no-regrid` diagnostic run is exempt (never "official" regardless
   of coverage).
5. **Stricter unit validation.** `unit_conversion_policy="auto"` now
   **fails** (not just warns) if the source's `units` attribute is
   missing or unrecognized, by default (`fail_on_unrecognized_units`).
   Previously this silently skipped conversion with only a warning,
   which risked shipping output under an unconfirmed unit label.
6. **Three-stage missingness diagnostics.** Missingness is now reported
   before subsetting (full native extent, best-effort/non-fatal),
   after subsetting (still pre-regrid), and after regridding --
   previously only pre-regrid and final (post-regrid) were reported.
7. **Accurate land/coastal documentation.** `valid_mask`'s description
   and a new `land_ocean_mask_caveat` global attribute now explicitly
   state that it is a single finite-value mask (not a verified
   land-sea mask), that `xarray.interp()` is not coastline-aware, and
   that a detected source mask variable (if any) is not currently
   folded into it.
8. **Explicit input/output collision check.** New
   `check_input_output_collision()`, run before anything is opened or
   written: fails if `output_dir` is the same directory as an input
   file, is literally an input file path, or an input file is located
   inside `output_dir`.
9. Everything from the previous round is unchanged unless noted above:
   region (5-30N/45-105E), target resolution (0.25 deg, 101x241),
   fatal-on-regrid-failure with no native-grid fallback, no synthetic
   data, no blind NaN filling, atomic writes, output reopen-and-verify,
   dry-run behavior.

**Two real bugs were found and fixed during this round's fixture
testing** (not just designed-in from the start): the end-date
exclusivity issue had a symmetric bug on the start-date side of the
*coverage-completeness check* specifically (the selection mask itself
only had the end-side issue) that was only caught by testing against a
fixture with noon-stamped timestamps; and the temporal-coverage check
was, before this round, checking the combined lat+lon+time coverage
flag, which meant a purely spatial (longitude) coverage gap was being
misreported as a temporal coverage problem. Both are described in
detail in "Testing performed" below.

## Run it

```bash
# Inspect/validate/regrid and see the would-be output shape -- writes nothing
python preprocessing/sst/preprocess_sst.py --dry-run

# Official run: subset to 5-30N/45-105E, regrid to 0.25 deg, write outputs
python preprocessing/sst/preprocess_sst.py --date-range-confirmed

# Point at a specific file or directory
python preprocessing/sst/preprocess_sst.py --input data/raw/sst/test/2025/METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_30.02E-119.97E_39.97S-29.98N_2025-01-01-2025-12-31.nc --date-range-confirmed

# Non-official native-grid diagnostic run (for comparison only)
python preprocessing/sst/preprocess_sst.py --no-regrid --date-range-confirmed
```

Run `--help` for the full flag list (region bounds, unit policy,
normalization, missing-value strictness, regrid method, logging, etc).
`--date-range-confirmed` marks the default full-2025-calendar-year
range as an intentional choice rather than an unconfirmed assumption
(see "Assumptions" below) -- it does not change what range is used,
only how the run is logged/reported.

### Inspection / dry-run mode

```bash
python preprocessing/sst/preprocess_sst.py --dry-run --input <path-or-dir>
```

Opens the file(s), detects variable/coordinate names, validates
coordinates, subsets, checks units, checks physical plausibility,
regrids, computes statistics, and writes `sst_dryrun_report.json` with
everything that *would* be produced -- without writing
`sst_target.nc/.npy/mask/stats`.

## Input discovery

Priority: `--input <file-or-directory>` (exact, no fallback) >
`--input-glob <pattern>` (all matches, combined) >
`sst_input_candidates` (exact filenames in `config_sst.py`) >
`sst_input_glob` (default `data/raw/sst/**/*.nc`). A directory passed
via `--input` is globbed for `*.nc` inside it. More than one matched
file is opened together via `xr.open_mfdataset(combine="by_coords")`
-- a year of daily SST is sometimes split across files.

**Real filename observed for this project** (per the task brief, not
independently verified against the file's contents in this
environment -- the file itself was reported as ~1.71 GB and too large
to upload here):

```
METOFFICE-GLO-SST-L4-REP-OBS-SST_analysed_sst_30.02E-119.97E_39.97S-29.98N_2025-01-01-2025-12-31.nc
```

located (on the reporting machine) at
`.../OceanEmbed/data/raw/sst/test/2025/`. The default
`sst_input_glob = "data/raw/sst/**/*.nc"` will match this file if the
project is run from `OceanEmbed/` as the working directory / project
root; no filename is hard-coded. The filename's own embedded region
(30.02E-119.97E, 39.97S-29.98N) comfortably covers the required
5-30N/45-105E OceanEmbed region -- but this is read from the filename
as reported, not verified against the file's actual coordinate values,
since the file could not be opened here.

## Coordinate / variable detection

`sst_var_candidates` tries `analysed_sst` first (the confirmed
CMEMS/OSTIA name), then a few fallbacks -- never assumed present
without checking `ds.data_vars`. Same pattern for time/latitude/
longitude. Optional companion variables (`mask`, `analysis_error`,
`sea_ice_fraction`) are detected if present and reported, but are not
required.

## Units: explicit, never blind

OSTIA `analysed_sst` is *conventionally* Kelvin, but this pipeline
never assumes that -- `check_units()` reads the variable's actual
`units` attribute, and `convert_units_if_needed()` only converts
Kelvin&rarr;Celsius if that attribute is recognized as a Kelvin label
(case-insensitive: `K`, `kelvin`, `degrees_K`, ...). If units are
missing or unrecognized, **no conversion happens** and this is recorded
as a warning -- guessing "probably Kelvin" would be exactly the blind
conversion the brief prohibits. `unit_conversion_policy` (`config_sst.py`
/ `--unit-policy`) has three modes:

| policy | behavior |
|---|---|
| `auto` (default) | Kelvin -> convert; Celsius -> keep as-is; missing/unrecognized -> **FAILS the run** (`fail_on_unrecognized_units=True` by default) rather than guessing or silently skipping |
| `never` | never convert, write values exactly as stored (no failure either -- an explicit, permissive choice) |
| `force_k_to_c` | always subtract 273.15, regardless of the units attribute -- an **expert override**: use only if you've independently confirmed (from product docs, not guessed) that the data is Kelvin despite a missing/wrong attribute; logged as a warning and recorded in metadata |

Set `fail_on_unrecognized_units=False` to downgrade the missing/
unrecognized case from a hard failure back to a warning-only skip --
not recommended for an official run, since it means SST could be
written under a unit label that was never actually confirmed.

Every run's metadata records `unit_conversion` (`declared_units`,
`policy`, `converted`, `output_units`, `note`) so the decision made is
always traceable, not implicit.

## Regridding: fatal on failure, exactly like the temperature pipeline

Same design as `temp_preprocess_trial3`: `netcdf_utils.build_target_grid()`
builds the 0.25 deg target lat/lon **explicitly** from
`config.lat_min/lat_max/lon_min/lon_max/target_resolution_deg` (cell-center,
`np.linspace`, both endpoints included) -- for the default bounds this is
exactly **101 latitude x 241 longitude** points. `regrid_to_target_grid()`
uses `xarray.DataArray.interp()` (scipy-backed), `linear` by default
(`--regrid-method nearest` also supported), with `bounds_error=False,
fill_value=np.nan` so out-of-coverage target cells are NaN, never
extrapolated. A descending source coordinate is sorted first and this is
recorded, not silent.

**Any regridding failure is fatal** -- missing scipy, non-monotonic/
duplicate/non-finite source coordinates, an interpolation error, or an
entirely-NaN result (region doesn't overlap the source at all) all raise
and stop the pipeline before any output is written. There is no
try/except-and-continue anywhere around this call. `--no-regrid` is the
only way to get a native-grid output, and it is written to
`data/processed/sst/native_diagnostic/`, with `is_official_output=false`
recorded in both the metadata and the NetCDF's own global attributes, so
it can never be mistaken for the official output. Verified directly (see
Testing below): a duplicate source coordinate raises `ValueError`; a
target region entirely outside the source's coverage raises `RuntimeError`;
neither falls back to the native grid.

**Official spatial coverage is enforced strictly and separately from
regridding mechanics** (new this round): `verify_official_spatial_coverage()`
in `validation.py` fails an official (regridded) run if the source does
not fully cover the requested 5-30N/45-105E region -- this is checked
independently of `allow_partial_spatial_overlap` (which only governs a
lower-level, more lenient check inside `subset_region_time()`) and is
skipped entirely for a `--no-regrid` diagnostic run, which is never
"official" regardless of coverage. Set `require_full_official_coverage=False`
explicitly (and document why) if a deliberately partial official run is
ever actually needed -- not a default, an escape hatch.

**Regridding is processed in configurable time batches** (new this
round): `config.regrid_time_batch_size` (default `None` = no batching)
splits the time dimension into batches, interpolates and materializes
each batch independently, and concatenates the results -- so peak
memory during regridding is bounded by one batch's worth of
native-resolution source data plus its regridded output, not the whole
subsetted array at once. A full-year 2025 run at native ~0.05 deg
resolution over this region should set this explicitly (e.g. 30-60);
verified (on a synthetic fixture) to produce numerically identical
output to an unbatched run of the same data, and any failure on any
batch still aborts the entire regrid (no partial-success silent
continuation).

`verify_final_grid()` (in `validation.py`) re-checks spacing/range/
monotonicity from the **actual resulting coordinates** right before
writing (and again on the reopened file after writing) -- it does not
trust that the regrid call did what was asked.

## Missing values, land mask, and the valid_mask output

NaN is preserved by default everywhere (`fill_enabled=False`) -- a
`valid_mask` (1=valid/0=invalid) is always saved alongside. **This is a
single finite-value mask, not a verified land-sea mask**: land, a
missing ocean observation, a sea-ice/invalid region, and (after
regridding) a target cell that fell outside the source's actual
coverage all collapse to `valid_mask=0` and are NOT distinguished from
each other by this variable's value. `xarray.DataArray.interp()`
(linear or nearest) is **not a coastline-aware interpolation method** --
it has no notion of where the true coastline is, and can produce
interpolation artifacts for cells near a land/ocean boundary; this is
stated explicitly in the output's `land_ocean_mask_caveat` global
attribute and in `valid_mask`'s own `description` attribute (both new
this round -- the previous version's wording ambiguously implied more
land-awareness than the mask actually provides). If a source
categorical land/sea `mask` variable is present, it is detected and its
name recorded (`source_land_sea_mask_variable`), but it is **not**
currently folded into `valid_mask` or propagated to the output --
deliberately, since its actual value semantics (which integer means
land vs. ice vs. lake, etc.) have not been confirmed against a real
file; propagating an unverified categorical mask would risk being
wrong in a way that's hard to notice. This pipeline never fabricates an
SST value over land: `fill_enabled=True` is present in the config for
future use but deliberately raises `NotImplementedError` in this pass
rather than performing a blind/naive fill.

Three missingness stages are now reported (new this round, previously
only pre-regrid and final): **before subsetting** (full native extent,
lazy/dask-aware, best-effort -- logged but never fails the run on its
own), **after subsetting** (still pre-regrid, still potentially
native-resolution), and **after regridding** (the final, small array).
All three appear in `sst_metadata.json` as `missing_values_before_subsetting`
/ `missing_values_after_subsetting` / `missing_values_after_regridding`,
plus the earlier `sst_stats_finite_only_pre_regrid` / `_final` for the
physical-value range checks specifically.

`missing_value_policy`:
- `preserve` (default): report only, never fail on missingness alone.
- `reject_excessive`: fails the run if the NaN fraction exceeds
  `max_missing_fraction` (default 0.9) -- a backstop against silently
  shipping an almost-empty file (wrong variable/region), not a
  statement that SST land/ocean masking itself is a problem.

## Normalization and statistics (Section I)

Statistics (`mean`, `std`, `min`, `max`, `n_finite_values_used`) are
**always** computed and saved to `sst_statistics.json`, regardless of
whether normalization is applied to the written values --
`normalization_method` (`none` default / `zscore` / `minmax`,
`--normalize`) controls only whether those stats are also applied to
`sst_target.nc`'s values.

**Data-leakage note, stated plainly:** for this internal round there is
no separate train/test time split defined yet, so by default statistics
are computed from the **same** subsetted/regridded data being written --
this is recorded verbatim in `statistics_report.scope` in the output,
not hidden. `config.stats_time_range` (e.g.
`["2025-01-01", "2025-09-30"]`) exists specifically so that, once a
train/test split is defined, statistics can be recomputed from the
training portion only, avoiding leakage into a held-out test period.
NaN cells are never included in the mean/std/min/max computation and
are never given a fabricated normalized value (they stay NaN after
normalization; `valid_mask` still marks them 0).

When normalization is applied, the output `units` attribute is changed
to `"1 (dimensionless)"` (with the original physical unit preserved
separately as `pre_normalization_units`) -- writing normalized values
under a physical unit label like `degrees_C` would be a real, dangerous
mislabeling; this was caught and fixed during testing (see below).

## Output

`data/processed/sst/` (official) or
`data/processed/sst/native_diagnostic/` (`--no-regrid`):

| file | contents |
|---|---|
| `sst_target.nc` | `sst(time, latitude, longitude)` float32, zlib-compressed, + `valid_mask` uint8, full metadata attrs |
| `sst_target.npy` | same values as a plain NumPy array |
| `sst_valid_mask.npy` | same mask as a plain NumPy boolean array |
| `sst_metadata.json` | full run report: detected names, coordinate validation, region/time subsetting, unit conversion, missing-value/implausible-value stats (pre- and post-regrid), regridding report, final-grid check, alignment check, warnings |
| `sst_statistics.json` | mean/std/min/max for normalization, with scope documented |
| `sst_diagnostic.png` | optional 2-panel plot (first/last time step), best-effort, skipped if matplotlib is unavailable |
| `sst_dryrun_report.json` | written instead of the above, in `--dry-run` mode |
| `sst_error_report.json` | written instead of the above, on a validation failure |

Output variable name is `sst`; output coordinates are `time`,
`latitude`, `longitude` (no `depth` -- SST is a surface variable). This
matches the coordinate-naming convention the temperature pipeline
standardizes to (`time`/`latitude`/`longitude`/`depth`), so a later
common tensor-generation step can use one shared naming convention
across variables, minus the depth axis SST doesn't have.

## Configuration parameters (selected; see `config_sst.py` for all)

| field | default | purpose |
|---|---|---|
| `sst_input_glob` | `data/raw/sst/**/*.nc` | primary input discovery |
| `lat_min/lat_max/lon_min/lon_max` | 5 / 30 / 45 / 105 | OceanEmbed region (unchanged) |
| `start_date/end_date` | 2025-01-01 / 2025-12-31 | ASSUMED full-year default -- see Assumptions. Selection is end-DATE-inclusive (fixed this round -- see "v0.2.0 upgrade") |
| `regrid_to_quarter_degree` | `True` | official 0.25 deg output; fatal on failure |
| `regrid_method` | `linear` | or `nearest` |
| `regrid_time_batch_size` | `None` | **new this round**: batch size (time steps) for regridding; `None` = no batching. Set explicitly (e.g. 30-60) for a full-year native-resolution run |
| `require_full_official_coverage` | `True` | **new this round**: official (regridded) runs fail if the source doesn't fully cover the requested region, independent of `allow_partial_spatial_overlap` |
| `unit_conversion_policy` | `auto` | Kelvin&rarr;Celsius if units say Kelvin; Celsius kept as-is; missing/unrecognized **fails by default** (see `fail_on_unrecognized_units`) |
| `fail_on_unrecognized_units` | `True` | **new this round**: only applies under `unit_conversion_policy="auto"`; set `False` to downgrade to a warning (not recommended for an official run) |
| `missing_value_policy` | `preserve` | or `reject_excessive` |
| `normalization_method` | `none` | or `zscore` / `minmax` |
| `implausible_low_c` / `implausible_high_c` | -3.0 / 45.0 | physical plausibility range for SST |
| `max_implausible_fraction` | 0.05 | fatal backstop |
| `dry_run` | `False` | inspect only, write nothing |

## Dependencies

See `requirements.txt`: `xarray`, `numpy`, `pandas`, `netCDF4`, `scipy`
(hard dependency of the official regridding path), `dask[array]`
(optional but recommended, for `chunks="auto"` lazy opening),
`matplotlib` (optional, diagnostic PNG only).

## Memory / performance

`open_dataset()` opens with `chunks="auto"` (dask-backed, lazy) by
default so a full year of daily 0.05 deg SST over the OceanEmbed crop
isn't necessarily pulled fully into memory just to open and validate
structure. **This round's key correction**: opening lazily is not
sufficient by itself -- several downstream operations previously called
`.values` on the subsetted (still native-resolution, pre-regrid) array,
which forces full materialization regardless of how the dataset was
opened. That is now fixed:

- `inspect_missing_values()` / `flag_implausible_values()` accept an
  `xr.DataArray` and use dask-lazy reductions (`.count()`, `.min()`,
  `.max()`, `.mean()`, `.std()`, boolean `.sum()`) that stay lazy until
  `.compute()` is called on the scalar **result**, not the array.
  `median_finite` is the one exception (dask has no fully lazy exact
  median) -- it's computed only when the array is small/already
  materialized (post-regrid), and reported as skipped-for-size
  otherwise.
- Regridding itself is where materialization genuinely has to happen
  (`.interp()` needs real values) -- this is now done in configurable
  **time batches** (`config.regrid_time_batch_size`) rather than one
  call on the whole array, bounding peak memory to one batch's worth
  of native-resolution data. For a full 2025 run at native ~0.05 deg
  over 5-30N/45-105E, a batch of 30-60 days keeps each batch in the
  tens-of-MB range rather than the ~800 MB-1 GB a single unbatched call
  on the full year would require.
- The final regridded array is small regardless of batching -- for a
  full 2025 x 101x241 float32 array that's roughly 39 MB (365 x 101 x
  241 x 4 bytes) -- so everything downstream of regridding (statistics,
  normalization, writing) operates on an in-memory array without
  further batching being necessary.

Output is zlib-compressed (`complevel=4` default). If a much larger
multi-year run is ever needed beyond what batched regridding alone
handles, the natural further mitigation is processing whole months/
years as separate pipeline runs (already supported via `--start-date`/
`--end-date`) rather than a single very long run.

## Testing performed

**No real OSTIA/CMEMS SST file was available in this environment** --
the actual 2025 file (~1.71 GB) lives only on the reporting machine
(`C:\...\OceanEmbed\data\raw\sst\test\2025\...`) and was reported as
too large to upload here. No synthetic data was substituted for it or
claimed to be real.

**Test 1 -- static/code validation: DONE.**
- All 4 files (`config_sst.py`, `netcdf_utils.py`, `validation.py`,
  `preprocess_sst.py`) compile cleanly and import without error.
- `config_sst.CONFIG` loads with the documented defaults;
  `build_target_grid()` produces exactly 101 x 241 points, asserted.
- `--help` prints the full CLI.
- Invalid `--input` path: exit code 2, clear "no input file found"
  message, error report written.
- Invalid region config (`lat_min >= lat_max`): raises `ValueError`
  from `SSTConfig.__post_init__`, exit code 1 -- fails fast rather than
  silently continuing with a nonsensical region.
- Invalid-variable handling: verified with a real (small, synthetic)
  NetCDF whose SST variable was renamed to something unrecognized --
  fails with exit code 1 and a message naming exactly which candidates
  were tried and which variables actually exist.
- Invalid-region handling (zero spatial overlap): verified against the
  fixture file with a region requested far outside its coverage --
  fails with exit code 1 and a message giving the actual vs. requested
  extents.
- Regridding-failure behavior: verified directly against
  `regrid_to_target_grid()` -- a duplicate source coordinate raises
  `ValueError`; a target region entirely outside the source's coverage
  raises `RuntimeError`. Neither case falls back to the native grid or
  produces partial output.

**Test 2 -- small "real-shaped" data test: DONE, using a clearly-labeled
SYNTHETIC fixture, not real data** (the real file could not be
obtained in this environment -- see above). The fixture
(`make_fixture.py`, not included in the delivered pipeline folder) is
random noise on a coarse 0.5 deg grid, `analysed_sst` in Kelvin, with an
arbitrary NaN "land" stripe pattern, explicitly labeled in its own
NetCDF attributes as a placeholder. Verified against it:
- Dataset opening (single file and multi-file `open_mfdataset`).
- Variable detection (`analysed_sst` found; a renamed variable
  correctly fails).
- Coordinate normalization/validation (time/lat/lon detected; leap-year
  date range -- Feb 2024, 29 days -- correctly reports zero missing
  dates rather than assuming 365/366 days per year).
- Regional subsetting (correct original/requested/actual extent
  reporting).
- Unit handling: `units="kelvin"` correctly auto-detected and converted
  to Celsius; conversion explicitly recorded in metadata and NetCDF
  attrs.
- Regridding: both `linear` and `nearest` methods produce exactly
  101 x 241 @ 0.25 deg; `--no-regrid` correctly produces a
  differently-shaped, differently-located, clearly-marked non-official
  diagnostic output.
- Missing-value reporting (`reject_excessive` policy verified to raise
  when the threshold is exceeded).
- Directory-based input discovery (`--input <dir>`).
- Statistics computation and both normalization methods
  (`zscore` verified; NaN cells confirmed to remain NaN, not
  fabricated, after normalization).
- Output writing, reopening, and self-verification
  (`verify_output_file`) -- dtype, units, mask dtype, shape all
  confirmed on the file actually written to disk, not just the
  in-memory array.
- Raw input file confirmed byte-for-byte unchanged (`md5sum`) across
  every run.
- **Bug found and fixed during this testing**: normalized output was
  initially still labeled `units: degrees_C`, which is incorrect --
  fixed to `"1 (dimensionless)"` with the original unit preserved
  separately as `pre_normalization_units` (see "Normalization and
  statistics" above).

**Test 3 -- full real-data test on the actual 2025 SST dataset: NOT
DONE / PENDING.** The real file was not accessible in this
environment. This is the fastest way to close the remaining gap:
upload the real file (or a small spatial/temporal crop of it, e.g. one
month) and run `--dry-run` first, then a full official run, from a
`project_root` set to the `OceanEmbed/` directory (or point `--input`
directly at the file). Once available, that run should report: real
input/output dimensions, real time range and any actual gaps, real
grid resolution after regridding (should be 101 x 241 @ 0.25 deg),
real declared units (confirm Kelvin vs. already-Celsius) and whether
conversion actually fired, real NaN percentage (land fraction over
5-30N/45-105E is expected to be non-trivial, since this region
includes the Indian subcontinent, Southeast Asia, and parts of the
Arabian Peninsula), real numerical statistics, real output file size,
and real processing duration -- none of which can be honestly reported
without the file.

### v0.2.0 upgrade testing (this round)

Same environment constraint as before -- no real OSTIA file available,
so everything below is against synthetic fixtures built specifically
to exercise the new/changed logic, never claimed as real-file
validation:

- **Memory-safety refactor**: confirmed `inspect_missing_values`/
  `flag_implausible_values` now accept `xr.DataArray` and work
  correctly on both dask-backed and plain-numpy-backed arrays; full
  regression suite (leap year, multi-file open, normalization,
  invalid-variable/region, `--reject-excessive-missing`, directory
  input, raw-file-untouched checksum) re-run clean after the change.
- **Batched regridding**: a 20-day synthetic fixture regridded with
  `regrid_time_batch_size=6` (4 batches) produced a numerically
  **identical** result (`np.allclose(..., equal_nan=True)` verified
  True) to the same fixture regridded unbatched -- confirms batching
  is a pure memory optimization, not a behavior change.
- **End-date-inclusive fix**: a fixture with timestamps stamped at
  noon UTC (not midnight) on each day, requesting `--end-date` equal
  to the fixture's actual last day, initially still failed with
  "requested time range... not fully covered" even though the correct
  5 days were being selected. Root cause: the coverage-completeness
  check (not the selection mask, which was already correct) compared
  raw timestamps rather than calendar dates on the **start** side too.
  Fixed by normalizing both `actual_time.min()` and `actual_time.max()`
  to calendar dates before comparing against `start`/`end`. Re-verified
  passing after the fix.
- **Misattributed coverage error**: testing `--no-regrid` against a
  fixture with only partial longitude coverage initially still failed
  the run, with the error blaming "temporal coverage" -- root cause:
  `verify_temporal_coverage()` was checking the combined lat+lon+time
  flag rather than a time-specific one. Fixed to check
  `full_time_coverage` specifically; re-verified the diagnostic run
  then succeeds (partial spatial coverage is fine for `--no-regrid`)
  while the equivalent **official** run against the same fixture still
  correctly fails, now with the accurate spatial-coverage error from
  the new `verify_official_spatial_coverage()`.
- **Unit-validation strictness**: a fixture with no `units` attribute
  at all, run under the default `unit_conversion_policy="auto"`,
  correctly fails with a clear error explaining the three ways to
  proceed (`force_k_to_c`, `never`, or `fail_on_unrecognized_units=False`).
  The same fixture with `--unit-policy never` correctly succeeds.
- **Input/output collision check**: `--output-dir /tmp` against an
  input file located directly in `/tmp` correctly fails with exit
  code 1 *before* the dataset is opened or any file is written (checked
  no file appeared at the given path afterward).
- **Three-stage missingness**: confirmed all three
  (`missing_values_before_subsetting` / `_after_subsetting` /
  `_after_regridding`) appear correctly populated in
  `sst_metadata.json` on a full run.

## Status summary

| | status |
|---|---|
| Code complete | Yes -- all required files present and mutually consistent; v0.2.0 patches applied |
| Reviewed against `temp_preprocess_trial3` for engineering rigor | Yes -- see "v0.2.0 upgrade" above for gaps found and closed |
| Tested on synthetic/fixture data | Yes -- mechanics-level (Test 1 + Test 2 from the previous round) plus targeted tests for every v0.2.0 change (see "v0.2.0 upgrade testing" above), including two real bugs found and fixed by that testing |
| Tested on real SST data | **No** -- real file not accessible in this environment |
| Still pending | Test 3 (full real-2025-data run); confirming the real file's actual coordinate names/units/resolution/mask-variable-semantics match what was assumed; confirming `stats_time_range` once an actual train/test split is defined for OceanEmbed; choosing a real `regrid_time_batch_size` against the real file's actual memory behavior |

## Unresolved issues / known limitations

- `fill_enabled=True` is a documented no-op (`NotImplementedError`) in
  this pass -- see "Missing values" above. Not needed for the default,
  recommended configuration (`fill_enabled=False`).
- Statistics currently default to using the same data as the output
  (no train/test split defined yet for this internal round) -- see
  "Normalization and statistics" above; `stats_time_range` is the
  intended fix once a split exists.
- The exact real coordinate/variable names, units, and native
  resolution of the actual OSTIA file used by this project are not
  independently confirmed here -- `sst_var_candidates` /
  `*_name_candidates` / `unit_conversion_policy` are set from product
  documentation and the brief, not from having opened the real file.
  Run `--dry-run` against the real file first and check the logged
  "Detected SST variable" / "Detected coordinates" / "Units" lines
  before trusting a full run's output.
- `land_sea_mask`/`analysis_error`/`sea_ice_fraction` companion
  variables are detected and reported if present but deliberately NOT
  folded into `valid_mask` or propagated to the output yet -- their
  real value semantics (which integer means land/ice/lake, etc.) are
  not confirmed against a real file; see "Missing values" above. This
  is an explicit decision to leave a gap rather than guess.
- `regrid_time_batch_size` defaults to `None` (no batching) -- fine for
  a short test range, but a full-year native-resolution run should set
  this explicitly (e.g. 30-60). The right default for the real file
  can only be tuned against its actual memory footprint, which is not
  known here.
- The "before subsetting" missingness diagnostic is a full lazy
  reduction over the entire native-resolution file (not just the
  requested region) -- correct and memory-safe (dask-lazy), but on a
  genuinely large multi-year file this could take noticeably longer
  than the rest of the pipeline combined. It's best-effort/non-fatal
  (a failure there only logs a warning and continues) for exactly this
  reason, but its runtime on the real file is unverified.
- `verify_official_spatial_coverage()`'s tolerance for "full coverage"
  uses the same `final_grid_tolerance_deg` (1e-6 deg) as the final-grid
  spacing check -- reasonable for OSTIA's much finer native resolution,
  but not independently re-derived from first principles; worth a
  second look once the real file's native resolution is confirmed.
