# SSS Preprocessing -- SIH26066 / OceanEmbed (year-agnostic)

Regrids the CNR raw SSS product (`sos` or CF-equivalent, ~0.125 deg
native grid) onto OceanEmbed's official 0.25 deg / 101x241 project grid
(5-30N x 45-105E), for **any year's** raw file, using a NaN-aware
normalized weighted bilinear interpolation that never fabricates data.

## Changelog

**0.2.1 (this version) -- reject mixed-calendar-year input by default.**
- `detect_year()` now raises `ValueError` (before any further processing)
  if a single input file's `time` coordinate spans more than one
  calendar year, instead of silently assigning the whole file to its
  dominant year. The error reports every year found, its timestamp
  count, the dominant year, and the input filename.
- Added `--allow-multi-year` (default **OFF**) as an explicit, advanced
  override: assigns the whole file to its dominant year (does **not**
  split it), with a loud warning recorded in metadata. Not intended for
  normal OceanEmbed year-by-year preprocessing.
- Nothing else changed: regridding algorithm, `min_valid_weight=0.5`,
  `.001` unit handling, the 0.25 deg / 101x241 target grid, the
  5-30N/45-105E region, missing-value/mask behavior, extrapolation
  policy, and the `sss` output variable name are all untouched --
  confirmed by re-running the identical 2025 synthetic file through
  both this version and 0.2.0 and diffing the output arrays
  (`np.array_equal(..., equal_nan=True)` -> `True` for both `sss` and
  `sss_valid_mask`).

**0.2.0 -- generalized to be year-agnostic.**
- No more 2025-specific default input path, `start_date`/`end_date`, or
  `expected_time_step_count=365`.
- Year is **detected from the time coordinate** (dominant calendar year
  across the file's actual timestamps). The filename is used only as an
  independent cross-check -- a mismatch is logged as a warning, never
  silently resolved by trusting the filename.
- `require_complete_temporal_coverage` now defaults to **False**: a
  partial-year file is processed and clearly reported as partial, not
  rejected, unless `--require-complete-year` is passed.
- Leap years handled correctly via `calendar.isleap()` (365 vs 366
  expected days).
- Output paths are `{year}`-templated (`sss_<year>_0p25deg.nc`, etc.),
  computed after the year is detected.
- Variable detection now cross-checks **name candidates** against **CF
  metadata** (`standard_name`/`long_name`) independently, and raises on
  ambiguity instead of taking the first name match.
- Added descending-coordinate normalization and longitude-convention
  detection/normalization (both recorded in metadata as explicit
  transformations, never silent).
- Added the three explicit mask-consistency checks
  (`validate_mask_consistency`), run both right after regridding and
  again on the reopened output file.
- Added `unit_interpretation_status` / `unit_documentation_reference` /
  `unit_interpretation_notes` metadata fields (see "Unit handling"
  below) -- `assume_psu_equivalent` still defaults to `False` and, even
  when enabled, only changes documented metadata, never a numeric value.
- Added batch mode (`--input-dir`/`--output-root`): processes every
  `.nc` file found, continues past a failed file, and prints a
  pass/fail summary.
- Diagnostic plot now picks a timestep with actual data
  (`select_representative_timestep`, argmax of valid-cell count)
  instead of always using index 0, and is filename-safe across years.

**0.1.0** -- initial pipeline, 2025-only.

## Files

```
preprocessing/sss/          (rename from sss_trial1/sss_preprocess_trial1 as you prefer)
    config_sss.py
    netcdf_utils.py
    preprocess_sss.py
    README.md
```

## Commands

Single year:
```powershell
python .\preprocessing\sss\preprocess_sss.py `
  --input ".\data\raw\sss\test\2024\sss_2024.nc" `
  --output-dir ".\data\processed\sss\2024"
```
If `--output-dir` is omitted, it's derived automatically as
`data/processed/sss/<detected_year>/`.

Batch (all years under a directory):
```powershell
python .\preprocessing\sss\preprocess_sss.py `
  --input-dir ".\data\raw\sss\test" `
  --output-root ".\data\processed\sss"
```
Discovers every `.nc` file under `--input-dir` (recursively, via
`--input-glob`, default `**/*.nc`), processes each independently into
`<output-root>/<year>/`, continues past a failed file (reporting it
clearly), and prints a final summary. Existing outputs are protected --
pass `--overwrite` to allow replacing them, in either mode.

Dry run (either mode): add `--dry-run`.

## How the year is detected

`netcdf_utils.detect_year()` takes the **dominant calendar year** across
the file's actual `time` coordinate values (not the filename). The
filename (if it contains a 4-digit year) is cross-checked and any
mismatch is logged as a warning -- the data always wins. `--year <N>`
force-overrides the year used for output naming if you need it, also
with a mismatch warning if it disagrees with the data.

## Leap years / partial years

- Expected days-in-year is computed via `calendar.isleap(year)` (365 or
  366) -- both are handled correctly, not hard-coded to 365.
- A file's actual date coverage is always compared against its own
  detected year's calendar and reported as `complete`, `partial`, or
  `empty` in `year_detection.coverage_status`.
- **Default behavior: partial years are processed, not rejected** --
  only their actual available dates are used; no dates are fabricated
  or padded. Pass `--require-complete-year` to make an incomplete year
  a hard failure instead.

## Missing values / validity mask

Unchanged scientific behavior from 0.1.0: NaN-aware normalized weighted
bilinear interpolation (`scipy.interpolate.RegularGridInterpolator`),
extrapolation disabled (`bounds_error=False, fill_value=0.0` on both the
value- and weight-interpolators). `min_valid_weight` (default 0.5) is a
genuine **interpolation support-weight fraction** -- how much of a
target cell's 4-corner bilinear weighted average came from finite
source cells -- not a simple valid-neighbor-pixel count.

`sss_valid_mask`: `1` = valid + sufficient support; `0` = invalid.
Boundary cells invalidated because the native grid doesn't reach the
requested edge are reported as **unsupported by source coordinate
coverage**, not asserted to be land.

Three invariants are checked explicitly (`validate_mask_consistency`),
both right after regridding and again on the reopened output:
```python
if np.any((mask == 1) & ~np.isfinite(sss)):
    raise ValueError("Mask marks NaN values as valid.")
if np.any((mask == 0) & np.isfinite(sss)):
    raise ValueError("Mask marks finite values as invalid.")
if not np.all(np.isin(mask, [0, 1])):
    raise ValueError("Validity mask contains values other than 0 and 1.")
```

## Unit handling (`.001`)

Numerical values are **never** multiplied/divided or auto-relabeled.
Every output's metadata JSON and NetCDF variable attrs carry:
```json
{
  "source_units": ".001",
  "unit_conversion": "none",
  "interpreted_units": null,
  "unit_interpretation_status": "pending_verification",
  "unit_documentation_reference": null,
  "unit_interpretation_notes": "Source units '.001' preserved; interpretation requires confirmation from product documentation."
}
```
**No official source-product documentation for the `.001` units string
was located or consulted in this task** -- `unit_documentation_reference`
is `null` and must stay that way until someone actually finds and cites
it; nothing here should be read as "the `.001` unit has been verified."

`--assume-psu-equivalent` sets `interpreted_units: "PSU-equivalent"` and
`unit_interpretation_status: "user_confirmed"` **in metadata only** --
it never touches a numeric value, and defaults to off.

## Output variable naming

`sos` (source) is standardized to `sss` (output) by default
(`config.output_variable_name`). **No existing "tensor pipeline" was
available to inspect in this environment** to confirm a shared
downstream naming convention -- `sss` is the documented, configurable
default; `source_variable`/`output_variable` are both recorded
separately in every metadata JSON regardless. If a tensor-loading
convention exists elsewhere in the project, point
`output_variable_name` at it and this note can be removed.

## Testing performed

All on **synthetic** NetCDF files built to match the real file's
structure/attrs/encoding (no access to any real raw SSS file in this
environment):

1. **Compile check**: `python -m py_compile` on all three files --
   clean.
2. **2025 dry-run**: completed, dry-run JSON report written, no
   processed output.
3. **2025 full run**: completed; output shape `(5, 101, 241)` for the
   5-timestep synthetic file (the real file would be `(365, 101, 241)`
   or `(366, 101, 241)` for the actual year's day count); reopened from
   disk and validated (dims/shape/dtype/mask-consistency all passed);
   raw input file size/mtime confirmed unchanged.
4. **v0.1.0 vs v0.2.0 regression check**: ran both pipeline versions on
   the *identical* synthetic 2025 file and confirmed the regridded
   `sss` values and `sss_valid_mask` are **exactly numerically
   identical** (`np.array_equal(..., equal_nan=True)` → `True`) --
   the refactor did not change the regridding algorithm's output.
5. **Leap year**: synthetic 2024 file (366 daily timesteps,
   2024-01-01..2024-12-31) → detected `is_leap_year=True`,
   `coverage_status="complete"`, `366/366` days, output shape
   `(366, 101, 241)`.
6. **Partial year**: synthetic 2023 file (74 days, Jan 1 -- Mar 15
   only) → detected `coverage_status="partial"`, `74/365` days,
   processed successfully with output shape `(74, 101, 241)` (no
   fabricated padding); confirmed `--require-complete-year` correctly
   raises `ValueError` for this same file.
7. **CF-metadata-only variable detection + descending coordinates**:
   synthetic file with variable named `salt_conc` (not in any name
   candidate list, only matched via `standard_name=sea_surface_salinity`)
   and latitude stored descending → variable correctly detected via CF
   metadata alone, latitude correctly sorted ascending with the
   transformation logged.
8. **Malformed input**: a non-NetCDF file with a `.nc` extension →
   raised a clear `OSError` naming the file and the underlying cause,
   not a silent failure or crash.
9. **Batch mode**: ran `--input-dir` over all 5 synthetic files above
   (2021 malformed, 2022, 2023, 2024, 2025) → 4 succeeded with correct
   per-year output directories/filenames, 1 (malformed) failed with a
   clear per-file error, batch continued and printed a final
   4-succeeded/1-failed summary; confirmed this in both `--dry-run` and
   real-write modes.
10. **Mask-consistency check unit test**: called
    `validate_mask_consistency` directly with three deliberately broken
    mask arrays (NaN marked valid, finite marked invalid, non-binary
    values) and confirmed each raises the exact expected error message.

**Still not tested: the real 2025 (or any other year's) raw file.** I
do not have access to it. The reported "successful 2025 run" numbers in
your message (49.3563% pre-regrid / 48.8312% post-regrid valid) are
your own reported results from running the prior pipeline version
yourself -- I have not independently reproduced them and can't confirm
they match this refactor without actually running it against that real
file. Please run `--dry-run` against the real 2025 file first and
compare its `pre_regrid_missing_value_stats` / `post_regrid_missing_value_stats`
against what you reported; if they match, the refactor is very likely
behavior-preserving (as the identical-synthetic-file test above
suggests); if they don't, please share the log rather than assuming
either version is correct.

## Unresolved decisions (unchanged from 0.1.0, still need sign-off)

1. `min_valid_weight = 0.5` default.
2. `.001` unit interpretation -- no documentation found or consulted.
3. Edge-cell policy (NaN by design, no nearest-neighbor fill).
4. Whether `.npy` companions are actually needed downstream.
5. `output_variable_name = "sss"` -- not checked against a known tensor
   pipeline convention (none was available here).
