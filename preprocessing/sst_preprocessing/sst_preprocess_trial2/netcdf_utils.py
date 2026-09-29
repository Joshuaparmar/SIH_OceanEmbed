"""
netcdf_utils.py (SST)
======================
Reusable, file-agnostic helpers for inspecting and safely subsetting an
OSTIA/CMEMS-style SST NetCDF file. Nothing here assumes a specific
variable or coordinate name -- every name is detected against the real
file's actual dims/coords/data_vars and reported, never assumed.

Structural pattern (dataclass-driven detection, boolean-mask subsetting
instead of slice(), fatal-by-design regridding) is shared with
preprocessing/temp_preprocess_trial3/netcdf_utils.py for project-wide
consistency, but this module is independently importable and does not
import that one. SST has no depth dimension (2-D per timestep: time,
lat, lon), so depth-selection logic is absent here; unit handling
(Kelvin vs Celsius) and land/ocean mask awareness are SST-specific
additions not present in the temperature module.

Functions raise ValueError / FileNotFoundError / OSError with a
specific, actionable message on genuine problems. Things worth knowing
but not fatal (descending coordinate order, duplicate timestamps, an
unusual longitude convention, partial region/time overlap) are reported
in the returned dicts instead of raised.

Subsetting always uses boolean masks / integer positions derived from
the coordinate VALUES (np.where(mask) -> isel), never slice(min, max)
or label-based .sel() -- deliberate, since slice() silently misbehaves
on a descending coordinate and this module does not assume ascending
order anywhere.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("preprocess_sst")

KELVIN_OFFSET = 273.15
KELVIN_UNIT_LABELS = ("k", "kelvin", "degrees_k", "deg_k", "degree_kelvin")
CELSIUS_UNIT_LABELS = ("degc", "degrees_c", "celsius", "deg_c", "degree_celsius", "degrees celsius")


@dataclass
class DetectedCoords:
    """The actual coordinate names found in a real file (not assumed)."""
    time: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------

def find_input_files(config) -> Tuple[List[Path], str]:
    """
    Decide which real SST file(s) to use.

    Priority:
      1. config.explicit_input_path (--input): one exact path or
         directory, used as-is. A directory is globbed for *.nc.
      2. config.explicit_input_glob (--input-glob): glob match, ALL
         matches, sorted, opened together.
      3. config.sst_input_candidates: exact filenames, first that
         exists.
      4. config.sst_input_glob: glob match, ALL matches, sorted.

    Returns (paths, source_label). paths is [] if nothing was found --
    the caller must report that clearly rather than crash. A year of
    daily SST can legitimately be split across several files (e.g. one
    per month); glob matches keep ALL of them, not just the first.
    """
    if config.explicit_input_path:
        p = config.resolve(config.explicit_input_path)
        if p.is_dir():
            matches = sorted(p.glob("*.nc"))
            if matches:
                return matches, f"explicit directory (--input): {p} ({len(matches)} file(s))"
            logger.warning("Explicit --input directory contains no .nc files: %s", p)
            return [], f"explicit directory (--input, NO .nc FILES): {p}"
        if p.exists():
            return [p], f"explicit path (--input): {p}"
        logger.warning("Explicit --input path does not exist: %s", p)
        return [], f"explicit path (--input, NOT FOUND): {p}"

    if config.explicit_input_glob:
        matches = sorted(Path(config.project_root).resolve().glob(config.explicit_input_glob))
        if matches:
            return matches, f"explicit glob (--input-glob) '{config.explicit_input_glob}': {len(matches)} file(s)"
        logger.warning("Explicit --input-glob matched nothing: %s", config.explicit_input_glob)
        return [], f"explicit glob (--input-glob, NO MATCHES): {config.explicit_input_glob}"

    for candidate in config.sst_input_candidates:
        p = config.resolve(candidate)
        if p.exists():
            return [p], f"candidate: {candidate}"

    glob_matches = sorted(Path(config.project_root).resolve().glob(config.sst_input_glob))
    if glob_matches:
        return glob_matches, f"glob '{config.sst_input_glob}': {len(glob_matches)} file(s)"

    logger.warning(
        "No SST input file found. Tried sst_input_candidates=%s and glob '%s' "
        "(resolved against project_root=%s).",
        config.sst_input_candidates, config.sst_input_glob, config.project_root,
    )
    return [], "none found"


def check_input_output_collision(input_paths: List[Path], output_dir: Path) -> None:
    """
    Explicit raw-data-protection check, run BEFORE anything is opened or
    written. Raises ValueError (never silently proceeds, never silently
    renames/redirects) if the configured output directory could cause a
    raw input file to be overwritten or shadowed:

      1. output_dir IS an input file path itself (a clearly broken
         config, but checked explicitly rather than trusted).
      2. output_dir is the SAME directory an input file already lives
         in -- writing sst_target.nc etc. there is not a guaranteed
         name collision with the raw file, but it violates "write
         outputs only to a separate output directory" and meaningfully
         increases the risk of an accidental future overwrite/cleanup
         deleting raw data alongside processed data.
      3. An input file is located INSIDE output_dir -- almost certainly
         a misconfiguration (raw data should not live inside the
         processed-output tree).

    Does not modify or touch the input files in any way -- this is a
    pure path check.
    """
    out_resolved = Path(output_dir).resolve()
    for p in input_paths:
        p_resolved = Path(p).resolve()
        if p_resolved == out_resolved:
            raise ValueError(
                f"Configured output_dir ({out_resolved}) is EXACTLY the same path as input file "
                f"{p_resolved}. Refusing to proceed -- this would risk overwriting raw data. "
                f"Set a separate output_dir in config_sst.py or via --output-dir."
            )
        if p_resolved.parent == out_resolved:
            raise ValueError(
                f"Configured output_dir ({out_resolved}) is the SAME directory the input file "
                f"{p_resolved} already lives in. Raw input data and processed output must live in "
                f"separate directories (project requirement). Set a separate output_dir."
            )
        if out_resolved in p_resolved.parents:
            raise ValueError(
                f"Input file {p_resolved} is located INSIDE the configured output_dir "
                f"({out_resolved}). This looks like a misconfiguration -- raw data should not live "
                f"inside the processed-output directory tree. Fix input discovery or output_dir."
            )


def open_dataset(paths: List[Path], use_dask_chunking: bool = True) -> xr.Dataset:
    """
    Open one or several NetCDF files with xarray, distinguishing
    "doesn't exist" from "exists but isn't valid NetCDF yet (e.g. a
    partial download)" from other errors -- with a clear message for
    each, instead of a raw traceback.

    use_dask_chunking=True (default) passes chunks="auto" so the
    dataset is dask-backed and lazily/chunk-read rather than pulling
    the whole file into memory just to open and inspect it. Multi-file
    opens are always dask-backed regardless (open_mfdataset requires it).
    """
    if not paths:
        raise FileNotFoundError("No input file path(s) given to open_dataset().")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Input file(s) do not exist: {missing}. If a download is "
            f"still running, re-run this script once it finishes."
        )
    try:
        if len(paths) == 1:
            ds = xr.open_dataset(paths[0], chunks="auto" if use_dask_chunking else None)
        else:
            ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords")
    except Exception as exc:  # noqa: BLE001 - re-raised below with context
        raise OSError(
            f"Found {[str(p) for p in paths]} but could not open/combine as "
            f"NetCDF ({type(exc).__name__}: {exc}). If a download is still "
            f"in progress this can happen with a partially-written file -- "
            f"wait for it to finish and re-run."
        ) from exc
    return ds


# ---------------------------------------------------------------------
# Variable / coordinate name detection
# ---------------------------------------------------------------------

def detect_variable(ds: xr.Dataset, candidates: List[str]) -> str:
    """Return the first candidate present in ds.data_vars. Raises with
    the full actual variable list if none match -- never guesses."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    raise ValueError(
        f"None of the expected SST variable names {candidates} were "
        f"found. Actual variables in the file: {list(ds.data_vars)}. Add "
        f"the real name to sst_var_candidates in config_sst.py, or pass "
        f"it explicitly if this pipeline is extended with a --var flag."
    )


def detect_optional_variable(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    """Like detect_variable, but returns None instead of raising when no
    candidate is present -- for genuinely optional companions to the
    main SST variable (mask, analysis_error, sea_ice_fraction)."""
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


def detect_coordinate_name(ds: xr.Dataset, candidates: List[str], logical_name: str) -> str:
    """Return the first candidate present in ds.coords (falling back to
    ds.dims for a dimension without an indexed coordinate variable).
    Raises with the actual dims/coords if nothing matches."""
    for name in candidates:
        if name in ds.coords:
            return name
    for name in candidates:
        if name in ds.dims:
            return name
    raise ValueError(
        f"None of the expected '{logical_name}' coordinate names "
        f"{candidates} were found. Actual dims: {list(ds.dims)}, actual "
        f"coords: {list(ds.coords)}. Add the real name to config_sst.py."
    )


def detect_all_coords(ds: xr.Dataset, config) -> DetectedCoords:
    return DetectedCoords(
        time=detect_coordinate_name(ds, config.time_name_candidates, "time"),
        latitude=detect_coordinate_name(ds, config.latitude_name_candidates, "latitude"),
        longitude=detect_coordinate_name(ds, config.longitude_name_candidates, "longitude"),
    )


# ---------------------------------------------------------------------
# Coordinate validation (read-only -- never modifies ds)
# ---------------------------------------------------------------------

def validate_coordinates(ds: xr.Dataset, coords: DetectedCoords) -> Dict[str, object]:
    """
    Inspect (never modify) the three coordinate arrays and return a
    dict of findings for the validation report.

    Raises ValueError only for problems serious enough that continuing
    would be scientifically misleading (NaN inside a coordinate array
    itself). Duplicates, descending order, and an unusual longitude
    convention are reported, not raised -- subsetting/regridding below
    handle non-ascending coordinates explicitly rather than assuming
    order.
    """
    report: Dict[str, object] = {"warnings": []}

    time_vals = pd.to_datetime(ds[coords.time].values)
    lat_vals = np.asarray(ds[coords.latitude].values, dtype=float)
    lon_vals = np.asarray(ds[coords.longitude].values, dtype=float)

    for label, vals in (("latitude", lat_vals), ("longitude", lon_vals)):
        if np.isnan(vals).any():
            raise ValueError(f"'{label}' coordinate contains NaN value(s); cannot proceed safely.")
    if pd.isna(time_vals).any():
        raise ValueError("'time' coordinate contains NaT/NaN value(s); cannot proceed safely.")

    def _dup_info(vals: np.ndarray) -> Tuple[bool, List[float]]:
        uniq, counts = np.unique(vals, return_counts=True)
        dups = uniq[counts > 1]
        return bool(len(dups) > 0), dups[:5].tolist()

    for label, vals in (("latitude", lat_vals), ("longitude", lon_vals)):
        has_dup, examples = _dup_info(vals)
        report[f"{label}_has_duplicates"] = has_dup
        if has_dup:
            report["warnings"].append(f"'{label}' coordinate has duplicate values (examples: {examples}).")
    has_dup_time, _ = _dup_info(time_vals.values.astype("int64"))
    report["time_has_duplicate_timestamps"] = has_dup_time
    if has_dup_time:
        report["warnings"].append("Time coordinate has duplicate timestamps.")

    report["time_ascending"] = bool(len(time_vals) < 2 or np.all(np.diff(time_vals.values.astype("int64")) > 0))
    report["latitude_ascending"] = bool(len(lat_vals) < 2 or lat_vals[0] < lat_vals[-1])
    report["longitude_ascending"] = bool(len(lon_vals) < 2 or lon_vals[0] < lon_vals[-1])
    if not report["latitude_ascending"]:
        report["warnings"].append("'latitude' coordinate is not ascending; reported, not reordered here.")

    report["latitude_range"] = (float(np.min(lat_vals)), float(np.max(lat_vals)))
    report["longitude_range"] = (float(np.min(lon_vals)), float(np.max(lon_vals)))
    report["longitude_convention"] = (
        "-180 to 180" if np.min(lon_vals) < 0 else "0 to 360 (or a positive-only subset)"
    )
    if not (-90.0 - 1e-6 <= np.min(lat_vals) and np.max(lat_vals) <= 90.0 + 1e-6):
        report["warnings"].append(
            f"Latitude values outside [-90, 90]: min={np.min(lat_vals)}, max={np.max(lat_vals)}."
        )

    report["n_time"] = int(len(time_vals))
    report["n_latitude"] = int(len(lat_vals))
    report["n_longitude"] = int(len(lon_vals))
    report["first_timestamp"] = str(time_vals[0]) if len(time_vals) else None
    report["last_timestamp"] = str(time_vals[-1]) if len(time_vals) else None

    if len(time_vals) > 1:
        sorted_times = np.sort(time_vals.values)
        diffs_days = np.diff(sorted_times).astype("timedelta64[h]").astype(float) / 24.0
        report["median_time_step_days"] = float(np.median(diffs_days))
        report["is_daily"] = bool(np.isclose(np.median(diffs_days), 1.0, atol=0.05))
        # Normalize both sides to midnight before diffing calendar dates
        # -- some L4 products stamp daily timestamps at a fixed hour
        # (e.g. 12:00 UTC or 00:00 UTC); comparing un-normalized dates
        # against a midnight-normalized expected range would falsely
        # report every date as "missing". This also naturally handles
        # leap years correctly (pd.date_range enumerates real calendar
        # days, so Feb 29 in a leap year is expected, not assumed 365
        # days every year).
        actual_days = pd.DatetimeIndex(sorted_times).normalize()
        expected_days = pd.date_range(actual_days.min(), actual_days.max(), freq="D")
        missing_dates = sorted(set(expected_days) - set(actual_days))
        report["missing_dates"] = [str(d.date()) for d in missing_dates]
        if missing_dates:
            report["warnings"].append(f"{len(missing_dates)} missing date(s) within the time span.")
        if not report["is_daily"]:
            report["warnings"].append(
                f"Median time step is {report['median_time_step_days']:.2f} days, not ~1 day."
            )
    else:
        report["median_time_step_days"] = None
        report["is_daily"] = None
        report["missing_dates"] = []

    if len(lat_vals) > 1:
        lat_diffs = np.diff(np.sort(lat_vals))
        report["latitude_resolution_deg"] = float(np.median(np.abs(lat_diffs)))
        report["latitude_grid_regular"] = bool(np.allclose(lat_diffs, lat_diffs[0], atol=1e-6))
    else:
        report["latitude_resolution_deg"] = None
        report["latitude_grid_regular"] = None
    if len(lon_vals) > 1:
        lon_diffs = np.diff(np.sort(lon_vals))
        report["longitude_resolution_deg"] = float(np.median(np.abs(lon_diffs)))
        report["longitude_grid_regular"] = bool(np.allclose(lon_diffs, lon_diffs[0], atol=1e-6))
    else:
        report["longitude_resolution_deg"] = None
        report["longitude_grid_regular"] = None

    return report


# ---------------------------------------------------------------------
# Region / time subsetting
# ---------------------------------------------------------------------

def subset_region_time(ds: xr.Dataset, coords: DetectedCoords, config) -> Tuple[xr.Dataset, Dict[str, object]]:
    """
    Select the configured lat/lon/time window using boolean masks over
    coordinate VALUES (not slice(), which silently misbehaves on a
    descending coordinate). Reports the actual overlap achieved and
    raises only if the overlap is completely empty.

    Longitude convention: if the file uses 0-360 and the requested
    lon_min/lon_max (45-105 for the OceanEmbed region) are given in
    that same positive convention, the mask works directly. If the
    file instead uses -180..180 while the requested bounds are still
    positive-only in the same numeric range (as here, 45-105, which is
    representable identically in both conventions), no conversion is
    needed either. A genuine convention mismatch (e.g. a request
    spanning the antimeridian) is out of scope for the fixed
    45-105E/5-30N OceanEmbed region and is reported via
    longitude_convention in validate_coordinates() rather than silently
    guessed at here.
    """
    lat_vals = ds[coords.latitude].values
    lon_vals = ds[coords.longitude].values
    time_vals = pd.to_datetime(ds[coords.time].values)

    lat_mask = (lat_vals >= config.lat_min) & (lat_vals <= config.lat_max)
    lon_mask = (lon_vals >= config.lon_min) & (lon_vals <= config.lon_max)
    # END-DATE HANDLING IS EXPLICITLY END-INCLUSIVE, not "<= midnight of
    # end_date". `time <= pd.Timestamp(end_date)` would silently exclude
    # any timestamp stamped later in the day on end_date itself (e.g. an
    # L4 product stamped at 12:00 UTC, or any non-midnight convention) --
    # a real, easy-to-miss data-loss bug. Instead: build an END-EXCLUSIVE
    # upper bound one calendar day past end_date, so every timestamp
    # whose calendar date is end_date is kept regardless of time-of-day,
    # and nothing from the following day leaks in either.
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)  # kept for reporting/coverage checks below
    end_exclusive = end + pd.Timedelta(days=1)
    time_mask = (time_vals >= start) & (time_vals < end_exclusive)

    report: Dict[str, object] = {
        "warnings": [],
        "requested_lat_range": (config.lat_min, config.lat_max),
        "requested_lon_range": (config.lon_min, config.lon_max),
        "requested_time_range": (config.start_date, config.end_date),
        "original_lat_range": (float(lat_vals.min()), float(lat_vals.max())),
        "original_lon_range": (float(lon_vals.min()), float(lon_vals.max())),
        "n_lat_matched": int(lat_mask.sum()),
        "n_lon_matched": int(lon_mask.sum()),
        "n_time_matched": int(time_mask.sum()),
    }

    if lat_mask.sum() == 0 or lon_mask.sum() == 0 or time_mask.sum() == 0:
        raise ValueError(
            f"Requested region/time has ZERO overlap with the file. "
            f"lat matched={lat_mask.sum()}, lon matched={lon_mask.sum()}, "
            f"time matched={time_mask.sum()}. File covers "
            f"lat[{lat_vals.min():.3f},{lat_vals.max():.3f}] "
            f"lon[{lon_vals.min():.3f},{lon_vals.max():.3f}] "
            f"time[{time_vals.min()},{time_vals.max()}]. If the file uses "
            f"a 0-360 longitude convention and the requested bounds don't "
            f"match, adjust lon_min/lon_max in config_sst.py."
        )

    actual_lat = lat_vals[lat_mask]
    actual_lon = lon_vals[lon_mask]
    actual_time = time_vals[time_mask]
    full_lat = bool(actual_lat.min() <= config.lat_min + 1e-6 and actual_lat.max() >= config.lat_max - 1e-6)
    full_lon = bool(actual_lon.min() <= config.lon_min + 1e-6 and actual_lon.max() >= config.lon_max - 1e-6)
    # Time coverage is checked by CALENDAR DATE on BOTH ends, not raw
    # timestamp comparison against `start`/`end` (which are midnight) --
    # comparing raw timestamps would falsely report incomplete coverage
    # whenever the file's daily timestamps are stamped at a non-midnight
    # time (e.g. noon UTC, common for L4 products): the first selected
    # timestamp would be "later than" a midnight `start` even though it
    # is the correct first day, and symmetrically for the last day
    # against `end` (see the end_exclusive fix above -- this is the
    # matching fix for the coverage-COMPLETENESS check, not just the
    # selection mask).
    full_time = bool(
        pd.Timestamp(actual_time.min()).normalize() <= start.normalize()
        and pd.Timestamp(actual_time.max()).normalize() >= end.normalize()
    )

    report["actual_lat_range"] = (float(actual_lat.min()), float(actual_lat.max()))
    report["actual_lon_range"] = (float(actual_lon.min()), float(actual_lon.max()))
    report["actual_time_range"] = (str(actual_time.min()), str(actual_time.max()))
    report["resulting_dimensions"] = {
        coords.time: int(time_mask.sum()), coords.latitude: int(lat_mask.sum()), coords.longitude: int(lon_mask.sum()),
    }
    # Separate flags (not just the combined AND) so callers -- notably
    # the official-spatial-coverage strict check -- can require full
    # SPATIAL coverage without conflating it with temporal coverage,
    # which is governed independently by require_complete_temporal_coverage.
    report["full_lat_coverage"] = full_lat
    report["full_lon_coverage"] = full_lon
    report["full_time_coverage"] = full_time
    report["full_region_time_coverage"] = bool(full_lat and full_lon and full_time)

    if not report["full_region_time_coverage"]:
        msg = (
            f"Requested region/time only PARTIALLY covered. Actual lat "
            f"[{actual_lat.min():.3f}, {actual_lat.max():.3f}] vs requested "
            f"[{config.lat_min}, {config.lat_max}]; actual lon "
            f"[{actual_lon.min():.3f}, {actual_lon.max():.3f}] vs requested "
            f"[{config.lon_min}, {config.lon_max}]; actual time "
            f"[{actual_time.min()}, {actual_time.max()}] vs requested "
            f"[{start}, {end}]."
        )
        logger.warning(msg)
        report["warnings"].append(msg)
        if not config.allow_partial_spatial_overlap:
            raise ValueError("Partial overlap and allow_partial_spatial_overlap=False. " + msg)

    subset = ds.isel({
        coords.latitude: np.where(lat_mask)[0],
        coords.longitude: np.where(lon_mask)[0],
        coords.time: np.where(time_mask)[0],
    })
    return subset, report


# ---------------------------------------------------------------------
# Missing values / units / plausibility (report-only, never mutates
# the input in place -- unit conversion returns a NEW DataArray)
# ---------------------------------------------------------------------

def _compute_scalar(x):
    """Force a lazy (dask-backed) xarray reduction result to a plain
    Python scalar. A no-op (just .item()) for an already-eager/numpy-
    backed result. Centralizing this is what lets every diagnostic
    below be written once and work correctly whether the DataArray is
    dask-chunked or not -- the caller never has to know."""
    if hasattr(x, "compute"):
        x = x.compute()
    return x.item() if hasattr(x, "item") else x


def compute_valid_mask(da: "xr.DataArray") -> np.ndarray:
    """
    True where finite (a real, usable ocean value); False for NaN/Inf
    (land or otherwise invalid/missing). xarray decodes NetCDF
    _FillValue / missing_value into NaN by default (mask_and_scale=True),
    so this catches those too.

    NOTE ON SEMANTICS: this is a single finite-value mask. It does NOT
    distinguish land, missing ocean observations, sea-ice/invalid
    regions, or (after regridding) target cells that fell outside the
    source's coverage from one another -- all of those collapse to
    False here. It is also not a verified, authoritative land-sea mask:
    it is only "wherever the SST variable itself is non-finite." If the
    source file ships its own categorical land/sea mask variable, that
    is a materially different (and more precise) piece of information;
    this pipeline detects such a variable if present (see
    mask_var_candidates) but does not fabricate one from SST NaNs and
    does not currently fold it into this mask -- see README "Land/ocean
    handling and mask semantics."

    Accepts an xr.DataArray (materializes it here, once, at the point
    the actual boolean array is needed for writing -- this is the right
    place to pay that cost, not several times earlier for diagnostics
    that only need scalar counts; see inspect_missing_values /
    flag_implausible_values for the lazy versions of those)."""
    values = da.values if hasattr(da, "values") else da
    return np.isfinite(values)


def inspect_missing_values(da: "xr.DataArray") -> Dict[str, object]:
    """
    Missing/invalid-cell counts computed via xarray reductions
    (da.isnull().sum() / da.count()), which stay dask-lazy until
    .compute() is called here on the scalar RESULT only -- this never
    materializes the full array into memory, unlike calling
    np.isfinite(da.values). This matters: a subsetted-but-not-yet-
    regridded SST array at native ~0.05 deg resolution over the full
    5-30N/45-105E region for a full year can be several hundred million
    cells, too large to comfortably hold as one in-memory numpy array
    just to count NaNs.

    Accepts either an xr.DataArray or a plain numpy array (for the
    handful of callers -- e.g. a small already-materialized regridded
    batch -- that already have one; the numpy path skips the lazy
    machinery entirely rather than needlessly wrapping it)."""
    if isinstance(da, np.ndarray):
        finite = np.isfinite(da)
        total = int(da.size)
        n_valid = int(finite.sum())
    else:
        total = int(da.size)
        n_valid = _compute_scalar(da.count())
    n_invalid = total - n_valid
    return {
        "total_cells": total,
        "nan_or_inf_count": n_invalid,
        "valid_count": n_valid,
        "fraction_invalid": (n_invalid / total) if total else None,
    }


def flag_implausible_values(da: "xr.DataArray", low: float, high: float) -> Dict[str, object]:
    """
    Soft check on FINITE values only: sea-surface temperature
    realistically falls within roughly [-3, 45] deg C (sea water
    freezes near -1.8 degC; the hottest open-ocean SSTs observed are
    in the mid-30s). Values outside this range among already-finite
    cells usually mean a _FillValue/missing_value sentinel slipped
    through unmasked, not real physics, or a units mismatch (e.g.
    Kelvin values not yet converted). Never modifies data -- report only.

    Uses xarray/dask-lazy reductions (min/max/mean/std, and a boolean
    count for the implausible fraction) rather than pulling the whole
    array into memory as numpy -- see inspect_missing_values docstring
    for why this matters for a native-resolution subsetted array.
    xarray's min/max/mean/std already skip NaN by default (equivalent
    to nanmin/nanmax/nanmean/nanstd), so no separate finite-masking
    step is needed for those. The implausible COUNT still has to
    exclude NaN explicitly (NaN is not < low or > high, so a plain
    boolean comparison already naturally excludes it -- but this is
    stated explicitly here rather than left implicit).

    `median_finite` is the one exception: dask does not support a fully
    lazy exact median, so this uses a boolean-indexed materialization
    of ONLY the (typically much smaller) already-regridded array when
    called post-regrid, and is best-effort (returns None rather than
    forcing a large materialization) when called on a still-large,
    dask-backed pre-regrid array.
    """
    is_dask = hasattr(da, "data") and hasattr(da.data, "chunks") and da.data.chunks is not None
    if isinstance(da, np.ndarray):
        finite = da[np.isfinite(da)]
        if finite.size == 0:
            return {"n_implausible": 0, "min_finite": None, "max_finite": None, "mean_finite": None, "std_finite": None, "median_finite": None}
        implausible = finite[(finite < low) | (finite > high)]
        return {
            "n_implausible": int(implausible.size),
            "min_finite": float(finite.min()),
            "max_finite": float(finite.max()),
            "mean_finite": float(finite.mean()),
            "std_finite": float(finite.std()),
            "median_finite": float(np.median(finite)),
        }

    n_valid = _compute_scalar(da.count())
    if n_valid == 0:
        return {"n_implausible": 0, "min_finite": None, "max_finite": None, "mean_finite": None, "std_finite": None, "median_finite": None}
    n_implausible = _compute_scalar(((da < low) | (da > high)).sum())
    result = {
        "n_implausible": int(n_implausible),
        "min_finite": float(_compute_scalar(da.min())),
        "max_finite": float(_compute_scalar(da.max())),
        "mean_finite": float(_compute_scalar(da.mean())),
        "std_finite": float(_compute_scalar(da.std())),
        "median_finite": None,
    }
    if not is_dask:
        result["median_finite"] = float(np.nanmedian(da.values))
    else:
        result["median_finite_note"] = (
            "skipped for a still-large dask-backed array to avoid a large materialization just "
            "for a diagnostic median; available on the final (post-regrid, already-small) array."
        )
    return result


def check_units(da: xr.DataArray) -> Tuple[Optional[str], List[str]]:
    """Report the variable's declared units without assuming them.
    Returns (units_string_or_None, warnings)."""
    warnings: List[str] = []
    units = da.attrs.get("units")
    if units is None:
        warnings.append(
            "SST variable has no 'units' attribute in the file. OSTIA "
            "analysed_sst is conventionally Kelvin, but this was NOT "
            "verified here -- check source documentation/metadata before "
            "assuming a unit for conversion."
        )
    elif str(units).strip().lower() not in KELVIN_UNIT_LABELS and str(units).strip().lower() not in CELSIUS_UNIT_LABELS:
        warnings.append(
            f"SST variable units attribute is {units!r}, which is neither "
            f"a recognized Kelvin nor Celsius label -- verify before "
            f"assuming a unit."
        )
    return units, warnings


def convert_units_if_needed(da: xr.DataArray, units: Optional[str], config) -> Tuple[xr.DataArray, Dict[str, object]]:
    """
    Explicit, never-silent unit handling. Returns (possibly-converted
    DataArray, report). The report always states what WAS or WAS NOT
    done and why, so a run's metadata is self-explanatory regardless of
    outcome.

    - unit_conversion_policy == "never": no conversion, regardless of
      the units attribute. Output units attr is left exactly as found
      (or unset).
    - "auto": convert Kelvin -> Celsius ONLY if the units attribute is
      recognized as Kelvin (case-insensitive match against
      KELVIN_UNIT_LABELS); retain as-is if recognized as Celsius. If
      units are missing/unrecognized, this RAISES ValueError by default
      (config.fail_on_unrecognized_units=True, the official default) --
      an unrecognized or missing units attribute must never be treated
      as "probably Celsius" (or "probably Kelvin") by guessing; writing
      SST under an unconfirmed unit label would be exactly the
      physically-ambiguous output the project brief prohibits. Set
      config.fail_on_unrecognized_units=False to downgrade this to a
      warning-only skip instead (not recommended for an official run).
    - "force_k_to_c": always subtract 273.15, regardless of the units
      attribute -- an EXPLICIT EXPERT OVERRIDE for a known-Kelvin file
      whose units attribute is missing/wrong (verified independently
      from product documentation, not guessed). Use deliberately, not
      by default; always recorded in the output metadata.
    """
    report: Dict[str, object] = {
        "declared_units": units,
        "policy": config.unit_conversion_policy,
        "converted": False,
        "output_units": units,
    }

    units_lower = str(units).strip().lower() if units is not None else None
    is_kelvin_labeled = units_lower in KELVIN_UNIT_LABELS
    is_celsius_labeled = units_lower in CELSIUS_UNIT_LABELS

    if config.unit_conversion_policy == "never":
        report["note"] = "unit_conversion_policy='never': values written exactly as stored in the source file."
        return da, report

    if config.unit_conversion_policy == "force_k_to_c":
        converted = da - KELVIN_OFFSET
        report.update(converted=True, output_units="degrees_C",
                       note=f"EXPERT OVERRIDE: forced Kelvin -> Celsius conversion regardless of the "
                            f"declared units attribute ({units!r}) (unit_conversion_policy='force_k_to_c').")
        logger.warning("unit_conversion_policy='force_k_to_c': forcing Kelvin->Celsius conversion "
                        "regardless of declared units=%r. This is an expert override -- verify it is "
                        "actually correct for this file.", units)
        return converted, report

    # "auto"
    if is_kelvin_labeled:
        converted = da - KELVIN_OFFSET
        report.update(converted=True, output_units="degrees_C",
                       note=f"Detected units={units!r} (Kelvin) -> converted to Celsius (x - 273.15).")
        return converted, report
    if is_celsius_labeled:
        report["note"] = f"Detected units={units!r} (Celsius) -> retained as Celsius, no conversion needed."
        return da, report

    # Missing/unrecognized units under "auto": fatal by default.
    if config.fail_on_unrecognized_units:
        raise ValueError(
            f"SST variable's units attribute is missing or unrecognized (units={units!r}). "
            f"unit_conversion_policy='auto' requires a units attribute recognized as either Celsius "
            f"or Kelvin to proceed with official processing -- writing SST under an unconfirmed unit "
            f"would be physically ambiguous output. Options: (1) inspect the source file's actual "
            f"metadata/documentation and, if it confirms the data is Kelvin despite this attribute "
            f"being missing/wrong, set unit_conversion_policy='force_k_to_c' explicitly (recorded as "
            f"an expert override in the metadata); (2) if the units are genuinely fine as stored and "
            f"conversion should simply be skipped, set unit_conversion_policy='never'; (3) set "
            f"fail_on_unrecognized_units=False to downgrade this to a warning-only skip (NOT "
            f"recommended for an official run)."
        )
    report["note"] = (
        f"units={units!r} is missing/unrecognized -- NOT assumed to be Kelvin or Celsius; no "
        f"conversion applied (fail_on_unrecognized_units=False, so this was a warning, not a fatal "
        f"error). Inspect the source file's metadata and set unit_conversion_policy='force_k_to_c' "
        f"or 'never' explicitly once the actual units are confirmed."
    )
    return da, report


# ---------------------------------------------------------------------
# Regridding -- OFFICIAL, default-on. Any failure is FATAL by design;
# there is deliberately no try/except-and-continue anywhere around this.
# ---------------------------------------------------------------------

def build_target_grid(config) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Explicitly construct the official target lat/lon grid from
    config's region bounds and target_resolution_deg -- NEVER derived
    from the source data's own (possibly slightly different)
    subsetted extent.

    Convention: cell-center, BOTH endpoints included.
    n_points = round((max - min) / spacing) + 1, then
    np.linspace(min, max, n_points) -- exact endpoints with no
    floating-point drift. For the default 5-30N / 45-105E bounds at
    0.25 deg this gives exactly 101 latitude x 241 longitude points.
    """
    n_lat = round((config.lat_max - config.lat_min) / config.target_resolution_deg) + 1
    n_lon = round((config.lon_max - config.lon_min) / config.target_resolution_deg) + 1
    new_lat = np.linspace(config.lat_min, config.lat_max, n_lat)
    new_lon = np.linspace(config.lon_min, config.lon_max, n_lon)
    grid_info: Dict[str, object] = {
        "convention": "cell-center, both endpoints included",
        "lat_min": float(new_lat[0]), "lat_max": float(new_lat[-1]), "n_lat": int(n_lat),
        "lat_spacing_deg": float(np.median(np.diff(new_lat))) if n_lat > 1 else None,
        "lon_min": float(new_lon[0]), "lon_max": float(new_lon[-1]), "n_lon": int(n_lon),
        "lon_spacing_deg": float(np.median(np.diff(new_lon))) if n_lon > 1 else None,
    }
    return new_lat, new_lon, grid_info


def regrid_to_target_grid(
    da: xr.DataArray, coords: DetectedCoords, config
) -> Tuple[xr.DataArray, Dict[str, object]]:
    """
    Regrid da's latitude/longitude onto the EXPLICIT official target
    grid (build_target_grid) via xarray's .interp() (scipy-backed).

    FATAL BY DESIGN: this is the official, default-on regridding path.
    Any failure -- missing scipy, non-monotonic/duplicate/non-finite
    source coordinates, insufficient coverage, an interpolation error,
    memory error, or anything else -- is raised as
    RuntimeError/ValueError and MUST propagate; callers must not
    catch-and-continue with the native grid. Call with
    regrid_to_quarter_degree=False (--no-regrid) for an explicit,
    clearly-labeled non-official diagnostic run instead.

    - Source lat/lon are validated immediately before interpolating:
      1-D, numeric, finite, no duplicates -- raised clearly if not.
    - A descending source coordinate is sorted ascending first via
      .sortby() -- an explicit, RECORDED operation, never silent.
    - No extrapolation: kwargs=dict(bounds_error=False, fill_value=NaN)
      passed to the interpolator. Target cells outside the source
      data's actual coverage become NaN, never a guessed/extrapolated
      value. This also means NaN land pixels in the source correctly
      propagate as NaN in the regridded output rather than being
      smeared into a fabricated ocean value.
    - If literally every target cell ends up NaN after a technically
      "successful" interpolation call, that's still treated as fatal
      rather than silently shipping an empty official output.
    """
    report: Dict[str, object] = {"warnings": [], "coordinates_sorted_ascending": []}

    lat_vals = da[coords.latitude].values
    lon_vals = da[coords.longitude].values
    for label, vals in ((coords.latitude, lat_vals), (coords.longitude, lon_vals)):
        if not np.issubdtype(vals.dtype, np.number):
            raise ValueError(f"Regridding requires a numeric '{label}' coordinate; got dtype {vals.dtype}.")
        if vals.ndim != 1:
            raise ValueError(f"Regridding requires a 1-D '{label}' coordinate; got ndim={vals.ndim}.")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Regridding requires finite '{label}' coordinate values; found non-finite value(s).")
        if len(np.unique(vals)) != len(vals):
            raise ValueError(f"Regridding requires unique '{label}' coordinate values; duplicates found.")

    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        da = da.sortby(coords.latitude)
        report["coordinates_sorted_ascending"].append(coords.latitude)
    if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
        da = da.sortby(coords.longitude)
        report["coordinates_sorted_ascending"].append(coords.longitude)
    if report["coordinates_sorted_ascending"]:
        report["warnings"].append(
            f"Sorted {report['coordinates_sorted_ascending']} ascending before regridding (recorded, not silent)."
        )

    src_lat = da[coords.latitude].values
    src_lon = da[coords.longitude].values
    report["source_latitude_resolution_deg"] = float(np.median(np.diff(src_lat))) if len(src_lat) > 1 else None
    report["source_longitude_resolution_deg"] = float(np.median(np.diff(src_lon))) if len(src_lon) > 1 else None

    new_lat, new_lon, grid_info = build_target_grid(config)
    report["target_grid"] = grid_info
    report["method"] = config.regrid_method
    report["extrapolation_policy"] = (
        "disabled (bounds_error=False, fill_value=NaN passed to the interpolator) -- "
        "out-of-coverage target cells are NaN, never extrapolated"
    )

    def _interp_one(block: xr.DataArray) -> xr.DataArray:
        result = block.interp(
            {coords.latitude: new_lat, coords.longitude: new_lon},
            method=config.regrid_method,
            kwargs={"bounds_error": False, "fill_value": np.nan},
        )
        return result.compute() if hasattr(result.data, "compute") else result

    # --- Batched over the time dimension if configured, so peak memory
    # is bounded by ONE BATCH's worth of (native-resolution) source data
    # plus its regridded result, not the entire subsetted array at once.
    # This matters most for a full year at OSTIA's native ~0.05 deg
    # resolution over the OceanEmbed region -- interpolating the whole
    # thing in a single .interp() call requires holding the full
    # (potentially ~1 GB+) source array and its output simultaneously.
    # Each batch is independently fatal-on-failure -- a failure on any
    # batch aborts the whole regrid (no partial-success silent
    # continuation), consistent with "any regridding failure is fatal."
    batch_size = getattr(config, "regrid_time_batch_size", None)
    n_time = int(da.sizes[coords.time])
    try:
        if not batch_size or batch_size >= n_time:
            regridded = _interp_one(da)
            report["batching"] = {"batch_size": None, "n_batches": 1, "note": "no batching (regrid_time_batch_size unset or >= n_time)"}
        else:
            n_batches = -(-n_time // batch_size)  # ceil division
            batches = []
            for i, start_idx in enumerate(range(0, n_time, batch_size)):
                end_idx = min(start_idx + batch_size, n_time)
                block = da.isel({coords.time: slice(start_idx, end_idx)})
                regridded_block = _interp_one(block)
                batches.append(regridded_block)
                logger.info("Regridded time batch %d/%d (steps %d:%d of %d)", i + 1, n_batches, start_idx, end_idx, n_time)
            regridded = xr.concat(batches, dim=coords.time)
            report["batching"] = {"batch_size": batch_size, "n_batches": n_batches}
    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY regridding failure is fatal, re-raised with context
        raise RuntimeError(
            f"Regridding to the official {config.target_resolution_deg} deg target grid FAILED "
            f"({type(exc).__name__}: {exc}). This is FATAL BY DESIGN -- the official pipeline "
            f"must never silently fall back to the native grid. Common causes: missing scipy "
            f"(pip install scipy), non-monotonic/duplicate source coordinates, or insufficient "
            f"source coverage for the requested region, or a genuine out-of-memory condition on a "
            f"large unbatched regrid (set config.regrid_time_batch_size, e.g. 30-60, to bound peak "
            f"memory). Fix the underlying cause and re-run, or pass --no-regrid for an explicit, "
            f"clearly-marked non-official native-grid DIAGNOSTIC run instead of the official one."
        ) from exc

    n_total = int(regridded.size)
    n_nan = int(np.isnan(regridded.values).sum())
    report["target_cells_total"] = n_total
    report["target_cells_invalid_after_regrid"] = n_nan
    report["target_cells_invalid_fraction"] = (n_nan / n_total) if n_total else None
    if n_total and n_nan == n_total:
        raise RuntimeError(
            "Regridding produced an ENTIRELY NaN output -- every target-grid cell fell outside "
            "the source data's actual coverage. This almost always means the requested region "
            f"(lat[{config.lat_min},{config.lat_max}] lon[{config.lon_min},{config.lon_max}]) does "
            "not overlap the source file's real coverage at all. Fatal by design, not silently shipped."
        )
    if n_nan > 0:
        report["warnings"].append(
            f"{n_nan}/{n_total} ({100.0 * n_nan / n_total:.2f}%) target-grid cells are NaN after "
            f"regridding (land, outside source coverage, or already-invalid source cells)."
        )

    return regridded, report


# ---------------------------------------------------------------------
# Output coordinate standardization (common-grid compatibility)
# ---------------------------------------------------------------------

def standardize_output_coordinates(
    da: xr.DataArray, coords: DetectedCoords, config
) -> Tuple[xr.DataArray, "DetectedCoords", Dict[str, str]]:
    """
    Rename the DETECTED source coordinate names to the standardized
    output names in config (output_time_name / output_latitude_name /
    output_longitude_name). Pure relabeling -- xarray's .rename() does
    not touch values -- and never modifies the source file, only the
    in-memory array about to be written to the NEW output file.
    """
    rename_map: Dict[str, str] = {}
    target_names = {
        coords.time: config.output_time_name,
        coords.latitude: config.output_latitude_name,
        coords.longitude: config.output_longitude_name,
    }
    for source_name, output_name in target_names.items():
        if source_name != output_name:
            rename_map[source_name] = output_name

    renamed = da.rename(rename_map) if rename_map else da
    new_coords = DetectedCoords(
        time=config.output_time_name,
        latitude=config.output_latitude_name,
        longitude=config.output_longitude_name,
    )
    return renamed, new_coords, rename_map


# ---------------------------------------------------------------------
# Atomic writes: a crash mid-write must never leave a corrupt or
# misleadingly-named file at the final output path.
# ---------------------------------------------------------------------

def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """
    Call write_fn(tmp_path) to write to a temp file in the SAME
    directory as `path`, then atomically replace `path` with it
    (os.replace -- atomic on both POSIX and Windows when source/dest
    share a filesystem, which a same-directory temp file guarantees).
    An exception during write_fn leaves the previous `path` untouched
    (or simply absent, on a first write) instead of a half-written
    file at the final name.

    tmp_path KEEPS path's suffix at the very end (e.g.
    ".sst_target.tmp1234.npy", not "...npy.tmp1234") on purpose:
    np.save() appends ".npy" itself to any filename that doesn't
    already end with it, which would otherwise silently write to a
    DIFFERENT path than the one this function replaces from.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        logger.warning("Overwriting existing output file: %s", path)
    tmp_path = path.with_name(f".{path.stem}.tmp{os.getpid()}{path.suffix}")
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
