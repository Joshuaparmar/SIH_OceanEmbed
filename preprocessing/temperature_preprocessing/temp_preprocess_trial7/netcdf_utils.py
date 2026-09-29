"""
netcdf_utils.py
================
Reusable, file-agnostic helpers for inspecting and safely subsetting a
GLORYS-style NetCDF temperature file. Nothing here assumes a specific
variable or coordinate name -- every name is detected against the real
file's actual dims/coords/data_vars and reported, never assumed.

Functions raise ValueError / FileNotFoundError / OSError with a
specific, actionable message on genuine problems, in the same spirit as
oceanembed_step5/validation.py: fail loudly and specifically rather
than silently reshaping, renaming, or coercing data. Things that are
worth knowing but not fatal (descending coordinate order, duplicate
timestamps, an unusual longitude convention, partial region/time
overlap) are reported in the returned dicts instead of raised, so the
caller can put them in the validation report.

Subsetting always uses boolean masks / integer positions derived from
the coordinate VALUES (np.where(mask) -> isel), never slice(min, max)
or label-based .sel(). This is deliberate: slice() silently returns
the wrong thing (often empty) if a coordinate turns out to be
descending, and this module does not assume ascending order anywhere.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("preprocess_temperature")


@dataclass
class DetectedCoords:
    """The actual coordinate names found in a real file (not assumed)."""
    time: str
    depth: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------
# Input discovery -- ACTIVE dataset is 2025; Jan-2026 prototype is
# discontinued and is deliberately NOT searched by find_input_files.
# ---------------------------------------------------------------------

def find_input_files(config) -> Tuple[List[Path], str]:
    """
    Decide which real file(s) to use for the ACTIVE 2025 dataset,
    WITHOUT assuming any of them exist and WITHOUT falling back to the
    discontinued Jan-2026 prototype paths.

    Priority:
      1. config.explicit_input_path (--input): one exact path, used
         as-is, no fallback.
      2. config.explicit_input_glob (--input-glob): glob match: ALL
         matches, sorted, returned together (opened as one combined
         dataset -- see open_dataset()).
      3. config.year2025_input_candidates: exact filenames, first that
         exists.
      4. config.year2025_input_glob: glob match, ALL matches, sorted.

    Returns (paths, source_label). paths is [] if nothing was found --
    the caller must report that clearly rather than crash. A full year
    of daily data is commonly split across several files (e.g. one per
    month); that's why this returns a list and glob matches are ALL
    kept, not just the first.
    """
    if config.explicit_input_path:
        p = config.resolve(config.explicit_input_path)
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

    for candidate in config.year2025_input_candidates:
        p = config.resolve(candidate)
        if p.exists():
            return [p], f"2025 candidate: {candidate}"

    glob_matches = sorted(Path(config.project_root).resolve().glob(config.year2025_input_glob))
    if glob_matches:
        return glob_matches, f"2025 glob '{config.year2025_input_glob}': {len(glob_matches)} file(s)"

    logger.warning(
        "No active 2025 input file found. Tried year2025_input_candidates=%s "
        "and glob '%s' (resolved against project_root=%s). The discontinued "
        "Jan-2026 prototype paths are deliberately NOT used as a fallback -- "
        "call find_obsolete_jan2026_file() explicitly if you really want "
        "that old prototype, not this function.",
        config.year2025_input_candidates, config.year2025_input_glob, config.project_root,
    )
    return [], "none found"


def find_obsolete_jan2026_file(config) -> Tuple[Optional[Path], str]:
    """
    DISCONTINUED. Kept only so the old Jan-2026 prototype can still be
    re-run manually on purpose (e.g. a side-by-side comparison). Never
    called by the default pipeline flow -- see find_input_files().
    """
    for candidate in config.obsolete_jan2026_monthly_candidates:
        p = config.resolve(candidate)
        if p.exists():
            return p, f"OBSOLETE Jan-2026 monthly candidate: {candidate}"
    for candidate in config.obsolete_jan2026_fifteenday_candidates:
        p = config.resolve(candidate)
        if p.exists():
            return p, f"OBSOLETE Jan-2026 15-day candidate: {candidate}"
    return None, "none found (obsolete Jan-2026 paths)"


def open_dataset(paths: List[Path], use_dask_chunking: bool = True) -> xr.Dataset:
    """
    Open one or several NetCDF files with xarray, distinguishing
    "doesn't exist" from "exists but isn't valid NetCDF yet (e.g. a
    partial download)" from other errors -- with a clear message for
    each, instead of a raw traceback. Several files are combined with
    xr.open_mfdataset(combine="by_coords"), the same approach
    oceanembed_step5/data_loader.py already uses for its own
    multi-file surface/target inputs.

    use_dask_chunking=True (default) passes chunks="auto" so the
    dataset is dask-backed and lazily/chunk-read rather than pulling
    the whole file into memory just to open and inspect it -- "auto"
    rather than a fixed per-dimension size because the real dimension
    names aren't known yet at open time (detect_all_coords() runs
    after this). Multi-file opens are always dask-backed regardless
    (open_mfdataset requires it).
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
        f"None of the expected temperature variable names {candidates} "
        f"were found. Actual variables in the file: {list(ds.data_vars)}. "
        f"Add the real name to temperature_var_candidates in "
        f"config_temperature.py."
    )


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
        f"coords: {list(ds.coords)}. Add the real name to "
        f"config_temperature.py."
    )


def detect_all_coords(ds: xr.Dataset, config) -> DetectedCoords:
    return DetectedCoords(
        time=detect_coordinate_name(ds, config.time_name_candidates, "time"),
        depth=detect_coordinate_name(ds, config.depth_name_candidates, "depth"),
        latitude=detect_coordinate_name(ds, config.latitude_name_candidates, "latitude"),
        longitude=detect_coordinate_name(ds, config.longitude_name_candidates, "longitude"),
    )


# ---------------------------------------------------------------------
# Coordinate validation (read-only -- never modifies ds)
# ---------------------------------------------------------------------

def validate_coordinates(ds: xr.Dataset, coords: DetectedCoords) -> Dict[str, object]:
    """
    Inspect (never modify) the four coordinate arrays and return a dict
    of findings for the validation report.

    Raises ValueError only for problems serious enough that continuing
    would be scientifically misleading (NaN inside a coordinate array
    itself). Duplicates, descending order, and an unusual longitude
    convention are reported, not raised -- this module's subsetting
    functions use positional boolean-mask selection throughout, so they
    do not require ascending or duplicate-free coordinates to behave
    correctly; they are still worth flagging as unusual for a validated
    ocean product.
    """
    report: Dict[str, object] = {"warnings": []}

    time_vals = pd.to_datetime(ds[coords.time].values)
    depth_vals = np.asarray(ds[coords.depth].values, dtype=float)
    lat_vals = np.asarray(ds[coords.latitude].values, dtype=float)
    lon_vals = np.asarray(ds[coords.longitude].values, dtype=float)

    for label, vals in (("depth", depth_vals), ("latitude", lat_vals), ("longitude", lon_vals)):
        if np.isnan(vals).any():
            raise ValueError(f"'{label}' coordinate contains NaN value(s); cannot proceed safely.")
    if pd.isna(time_vals).any():
        raise ValueError("'time' coordinate contains NaT/NaN value(s); cannot proceed safely.")

    def _dup_info(vals: np.ndarray) -> Tuple[bool, List[float]]:
        uniq, counts = np.unique(vals, return_counts=True)
        dups = uniq[counts > 1]
        return bool(len(dups) > 0), dups[:5].tolist()

    for label, vals in (("depth", depth_vals), ("latitude", lat_vals), ("longitude", lon_vals)):
        has_dup, examples = _dup_info(vals)
        report[f"{label}_has_duplicates"] = has_dup
        if has_dup:
            report["warnings"].append(
                f"'{label}' coordinate has duplicate values (examples: {examples})."
            )
    has_dup_time, _ = _dup_info(time_vals.values.astype("int64"))
    report["time_has_duplicate_timestamps"] = has_dup_time
    if has_dup_time:
        report["warnings"].append("Time coordinate has duplicate timestamps.")

    report["time_ascending"] = bool(len(time_vals) < 2 or np.all(np.diff(time_vals.values.astype("int64")) > 0))
    report["depth_ascending"] = bool(len(depth_vals) < 2 or np.all(np.diff(depth_vals) > 0))
    report["latitude_ascending"] = bool(len(lat_vals) < 2 or lat_vals[0] < lat_vals[-1])
    report["longitude_ascending"] = bool(len(lon_vals) < 2 or lon_vals[0] < lon_vals[-1])
    for label, is_asc in (("depth", report["depth_ascending"]), ("latitude", report["latitude_ascending"])):
        if not is_asc:
            report["warnings"].append(f"'{label}' coordinate is not ascending; reported, not reordered.")

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
    report["n_depth"] = int(len(depth_vals))
    report["n_latitude"] = int(len(lat_vals))
    report["n_longitude"] = int(len(lon_vals))
    report["available_depths_m"] = depth_vals.tolist()
    report["first_timestamp"] = str(time_vals[0]) if len(time_vals) else None
    report["last_timestamp"] = str(time_vals[-1]) if len(time_vals) else None

    if len(time_vals) > 1:
        sorted_times = np.sort(time_vals.values)
        diffs_days = np.diff(sorted_times).astype("timedelta64[h]").astype(float) / 24.0
        report["median_time_step_days"] = float(np.median(diffs_days))
        report["is_daily"] = bool(np.isclose(np.median(diffs_days), 1.0, atol=0.05))
        # Normalize BOTH sides to midnight before diffing calendar dates --
        # GLORYS daily timestamps are often stamped at noon (12:00 UTC).
        # Building expected_days from an un-normalized start/end would
        # compare noon-stamped expected dates against midnight-normalized
        # actual dates and falsely report every date as "missing".
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
    """
    lat_vals = ds[coords.latitude].values
    lon_vals = ds[coords.longitude].values
    time_vals = pd.to_datetime(ds[coords.time].values)

    lat_mask = (lat_vals >= config.lat_min) & (lat_vals <= config.lat_max)
    lon_mask = (lon_vals >= config.lon_min) & (lon_vals <= config.lon_max)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    time_mask = (time_vals >= start) & (time_vals <= end)

    report: Dict[str, object] = {
        "warnings": [],
        "requested_lat_range": (config.lat_min, config.lat_max),
        "requested_lon_range": (config.lon_min, config.lon_max),
        "requested_time_range": (config.start_date, config.end_date),
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
            f"time[{time_vals.min()},{time_vals.max()}]."
        )

    actual_lat = lat_vals[lat_mask]
    actual_lon = lon_vals[lon_mask]
    actual_time = time_vals[time_mask]
    full_lat = actual_lat.min() <= config.lat_min + 1e-6 and actual_lat.max() >= config.lat_max - 1e-6
    full_lon = actual_lon.min() <= config.lon_min + 1e-6 and actual_lon.max() >= config.lon_max - 1e-6
    full_time = actual_time.min() <= start and actual_time.max() >= end

    report["actual_lat_range"] = (float(actual_lat.min()), float(actual_lat.max()))
    report["actual_lon_range"] = (float(actual_lon.min()), float(actual_lon.max()))
    report["actual_time_range"] = (str(actual_time.min()), str(actual_time.max()))
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
# Depth selection
# ---------------------------------------------------------------------

def select_depths(da: xr.DataArray, coords: DetectedCoords, config) -> Tuple[xr.DataArray, List[Dict[str, object]]]:
    """
    Select config.requested_depths_m from da's depth coordinate using
    integer positions (never label-based .sel on floats, which risks
    float32/float64 comparison mismatches).

    "exact": keep only requested depths with a match (tolerance 1e-6);
    anything else is reported as unavailable and excluded.
    "nearest": for every requested depth, pick the closest available
    depth and report (requested, actual, difference). If
    depth_max_tolerance_m is set and exceeded, that depth is excluded
    instead of silently kept.

    Never fabricates a depth that is not actually in the file, and
    never silently reduces the count below what was actually
    available/requested.
    """
    available = np.asarray(da[coords.depth].values, dtype=float)
    selected_indices: List[int] = []
    depth_report: List[Dict[str, object]] = []

    for requested in config.requested_depths_m:
        requested = float(requested)
        entry: Dict[str, object] = {"requested_m": requested}

        # --- Targeted surface exception (NOT a general tolerance
        # increase, NOT a require_all_requested_depths bypass): a
        # nominal request at or near the surface (<= surface_depth_max_m)
        # that is shallower than every available depth is mapped to the
        # single shallowest available level, because ocean reanalysis
        # products essentially never have an exact 0 m grid level. Every
        # other requested depth below is completely unaffected by this
        # branch and still goes through standard exact/nearest matching.
        if (
            config.allow_surface_depth_mapping
            and requested <= config.surface_depth_max_m
            and requested < available.min() - 1e-9
        ):
            idx = int(np.argmin(available))
            actual = float(available[idx])
            diff = abs(actual - requested)
            entry.update(
                actual_m=actual, difference_m=diff, status="SURFACE_MAPPED",
                note=(f"Nominal surface request ({requested}m) has no available GLORYS level at "
                      f"or near it; mapped to the shallowest available level ({actual}m) because "
                      f"allow_surface_depth_mapping=True and requested <= surface_depth_max_m "
                      f"({config.surface_depth_max_m}m). Targeted exception for this request "
                      f"only -- all other requested depths use standard exact/nearest matching."),
            )
            depth_report.append(entry)
            selected_indices.append(idx)
            continue

        if requested < available.min() - 1e-9 or requested > available.max() + 1e-9:
            entry.update(actual_m=None, difference_m=None, status="OUT_OF_RANGE",
                         note=f"outside available range [{available.min():.3f}, {available.max():.3f}] m")
            depth_report.append(entry)
            continue

        exact_idx = np.where(np.isclose(available, requested, atol=1e-6))[0]
        if len(exact_idx) > 0:
            idx = int(exact_idx[0])
            entry.update(actual_m=float(available[idx]), difference_m=0.0, status="EXACT")
            depth_report.append(entry)
            selected_indices.append(idx)
            continue

        if config.depth_selection_method == "exact":
            entry.update(actual_m=None, difference_m=None, status="NO_EXACT_MATCH",
                         note="depth_selection_method='exact' and no exact match exists")
            depth_report.append(entry)
            continue

        idx = int(np.argmin(np.abs(available - requested)))
        actual = float(available[idx])
        diff = abs(actual - requested)
        if config.depth_max_tolerance_m is not None and diff > config.depth_max_tolerance_m:
            entry.update(actual_m=actual, difference_m=diff, status="EXCLUDED_TOLERANCE",
                         note=f"nearest match {actual}m differs by {diff:.3f}m > tolerance {config.depth_max_tolerance_m}m")
            depth_report.append(entry)
            continue

        entry.update(actual_m=actual, difference_m=diff, status="NEAREST")
        depth_report.append(entry)
        selected_indices.append(idx)

    if not selected_indices:
        raise ValueError(
            f"No requested depth could be matched to the file's available "
            f"depths {available.tolist()}. Requested: {config.requested_depths_m}."
        )

    # De-duplicate (two requested depths could round to the same nearest
    # level) and sort by the ACTUAL DEPTH VALUE (not by position index).
    # GLORYS depth is virtually always stored ascending already, so this
    # is a no-op in the normal case -- but oceanembed_step5/validation.py
    # hard-requires the target's depth coordinate to be strictly
    # ascending (np.diff(depth) > 0) and raises otherwise, so sorting by
    # value here (rather than assuming position order == value order)
    # guarantees that Step 5 requirement is met regardless of the source
    # file's own storage order.
    unique_indices = sorted(set(selected_indices), key=lambda i: available[i])
    subset = da.isel({coords.depth: unique_indices})
    return subset, depth_report


# ---------------------------------------------------------------------
# Missing values / units / plausibility (report-only, never mutates)
# ---------------------------------------------------------------------

def compute_valid_mask(da) -> xr.DataArray:
    """True where finite (a real, usable ocean value); False for
    NaN/Inf (land or otherwise invalid/missing). xarray decodes NetCDF
    _FillValue / missing_value into NaN by default (mask_and_scale=True),
    so this catches those too -- same convention as
    oceanembed_step5/preprocessing.py's compute_valid_mask.

    Returns a LAZY xr.DataArray (dask-backed if da is) rather than a
    materialized numpy array -- np.isfinite(da) dispatches through
    xarray's array-ufunc protocol and stays lazy on a dask-backed
    input, so calling this does not by itself force the full array
    into memory. Callers should .compute() it only where a concrete
    array is actually needed."""
    return np.isfinite(da)


def inspect_missing_by_dimension(da: xr.DataArray, coords: DetectedCoords) -> Dict[str, object]:
    """
    Dask/xarray-aware: per-depth and per-time missing-value breakdown,
    plus EXPLICIT detection of any depth level or time step that is
    COMPLETELY missing (100% NaN/Inf across every other dimension).
    An aggregate missing-value count (inspect_missing_values) can
    hide this entirely -- e.g. one corrupted/blank day, or one
    depth level that failed to download, buried in an otherwise-mostly-
    valid year looks the same in an aggregate percentage as scattered
    noise, but is a much more serious, structural problem.

    Every reduction here collapses to at most max(n_depth, n_time)
    numbers, bundled into one small Dataset and .compute()'d ONCE --
    never touches the full array via .values. A deliberately SEPARATE
    reduction pass from inspect_missing_values/flag_implausible_values
    (not merged into one shared .compute()) so as not to touch or risk
    that already-verified code -- costs one extra pass over the data,
    not extra peak memory.
    """
    finite = np.isfinite(da)
    dims_other_than_depth = [d for d in da.dims if d != coords.depth]
    dims_other_than_time = [d for d in da.dims if d != coords.time]

    summary = xr.Dataset({
        "valid_by_depth": finite.sum(dim=dims_other_than_depth),
        "valid_by_time": finite.sum(dim=dims_other_than_time),
    }).compute()

    n_per_depth = 1
    for d in dims_other_than_depth:
        n_per_depth *= da.sizes[d]
    n_per_time = 1
    for d in dims_other_than_time:
        n_per_time *= da.sizes[d]

    depth_vals = np.asarray(da[coords.depth].values, dtype=float)
    time_vals = da[coords.time].values

    by_depth: List[Dict[str, object]] = []
    completely_missing_depths: List[float] = []
    valid_by_depth = summary["valid_by_depth"].values
    for i, dv in enumerate(depth_vals):
        valid = int(valid_by_depth[i])
        frac = (valid / n_per_depth) if n_per_depth else None
        by_depth.append({"depth_m": float(dv), "valid_count": valid, "valid_fraction": frac})
        if valid == 0:
            completely_missing_depths.append(float(dv))

    by_time: List[Dict[str, object]] = []
    completely_missing_times: List[str] = []
    valid_by_time = summary["valid_by_time"].values
    for i, tv in enumerate(time_vals):
        valid = int(valid_by_time[i])
        frac = (valid / n_per_time) if n_per_time else None
        ts = str(pd.Timestamp(tv))
        by_time.append({"time": ts, "valid_count": valid, "valid_fraction": frac})
        if valid == 0:
            completely_missing_times.append(ts)

    return {
        "valid_fraction_by_depth": by_depth,
        "valid_fraction_by_time": by_time,
        "completely_missing_depths_m": completely_missing_depths,
        "completely_missing_times": completely_missing_times,
    }


def inspect_missing_values(da) -> Dict[str, object]:
    """
    Dask/xarray-aware: takes the xr.DataArray directly (NOT .values)
    and uses a lazy reduction, .compute()-ing only the small scalar
    result -- never materializes the full array. This replaced an
    earlier version that took a raw numpy array via da.values, which
    forced full materialization and caused a real OOM
    (365 x 15 x 301 x 721 float64 = 8.85 GiB) on a real 2025 file.
    np.isfinite(da), not da.isnull(), on purpose: isnull() alone would
    not count +/-Inf as invalid, silently changing validation behavior
    from the original np.isfinite-based check.
    """
    total = int(da.size)  # static shape property -- no computation triggered
    n_valid = int(np.isfinite(da).sum().compute())
    n_invalid = total - n_valid
    return {
        "total_cells": total,
        "nan_or_inf_count": n_invalid,
        "valid_count": n_valid,
        "fraction_invalid": (n_invalid / total) if total else None,
    }


def flag_implausible_values(da, low: float = -5.0, high: float = 45.0) -> Dict[str, object]:
    """
    Dask/xarray-aware equivalent of the original array-based check:
    soft check on FINITE values only -- ocean potential temperature
    realistically falls within roughly [low, high] deg C. Values
    outside this range among already-finite cells usually mean a
    _FillValue/missing_value sentinel (e.g. 1e20, -32767) slipped
    through unmasked, not real physics. Never modifies data -- report
    only.

    Every scalar reduction (count, min, max, mean) is bundled into one
    small Dataset and .compute()'d ONCE, so dask shares the underlying
    chunk reads across all five stats instead of re-scanning the
    source data once per stat. Never touches .values on the full array.
    """
    finite_mask = np.isfinite(da)
    finite_only = da.where(finite_mask)  # non-finite cells -> NaN, skipna reductions below ignore them
    implausible_mask = finite_mask & ((da < low) | (da > high))

    summary = xr.Dataset({
        "finite_count": finite_mask.sum(),
        "n_implausible": implausible_mask.sum(),
        "min_finite": finite_only.min(),
        "max_finite": finite_only.max(),
        "mean_finite": finite_only.mean(),
    }).compute()

    finite_count = int(summary["finite_count"])
    if finite_count == 0:
        return {"n_implausible": 0, "min_finite": None, "max_finite": None, "mean_finite": None}
    return {
        "n_implausible": int(summary["n_implausible"]),
        "min_finite": float(summary["min_finite"]),
        "max_finite": float(summary["max_finite"]),
        "mean_finite": float(summary["mean_finite"]),
    }


def check_units(da: xr.DataArray) -> Tuple[Optional[str], List[str]]:
    """Report the variable's declared units without assuming them.
    Returns (units_string_or_None, warnings)."""
    warnings: List[str] = []
    units = da.attrs.get("units")
    recognized = ("degc", "degrees_c", "celsius", "deg_c", "degree_celsius", "degrees celsius")
    if units is None:
        warnings.append(
            "Temperature variable has no 'units' attribute in the file. "
            "GLORYS thetao is conventionally degrees_C, but this was NOT "
            "verified here -- check source documentation before assuming."
        )
    elif str(units).strip().lower() not in recognized:
        warnings.append(
            f"Temperature variable units attribute is {units!r}, which "
            f"does not look like a recognized Celsius label -- verify "
            f"before treating values as degrees Celsius."
        )
    return units, warnings


# ---------------------------------------------------------------------
# Regridding -- OFFICIAL, default-on. Any failure is FATAL by design
# (see docstring below); there is deliberately no try/except-and-
# continue anywhere around this.
# ---------------------------------------------------------------------

def build_target_grid(config) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Explicitly construct the official target lat/lon grid from
    config's region bounds and target_resolution_deg -- NEVER derived
    from the source data's own (possibly slightly different)
    subsetted extent, so the official output grid is always exactly
    the documented one regardless of the source file's native points.

    Convention: cell-center, BOTH endpoints included.
    n_points = round((max - min) / spacing) + 1, then
    np.linspace(min, max, n_points) -- exact endpoints with no
    floating-point drift (unlike building the array via repeated
    addition, e.g. np.arange). For the default 5-30N / 45-105E bounds
    at 0.25 deg this gives exactly 101 latitude x 241 longitude points.
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

    FATAL BY DESIGN: this is the official, default-on regridding path
    ("the temperature pipeline must not produce the native GLORYS
    resolution as its official final output"). Any failure -- missing
    scipy, non-monotonic/duplicate/non-finite source coordinates,
    insufficient coverage, an interpolation error, memory error, or
    anything else -- is raised as RuntimeError/ValueError and MUST
    propagate; callers must not catch-and-continue with the native
    grid. Call with regrid_to_quarter_degree=False (--no-regrid) for
    an explicit, clearly-labeled non-official diagnostic run instead.

    Explicit, documented behavior:
    - Source lat/lon are validated immediately before interpolating:
      1-D, numeric, finite, no duplicates -- raised clearly if not.
    - A descending source coordinate is sorted ascending first via
      .sortby() -- an explicit, RECORDED operation (in the returned
      report), never silent -- because interpolation needs a
      consistently ordered source grid.
    - No extrapolation: kwargs=dict(bounds_error=False,
      fill_value=NaN) passed to the interpolator. Target cells outside
      the source data's actual coverage become NaN, never a
      guessed/extrapolated value.
    - If literally every target cell ends up NaN after a technically
      "successful" interpolation call, that's still treated as fatal
      (almost certainly means the requested region and the source
      file's actual coverage don't overlap at all) rather than
      silently shipping an empty official output.
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

    # Source resolution for the metadata report, computed from the
    # (now-ascending) native grid actually being interpolated from.
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

    try:
        n_time = da.sizes[coords.time]
        batch_size = config.regrid_time_batch_size
        regridded_batches = []
        for start in range(0, n_time, batch_size):
            end = min(start + batch_size, n_time)
            # Interpolating the FULL time range in one .interp() call
            # needs the whole native-resolution array in memory to
            # build the interpolator, regardless of dask chunking
            # upstream -- this is what actually OOM'd on a real file
            # (365 x 15 x 301 x 721 float64 = 8.85 GiB). Horizontal
            # interpolation for one day never depends on any other
            # day, so batching along time changes peak memory, not the
            # numerical result (verified: batched vs. non-batched
            # regridding produce identical output).
            batch = da.isel({coords.time: slice(start, end)})
            regridded_batch = batch.interp(
                {coords.latitude: new_lat, coords.longitude: new_lon},
                method=config.regrid_method,
                kwargs={"bounds_error": False, "fill_value": np.nan},
            )
            # Each batch is small by construction (batch_size instead
            # of the full time range) -- materializing it here is safe
            # and keeps peak memory bounded regardless of whether
            # .interp() itself stayed lazy internally.
            regridded_batch = regridded_batch.compute() if hasattr(regridded_batch.data, "compute") else regridded_batch
            regridded_batches.append(regridded_batch)
        regridded = xr.concat(regridded_batches, dim=coords.time) if len(regridded_batches) > 1 else regridded_batches[0]
        report["regrid_time_batch_size"] = batch_size
        report["n_regrid_batches"] = len(regridded_batches)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY regridding failure is fatal, re-raised with context
        raise RuntimeError(
            f"Regridding to the official {config.target_resolution_deg} deg target grid FAILED "
            f"({type(exc).__name__}: {exc}). This is FATAL BY DESIGN -- the official pipeline "
            f"must never silently fall back to the native grid. Common causes: missing scipy "
            f"(pip install scipy), non-monotonic/duplicate source coordinates, or insufficient "
            f"source coverage for the requested region. Fix the underlying cause and re-run, or "
            f"pass --no-regrid for an explicit, clearly-marked non-official native-grid "
            f"DIAGNOSTIC run instead of the official one."
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
            f"regridding (outside source coverage or already-invalid source cells)."
        )

    return regridded, report


def verify_final_grid(
    da: xr.DataArray, coords: DetectedCoords, config, tolerance_deg: Optional[float] = None
) -> Dict[str, object]:
    """
    Explicit post-regrid (or post-depth-selection, in native-diagnostic
    mode) validation: "calculate the actual spacing from the resulting
    coordinates rather than trusting configuration alone." Raises
    ValueError with the specific violation -- never just reports and
    continues -- for depth count/ordering and (when
    config.regrid_to_quarter_degree) grid spacing/range, which apply
    to the OFFICIAL output unconditionally.
    """
    tolerance_deg = config.final_grid_tolerance_deg if tolerance_deg is None else tolerance_deg
    lat = np.asarray(da[coords.latitude].values, dtype=float)
    lon = np.asarray(da[coords.longitude].values, dtype=float)
    depth = np.asarray(da[coords.depth].values, dtype=float)
    report: Dict[str, object] = {}

    for label, vals in ((coords.latitude, lat), (coords.longitude, lon)):
        if len(np.unique(vals)) != len(vals):
            raise ValueError(f"Final output grid: '{label}' contains duplicate coordinate values.")
        if len(vals) > 1 and not np.all(np.diff(vals) > 0):
            raise ValueError(f"Final output grid: '{label}' is not strictly monotonic ascending.")

    lat_spacing = float(np.median(np.diff(lat))) if len(lat) > 1 else None
    lon_spacing = float(np.median(np.diff(lon))) if len(lon) > 1 else None
    report.update(
        latitude_min=float(lat.min()), latitude_max=float(lat.max()), n_latitude=int(len(lat)),
        latitude_spacing_deg=lat_spacing,
        longitude_min=float(lon.min()), longitude_max=float(lon.max()), n_longitude=int(len(lon)),
        longitude_spacing_deg=lon_spacing,
    )

    if config.regrid_to_quarter_degree:
        if lat_spacing is None or abs(lat_spacing - config.target_resolution_deg) > tolerance_deg:
            raise ValueError(
                f"Final output latitude spacing {lat_spacing} deg != required "
                f"{config.target_resolution_deg} deg (tolerance {tolerance_deg})."
            )
        if lon_spacing is None or abs(lon_spacing - config.target_resolution_deg) > tolerance_deg:
            raise ValueError(
                f"Final output longitude spacing {lon_spacing} deg != required "
                f"{config.target_resolution_deg} deg (tolerance {tolerance_deg})."
            )
        if abs(lat.min() - config.lat_min) > tolerance_deg or abs(lat.max() - config.lat_max) > tolerance_deg:
            raise ValueError(
                f"Final output latitude range [{lat.min()},{lat.max()}] != requested "
                f"[{config.lat_min},{config.lat_max}] (tolerance {tolerance_deg})."
            )
        if abs(lon.min() - config.lon_min) > tolerance_deg or abs(lon.max() - config.lon_max) > tolerance_deg:
            raise ValueError(
                f"Final output longitude range [{lon.min()},{lon.max()}] != requested "
                f"[{config.lon_min},{config.lon_max}] (tolerance {tolerance_deg})."
            )

    if len(depth) != len(config.requested_depths_m):
        raise ValueError(
            f"Final output has {len(depth)} depth level(s), expected exactly "
            f"{len(config.requested_depths_m)}."
        )
    if len(depth) > 1 and not np.all(np.diff(depth) > 0):
        raise ValueError("Final output depth coordinate is not strictly ascending.")

    expected_dims = {coords.time, coords.depth, coords.latitude, coords.longitude}
    if set(da.dims) != expected_dims:
        raise ValueError(f"Final output has unexpected dimensions {da.dims}, expected exactly {sorted(expected_dims)}.")

    return report


# ---------------------------------------------------------------------
# Step 5 output compatibility -- coordinate standardization
# ---------------------------------------------------------------------

def standardize_output_coordinates(
    da: xr.DataArray, coords: DetectedCoords, config
) -> Tuple[xr.DataArray, "DetectedCoords", Dict[str, str]]:
    """
    Rename the DETECTED source coordinate names to the standardized
    output names in config (output_time_name / output_depth_name /
    output_latitude_name / output_longitude_name), which were verified
    against oceanembed_step5/config.py's CoordNames defaults (time /
    latitude / longitude / depth). Pure relabeling -- xarray's
    .rename() does not touch values -- and never modifies the source
    file, only the in-memory array about to be written to the NEW
    output file.

    Returns (renamed_da, new_coords, rename_map). rename_map only
    contains entries that actually changed name -- most GLORYS files
    already use "latitude"/"longitude"/"depth"/"time", so this is
    often empty; when it isn't, it's recorded in the metadata so the
    rename is traceable.
    """
    rename_map: Dict[str, str] = {}
    target_names = {
        coords.time: config.output_time_name,
        coords.depth: config.output_depth_name,
        coords.latitude: config.output_latitude_name,
        coords.longitude: config.output_longitude_name,
    }
    for source_name, output_name in target_names.items():
        if source_name != output_name:
            rename_map[source_name] = output_name

    renamed = da.rename(rename_map) if rename_map else da
    new_coords = DetectedCoords(
        time=config.output_time_name,
        depth=config.output_depth_name,
        latitude=config.output_latitude_name,
        longitude=config.output_longitude_name,
    )
    return renamed, new_coords, rename_map


# ---------------------------------------------------------------------
# Strict-mode verification (Phase 3 audit requirements) -- these
# raise ValueError with a specific, actionable message; they never
# silently downgrade "incomplete" to "complete".
# ---------------------------------------------------------------------

def verify_depth_count(depth_report: List[Dict[str, object]], config) -> None:
    """Raise if config.require_all_requested_depths and not every
    requested depth was matched (status EXACT, NEAREST, or the
    targeted SURFACE_MAPPED exception -- see select_depths)."""
    if not config.require_all_requested_depths:
        return
    failed = [d for d in depth_report if d["status"] not in ("EXACT", "NEAREST", "SURFACE_MAPPED")]
    if failed:
        raise ValueError(
            f"require_all_requested_depths=True but {len(failed)} of "
            f"{len(depth_report)} requested depths could not be matched: "
            f"{failed}. Either the file genuinely lacks these depths (check "
            f"available_depths_m in the coordinate report), or set "
            f"require_all_requested_depths=False / adjust "
            f"depth_max_tolerance_m if this is expected."
        )


def infer_date_range_from_coords(coord_report: Dict[str, object]) -> Tuple[str, str]:
    """
    Infer a processing date range from the INPUT FILE's own time
    coordinate (coord_report["first_timestamp"]/["last_timestamp"],
    from validate_coordinates) -- the full calendar year(s) the data
    actually spans, e.g. a file spanning 2021-01-01..2021-12-31 infers
    exactly that; a file spanning 2021-06-01..2022-05-31 infers
    2021-01-01..2022-12-31 (both years it touches, in full). Used only
    when neither an explicit --start-date/--end-date nor a config
    default was given -- see preprocess_temperature.py's date-range
    resolution step, which always prefers an explicit value over this.
    Never assumes any specific year.
    """
    first_ts = pd.Timestamp(coord_report["first_timestamp"])
    last_ts = pd.Timestamp(coord_report["last_timestamp"])
    inferred_start = f"{first_ts.year}-01-01"
    inferred_end = f"{last_ts.year}-12-31"
    return inferred_start, inferred_end


def check_output_directory_year_consistency(
    out_dir: Path, metadata_filenames: List[str], effective_start: str, effective_end: str
) -> Optional[str]:
    """
    If out_dir already contains a previous run's metadata OR dry-run
    report for a DIFFERENT processing year than this run's, that's
    very likely an accidental overwrite-in-progress (e.g. a shared
    output_dir reused across 2021/2022/.../2025 runs instead of a
    year-specific --output-dir). Checks every filename in
    metadata_filenames (both the real metadata JSON and the dry-run
    report -- a directory that's only ever seen dry runs so far would
    have no metadata JSON yet, and checking metadata_filename alone
    would silently miss that case). Returns a warning message
    describing the mismatch, or None if nothing previous is found or
    the year(s) match. Never raises, never deletes/modifies the
    existing file(s) -- purely a read-only check; the caller decides
    whether to warn or fail based on config.fail_on_output_year_mismatch.
    """
    for filename in metadata_filenames:
        meta_path = out_dir / filename
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                previous = json.load(f)
            prev_range = previous.get("config_summary", {}).get("requested_time_range")
            if not prev_range or len(prev_range) != 2 or not prev_range[0] or not prev_range[1]:
                continue
            prev_years = {pd.Timestamp(prev_range[0]).year, pd.Timestamp(prev_range[1]).year}
            current_years = {pd.Timestamp(effective_start).year, pd.Timestamp(effective_end).year}
            if prev_years != current_years:
                return (
                    f"Output directory '{out_dir}' already contains a previous run's output "
                    f"('{filename}') for year(s) {sorted(prev_years)}, but this run is processing "
                    f"year(s) {sorted(current_years)}. Writing here will overwrite that previous "
                    f"run's outputs. Use a year-specific --output-dir to keep them separate."
                )
        except Exception:  # noqa: BLE001 - this check is a courtesy, never allowed to block a real run
            continue
    return None


def verify_temporal_coverage(coord_report: Dict[str, object], region_report: Dict[str, object], config) -> None:
    """Raise if config.require_complete_temporal_coverage and actual
    time coverage doesn't match what was requested (incomplete overlap,
    missing dates, or a short time-step count)."""
    if not config.require_complete_temporal_coverage:
        return

    problems: List[str] = []
    if not region_report.get("full_region_time_coverage", False):
        problems.append(
            f"requested time range {config.start_date}..{config.end_date} is "
            f"not fully covered; actual matched range "
            f"{region_report.get('actual_time_range')}."
        )
    if coord_report.get("missing_dates"):
        problems.append(
            f"{len(coord_report['missing_dates'])} missing date(s) in the file's own time coordinate."
        )

    expected = config.expected_time_step_count
    if expected is None:
        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        expected = (end - start).days + 1
    actual = region_report.get("n_time_matched")
    if actual is not None and actual != expected:
        problems.append(f"expected {expected} daily time step(s) for the requested range, matched {actual}.")

    if problems:
        raise ValueError(
            "require_complete_temporal_coverage=True but temporal coverage is "
            "incomplete: " + " ".join(problems) + " Set "
            "require_complete_temporal_coverage=False (and date_range_confirmed=True "
            "with an explicit narrower start_date/end_date, if a partial range is "
            "actually intended) to proceed anyway."
        )


def check_implausible_fraction(
    implausible_report: Dict[str, object], missing_report: Dict[str, object], config
) -> None:
    """Raise if the fraction of finite-but-implausible values exceeds
    config.max_implausible_fraction -- the hard safety backstop behind
    flag_implausible_values()'s (non-fatal) per-value report."""
    finite_count = missing_report.get("valid_count", 0)
    n_implausible = implausible_report.get("n_implausible", 0)
    if not finite_count:
        return
    fraction = n_implausible / finite_count
    if fraction > config.max_implausible_fraction:
        raise ValueError(
            f"{n_implausible}/{finite_count} finite values ({fraction:.1%}) fall "
            f"outside the plausible [{config.implausible_low_c}, "
            f"{config.implausible_high_c}] degC range, exceeding "
            f"max_implausible_fraction={config.max_implausible_fraction:.1%}. This "
            f"usually means an unmasked _FillValue/missing_value sentinel, a units "
            f"mismatch, or the wrong variable was selected -- unsafe to proceed. "
            f"Investigate before raising max_implausible_fraction."
        )


# ---------------------------------------------------------------------
# Atomic writes (Phase 3G): a crash mid-write must never leave a
# corrupt or misleadingly-named file at the final output path.
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
    ".temperature_target.tmp1234.npy", not "...npy.tmp1234") on
    purpose: np.save() appends ".npy" itself to any filename that
    doesn't already end with it, which would otherwise silently write
    to a DIFFERENT path than the one this function replaces from.
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


# ---------------------------------------------------------------------
# Output self-verification (Phase 4 test: reopen and check)
# ---------------------------------------------------------------------

def verify_output_file(nc_path: Path, config, expected_shape: Tuple[int, int, int, int]) -> Dict[str, object]:
    """
    Reopen the just-written NetCDF with a FRESH xr.open_dataset call
    (never reusing the in-memory object that was written) and check it
    actually contains what was intended: variable name, dimension
    names/order, dtype, units, mask presence/shape, and overall shape.
    Raises ValueError with the specific mismatch rather than assuming
    success just because to_netcdf() didn't raise.
    """
    ds = xr.open_dataset(nc_path)
    try:
        report: Dict[str, object] = {"checked_file": str(nc_path), "warnings": []}
        if "temperature_target" not in ds.data_vars:
            raise ValueError(
                f"Reopened {nc_path} has no 'temperature_target' variable; found {list(ds.data_vars)}."
            )
        da = ds["temperature_target"]
        expected_dims = (
            config.output_time_name, config.output_depth_name,
            config.output_latitude_name, config.output_longitude_name,
        )
        if tuple(da.dims) != expected_dims:
            raise ValueError(f"Reopened dims {da.dims} != expected {expected_dims}.")
        if tuple(da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened shape {da.shape} != expected {expected_shape}.")
        if da.sizes[config.output_depth_name] != len(config.requested_depths_m):
            report["warnings"].append(
                f"Reopened depth count {da.sizes[config.output_depth_name]} != requested "
                f"{len(config.requested_depths_m)} (only possible if require_all_requested_depths=False)."
            )
        if "valid_mask" not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no 'valid_mask' variable.")
        mask_da = ds["valid_mask"]
        if tuple(mask_da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened valid_mask shape {mask_da.shape} != expected {expected_shape}.")

        report["reopened_dtype"] = str(da.dtype)
        report["reopened_units"] = da.attrs.get("units")
        report["reopened_mask_dtype"] = str(mask_da.dtype)
        report["reopened_dims"] = list(da.dims)
        report["reopened_shape"] = list(da.shape)

        # Grid check on the REOPENED file (independent of whatever was
        # computed in-memory before writing) -- this is what
        # "compatibility with the Step 5 loader" and "final output is
        # validated as 0.25 x 0.25" actually mean: what's really on
        # disk, not what was intended.
        coords_reopened = DetectedCoords(
            time=config.output_time_name, depth=config.output_depth_name,
            latitude=config.output_latitude_name, longitude=config.output_longitude_name,
        )
        grid_report = verify_final_grid(da, coords_reopened, config)
        report["reopened_grid"] = grid_report

        report["ok"] = True
        return report
    finally:
        ds.close()
