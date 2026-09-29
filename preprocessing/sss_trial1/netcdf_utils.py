"""
netcdf_utils.py (SSS)
======================
Reusable, file-agnostic helpers for inspecting, validating, and
regridding a CNR-style Sea Surface Salinity NetCDF file (variable
'sos'). Nothing here assumes a specific variable/coordinate name --
every name is detected against the real file's actual dims/coords/
data_vars and reported, never assumed.

This module is INDEPENDENT of the temperature pipeline's
netcdf_utils.py -- no shared state, no shared output paths. Several
functions (detection, coordinate validation, atomic writes, reopen
verification) follow the same *pattern* as the temperature pipeline on
purpose (proven conventions), but the regridding function is
DELIBERATELY DIFFERENT -- see regrid_weighted_linear() docstring.

Functions raise ValueError / FileNotFoundError / OSError with a
specific, actionable message on genuine problems, and never silently
reshape, rename, or coerce data. Subsetting uses boolean masks / integer
positions derived from coordinate VALUES (np.where(mask) -> isel),
never slice(min, max) or label-based .sel(), so a descending or
unusual coordinate never silently produces an empty or wrong result.
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

logger = logging.getLogger("preprocess_sss")


@dataclass
class DetectedCoords4D:
    """Coordinate names as actually found in the raw file (time, depth,
    latitude, longitude) -- before the singleton depth is dropped."""
    time: str
    depth: str
    latitude: str
    longitude: str


@dataclass
class DetectedCoords3D:
    """Coordinate names after the singleton depth dimension has been
    validated and dropped (time, latitude, longitude only)."""
    time: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------

def find_input_file(config) -> Tuple[Optional[Path], str]:
    """
    Decide which real file to use.

    Priority:
      1. config.explicit_input_path (--input): exact path, used as-is.
      2. config.default_input_candidate: the known 2025 raw SSS path
         given in the project brief -- still checked on disk, not
         assumed to exist.

    Returns (path_or_None, source_label). path is None if nothing was
    found -- the caller must report that clearly rather than crash.
    """
    if config.explicit_input_path:
        p = config.resolve(config.explicit_input_path)
        if p.exists():
            return p, f"explicit path (--input): {p}"
        return None, f"explicit path (--input, NOT FOUND): {p}"

    p = config.resolve(config.default_input_candidate)
    if p.exists():
        return p, f"default 2025 candidate: {config.default_input_candidate}"
    return None, f"default 2025 candidate (NOT FOUND): {config.default_input_candidate}"


def open_dataset(path: Path, use_dask_chunking: bool = True) -> xr.Dataset:
    """
    Open the NetCDF file with xarray, distinguishing "doesn't exist"
    from "exists but isn't valid NetCDF yet" from other errors, with a
    clear message for each. Uses xarray's default CF decoding
    (mask_and_scale=True, decode_times=True) so _FillValue,
    scale_factor, add_offset, and time units/calendar are all applied
    automatically and correctly -- never a hand-rolled decode.

    use_dask_chunking=True (default) passes chunks="auto" so the file
    is dask-backed and lazily read rather than pulled fully into memory
    just to open and inspect it.
    """
    if path is None:
        raise FileNotFoundError("No input file path given to open_dataset().")
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}.")
    try:
        ds = xr.open_dataset(path, chunks="auto" if use_dask_chunking else None)
    except Exception as exc:  # noqa: BLE001 - re-raised below with context
        raise OSError(
            f"Found {path} but could not open as NetCDF ({type(exc).__name__}: {exc}). "
            f"If a download is still in progress this can happen with a partially-written "
            f"file -- wait for it to finish and re-run."
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
        f"None of the expected SSS variable names {candidates} were found. "
        f"Actual variables in the file: {list(ds.data_vars)}. Add the real "
        f"name to sss_var_candidates in config_sss.py."
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
        f"None of the expected '{logical_name}' coordinate names {candidates} "
        f"were found. Actual dims: {list(ds.dims)}, actual coords: {list(ds.coords)}. "
        f"Add the real name to config_sss.py."
    )


def detect_all_coords(ds: xr.Dataset, config) -> DetectedCoords4D:
    return DetectedCoords4D(
        time=detect_coordinate_name(ds, config.time_name_candidates, "time"),
        depth=detect_coordinate_name(ds, config.depth_name_candidates, "depth"),
        latitude=detect_coordinate_name(ds, config.latitude_name_candidates, "latitude"),
        longitude=detect_coordinate_name(ds, config.longitude_name_candidates, "longitude"),
    )


# ---------------------------------------------------------------------
# Coordinate validation (read-only -- never modifies ds)
# ---------------------------------------------------------------------

def validate_coordinates(ds: xr.Dataset, coords: DetectedCoords4D) -> Dict[str, object]:
    """
    Inspect (never modify) the four coordinate arrays and return a dict
    of findings. Raises ValueError only for problems serious enough
    that continuing would be scientifically misleading (NaN inside a
    coordinate array itself); everything else (duplicates, descending
    order, unusual longitude convention) is reported, not raised.
    """
    report: Dict[str, object] = {"warnings": []}

    time_vals = pd.to_datetime(ds[coords.time].values)
    depth_vals = np.asarray(ds[coords.depth].values, dtype=float)
    lat_vals = np.asarray(ds[coords.latitude].values, dtype=float)
    lon_vals = np.asarray(ds[coords.longitude].values, dtype=float)

    for label, vals in (("depth", depth_vals), ("latitude", lat_vals), ("longitude", lon_vals)):
        if np.isnan(vals).any():
            raise ValueError(f"'{label}' coordinate contains NaN value(s); cannot proceed safely.")
        if not np.issubdtype(vals.dtype, np.number):
            raise ValueError(f"'{label}' coordinate is not numeric (dtype={vals.dtype}).")
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
            report["warnings"].append(f"'{label}' coordinate has duplicate values (examples: {examples}).")
    has_dup_time, _ = _dup_info(time_vals.values.astype("int64"))
    report["time_has_duplicate_timestamps"] = has_dup_time
    if has_dup_time:
        report["warnings"].append("Time coordinate has duplicate timestamps.")

    report["time_ascending"] = bool(len(time_vals) < 2 or np.all(np.diff(time_vals.values.astype("int64")) > 0))
    report["latitude_ascending"] = bool(len(lat_vals) < 2 or lat_vals[0] < lat_vals[-1])
    report["longitude_ascending"] = bool(len(lon_vals) < 2 or lon_vals[0] < lon_vals[-1])
    for label, is_asc in (("latitude", report["latitude_ascending"]), ("longitude", report["longitude_ascending"])):
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
    report["depth_values_m"] = depth_vals.tolist()
    report["first_timestamp"] = str(time_vals[0]) if len(time_vals) else None
    report["last_timestamp"] = str(time_vals[-1]) if len(time_vals) else None

    if len(time_vals) > 1:
        sorted_times = np.sort(time_vals.values)
        diffs_days = np.diff(sorted_times).astype("timedelta64[h]").astype(float) / 24.0
        report["median_time_step_days"] = float(np.median(diffs_days))
        report["is_daily"] = bool(np.isclose(np.median(diffs_days), 1.0, atol=0.05))
        actual_days = pd.DatetimeIndex(sorted_times).normalize()
        expected_days = pd.date_range(actual_days.min(), actual_days.max(), freq="D")
        missing_dates = sorted(set(expected_days) - set(actual_days))
        report["missing_dates"] = [str(d.date()) for d in missing_dates]
        if missing_dates:
            report["warnings"].append(f"{len(missing_dates)} missing date(s) within the time span.")
        if not report["is_daily"]:
            report["warnings"].append(f"Median time step is {report['median_time_step_days']:.2f} days, not ~1 day.")
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


def validate_singleton_depth(ds: xr.Dataset, coords: DetectedCoords4D, config) -> Dict[str, object]:
    """
    Step A/C requirement: the depth coordinate must contain EXACTLY one
    value, and that value must be confirmed as surface (within
    config.surface_depth_tolerance_m of config.expected_surface_depth_m).

    Never arbitrarily selects a depth if multiple are present -- raises
    instead, so a future multi-depth SSS product is never silently
    mishandled by this pipeline (which is built for a singleton-depth
    surface product only).
    """
    depth_vals = np.asarray(ds[coords.depth].values, dtype=float)
    report: Dict[str, object] = {"n_depth": int(len(depth_vals)), "depth_values_m": depth_vals.tolist()}

    if len(depth_vals) != 1:
        raise ValueError(
            f"Expected exactly one depth level (singleton surface dimension), found "
            f"{len(depth_vals)}: {depth_vals.tolist()}. This pipeline is built for a "
            f"single-surface-depth SSS product only -- if the real file legitimately "
            f"has multiple depths, this needs an explicit design decision, not an "
            f"automatic 'pick one' here."
        )

    depth_value = float(depth_vals[0])
    diff = abs(depth_value - config.expected_surface_depth_m)
    report["depth_value_m"] = depth_value
    report["expected_surface_depth_m"] = config.expected_surface_depth_m
    report["difference_from_expected_m"] = diff
    if diff > config.surface_depth_tolerance_m:
        raise ValueError(
            f"Depth value {depth_value} m is not within surface_depth_tolerance_m="
            f"{config.surface_depth_tolerance_m} m of the expected surface depth "
            f"{config.expected_surface_depth_m} m. Refusing to silently treat this as "
            f"surface data -- confirm explicitly (adjust expected_surface_depth_m / "
            f"surface_depth_tolerance_m in config_sss.py) if this is genuinely surface."
        )
    report["confirmed_surface"] = True
    return report


# ---------------------------------------------------------------------
# Region / time subsetting
# ---------------------------------------------------------------------

def subset_region_time(ds: xr.Dataset, coords: DetectedCoords4D, config) -> Tuple[xr.Dataset, Dict[str, object]]:
    """
    Select the configured lat/lon/time window using boolean masks over
    coordinate VALUES (never slice(), which silently misbehaves on a
    descending coordinate).

    For this dataset the native lat/lon range (5.0625-29.9375,
    45.0625-104.9375) is expected to fall ENTIRELY inside the requested
    region (5-30, 45-105), so this is expected to be a near-no-op
    spatially -- still performed and reported explicitly rather than
    skipped, so a future differently-shaped input is handled correctly
    too.
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
            f"lat[{lat_vals.min():.4f},{lat_vals.max():.4f}] "
            f"lon[{lon_vals.min():.4f},{lon_vals.max():.4f}] "
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

    if not (full_lat and full_lon):
        msg = (
            f"Native spatial coverage does not reach the exact requested boundary. Actual lat "
            f"[{actual_lat.min():.4f}, {actual_lat.max():.4f}] vs requested "
            f"[{config.lat_min}, {config.lat_max}]; actual lon "
            f"[{actual_lon.min():.4f}, {actual_lon.max():.4f}] vs requested "
            f"[{config.lon_min}, {config.lon_max}]. EXPECTED for this dataset (native grid is "
            f"~0.0625 deg inside the requested boundary on each side) -- the outermost target "
            f"grid row/column will be NaN after regridding (no extrapolation); see regridding report."
        )
        logger.info(msg)
        report["warnings"].append(msg)
        if not config.allow_partial_spatial_overlap:
            raise ValueError("Partial spatial overlap and allow_partial_spatial_overlap=False. " + msg)
    if not full_time:
        msg = (
            f"Requested time range only partially covered. Actual time "
            f"[{actual_time.min()}, {actual_time.max()}] vs requested [{start}, {end}]."
        )
        logger.warning(msg)
        report["warnings"].append(msg)

    subset = ds.isel({
        coords.latitude: np.where(lat_mask)[0],
        coords.longitude: np.where(lon_mask)[0],
        coords.time: np.where(time_mask)[0],
    })
    return subset, report


def verify_temporal_coverage(coord_report: Dict[str, object], region_report: Dict[str, object], config) -> None:
    """Raise if config.require_complete_temporal_coverage and actual
    time coverage doesn't match what was requested."""
    if not config.require_complete_temporal_coverage:
        return
    problems: List[str] = []
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)
    actual_range = region_report.get("actual_time_range")
    if actual_range is not None:
        actual_start = pd.Timestamp(actual_range[0])
        actual_end = pd.Timestamp(actual_range[1])
        if not (actual_start <= start and actual_end >= end):
            problems.append(
                f"requested time range {config.start_date}..{config.end_date} is not fully "
                f"covered; actual matched range {actual_range}."
            )
    if coord_report.get("missing_dates"):
        problems.append(f"{len(coord_report['missing_dates'])} missing date(s) in the file's own time coordinate.")

    expected = config.expected_time_step_count
    if expected is None:
        expected = (end - start).days + 1
    actual = region_report.get("n_time_matched")
    if actual is not None and actual != expected:
        problems.append(f"expected {expected} daily time step(s) for the requested range, matched {actual}.")

    if problems:
        raise ValueError(
            "require_complete_temporal_coverage=True but temporal coverage is incomplete: "
            + " ".join(problems)
            + " Set require_complete_temporal_coverage=False (--allow-partial-time) if a "
              "partial range is actually intended."
        )


# ---------------------------------------------------------------------
# Missing-value / plausibility inspection
# ---------------------------------------------------------------------

def compute_valid_mask(da) -> xr.DataArray:
    """True where finite (a real, usable SSS value); False for
    NaN/Inf (land, satellite-coverage gap, or otherwise invalid).
    xarray decodes NetCDF _FillValue into NaN by default
    (mask_and_scale=True), so this catches that too. Lazy (dask-backed
    if da is) -- does not force full materialization by itself."""
    return np.isfinite(da)


def inspect_missing_values(da) -> Dict[str, object]:
    """Dask/xarray-aware global missing-value stats; .compute()'s only
    the small scalar result, never the full array via .values."""
    total = int(da.size)
    n_valid = int(np.isfinite(da).sum().compute())
    n_invalid = total - n_valid
    return {
        "total_cells": total,
        "nan_or_inf_count": n_invalid,
        "valid_count": n_valid,
        "fraction_invalid": (n_invalid / total) if total else None,
        "fraction_valid": (n_valid / total) if total else None,
    }


def inspect_missing_by_time(da: xr.DataArray, coords) -> Dict[str, object]:
    """
    Per-time-step missing-value breakdown, plus explicit detection of
    any time step that is COMPLETELY missing. An aggregate percentage
    can hide a single corrupted/blank day; this is a deliberately
    separate reduction pass (not merged into inspect_missing_values'
    .compute()) so as not to risk that already-verified code.
    """
    finite = np.isfinite(da)
    dims_other_than_time = [d for d in da.dims if d != coords.time]
    valid_by_time = finite.sum(dim=dims_other_than_time).compute()

    n_per_time = 1
    for d in dims_other_than_time:
        n_per_time *= da.sizes[d]

    time_vals = da[coords.time].values
    by_time: List[Dict[str, object]] = []
    completely_missing_times: List[str] = []
    values = valid_by_time.values
    for i, tv in enumerate(time_vals):
        valid = int(values[i])
        frac = (valid / n_per_time) if n_per_time else None
        ts = str(pd.Timestamp(tv))
        by_time.append({"time": ts, "valid_count": valid, "valid_fraction": frac})
        if valid == 0:
            completely_missing_times.append(ts)

    fractions = [d["valid_fraction"] for d in by_time if d["valid_fraction"] is not None]
    return {
        "valid_fraction_by_time": by_time,
        "completely_missing_times": completely_missing_times,
        "valid_fraction_constant_across_time": bool(
            len(fractions) > 1 and np.allclose(fractions, fractions[0], atol=1e-9)
        ),
        "valid_fraction_min": float(min(fractions)) if fractions else None,
        "valid_fraction_max": float(max(fractions)) if fractions else None,
    }


def flag_implausible_values(da, low: float, high: float) -> Dict[str, object]:
    """Soft check on FINITE values only -- see config_sss.py's
    implausible_low/high docstring for why these bounds are generous
    (source units are unverified as PSU). Never modifies data."""
    finite_mask = np.isfinite(da)
    finite_only = da.where(finite_mask)
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


def check_implausible_fraction(implausible_report: Dict[str, object], missing_report: Dict[str, object], config) -> None:
    """Fatal backstop behind flag_implausible_values()'s report-only check."""
    finite_count = missing_report.get("valid_count", 0)
    n_implausible = implausible_report.get("n_implausible", 0)
    if not finite_count:
        return
    fraction = n_implausible / finite_count
    if fraction > config.max_implausible_fraction:
        raise ValueError(
            f"{n_implausible}/{finite_count} finite values ({fraction:.1%}) fall outside the "
            f"plausible [{config.implausible_low}, {config.implausible_high}] range, exceeding "
            f"max_implausible_fraction={config.max_implausible_fraction:.1%}. This usually means "
            f"an unmasked fill-value sentinel or the wrong variable was selected -- unsafe to "
            f"proceed. Investigate before raising max_implausible_fraction."
        )


def check_units(da: xr.DataArray) -> Tuple[Optional[str], List[str]]:
    """Report the variable's declared units without assuming or
    relabeling them. Returns (units_string_or_None, warnings)."""
    warnings: List[str] = []
    units = da.attrs.get("units")
    if units is None:
        warnings.append("SSS variable has no 'units' attribute in the file.")
    elif str(units).strip() == ".001":
        warnings.append(
            "SSS variable units attribute is '.001' -- preserved verbatim, NOT auto-relabeled "
            "as PSU or any other unit. See config_sss.py assume_psu_equivalent and README "
            "'Unit handling' for the unresolved interpretation question."
        )
    else:
        warnings.append(
            f"SSS variable units attribute is {units!r} (not the expected '.001') -- verify "
            f"before assuming the documented unit interpretation still applies."
        )
    return units, warnings


# ---------------------------------------------------------------------
# Regridding -- OFFICIAL, default-on. Any failure is FATAL by design.
# ---------------------------------------------------------------------

def build_target_grid(config) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Explicitly construct the official target lat/lon grid from
    config's region bounds and target_resolution_deg -- NEVER derived
    from the source file's own (smaller) native extent.

    Convention: cell-center, BOTH endpoints included.
    n_points = round((max - min) / spacing) + 1, then
    np.linspace(min, max, n_points) -- exact endpoints with no
    floating-point drift (unlike np.arange). For 5-30N / 45-105E at
    0.25 deg this gives exactly 101 x 241 points.
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


def regrid_weighted_linear(
    da: xr.DataArray, coords: DetectedCoords3D, config
) -> Tuple[xr.DataArray, xr.DataArray, Dict[str, object]]:
    """
    Regrid da (time, lat, lon) from the native ~0.125 deg grid onto the
    EXPLICIT official 0.25 deg target grid using NaN-aware, normalized
    weighted bilinear interpolation. Returns (regridded_values,
    valid_mask, report).

    WHY THIS METHOD (not the temperature pipeline's naive
    xarray.interp()): SSS is ~50.6% missing (a spatially-fixed
    satellite-coverage / land pattern, same fraction every single
    timestep -- confirmed in the pre-regrid missing-value report, not
    assumed). A plain bilinear/linear interpolator built directly on an
    array containing NaN would let ANY NaN neighbour in a target cell's
    4-corner stencil either NaN out that cell unpredictably or (for
    some scipy configurations) propagate silently, with no record of
    how much of the interpolated value was actually real support. That
    is exactly the "naive interpolation ... NaN contamination spreads
    unpredictably" failure mode the brief warns against.

    ALGORITHM (mask-weighted / normalized convolution, per Step E):
      1. finite_mask = isfinite(source values)               [per cell]
      2. values_filled = where(finite_mask, values, 0.0)
      3. weighted_numerator = RegularGridInterpolator(values_filled)
                               evaluated at every target (lat, lon)
      4. weight = RegularGridInterpolator(finite_mask.astype(float))
                  evaluated at the same target points
      5. result = weighted_numerator / weight   (elementwise; see
         divide-by-zero handling below)
      6. result is set to NaN wherever weight < config.min_valid_weight
         (default 0.5: a target cell's interpolated value must be
         majority-built from real, finite source observations, not
         from an interpolation call that "returned a number" built
         mostly from filled zeros)
      7. valid_mask = (weight >= min_valid_weight) & isfinite(result)

    NO EXTRAPOLATION: both interpolators are built with
    bounds_error=False, fill_value=0.0. A target point outside the
    native grid's convex hull therefore gets weight=0 from BOTH
    interpolators -> result is NaN via the divide-by-zero guard below,
    exactly like an interior all-missing cell. This is what makes the
    outermost target row/column (see config_sss.py
    allow_edge_nearest_fill docstring) come back NaN without a
    separate bounding-box check -- confirmed by report below via
    n_edge_out_of_hull.

    FATAL BY DESIGN: any failure here (missing scipy, non-monotonic/
    duplicate/non-finite source coordinates, or an all-NaN result) is
    raised and MUST propagate -- no silent fallback to the native grid.
    """
    from scipy.interpolate import RegularGridInterpolator

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
    report["source_latitude_range"] = (float(src_lat.min()), float(src_lat.max()))
    report["source_longitude_range"] = (float(src_lon.min()), float(src_lon.max()))

    new_lat, new_lon, grid_info = build_target_grid(config)
    report["target_grid"] = grid_info
    report["method"] = "weighted_linear (NaN-aware normalized bilinear interpolation)"
    report["min_valid_weight"] = config.min_valid_weight
    report["extrapolation_policy"] = (
        "disabled (bounds_error=False, fill_value=0.0 passed to BOTH the value- and "
        "weight-interpolators) -- a target point outside the native grid's convex hull "
        "receives interpolation weight 0 from both, which the divide-by-zero guard below "
        "converts to NaN/invalid, never a guessed or edge-extended value."
    )
    report["missing_data_handling"] = (
        "source NaN cells are masked to weight=0 (not treated as valid zero-salinity "
        "observations) and excluded from the weighted average; a target cell's value is "
        "set to NaN and marked invalid unless at least min_valid_weight fraction of its "
        "bilinear interpolation stencil came from finite source cells."
    )

    # Target query points (flattened meshgrid), built once and reused
    # for every time batch.
    lon_grid, lat_grid = np.meshgrid(new_lon, new_lat)  # shape (n_lat, n_lon)
    query_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])  # (n_lat*n_lon, 2)
    n_lat_t, n_lon_t = len(new_lat), len(new_lon)

    try:
        n_time = da.sizes[coords.time]
        batch_size = config.regrid_time_batch_size
        result_batches = []
        mask_batches = []
        n_edge_out_of_hull = 0

        for start in range(0, n_time, batch_size):
            end = min(start + batch_size, n_time)
            batch = da.isel({coords.time: slice(start, end)})
            # (n_batch_time, n_src_lat, n_src_lon) -> (n_src_lat, n_src_lon, n_batch_time)
            values = np.asarray(batch.transpose(coords.latitude, coords.longitude, coords.time).values, dtype=np.float64)
            finite = np.isfinite(values)
            values_filled = np.where(finite, values, 0.0)
            weight_source = finite.astype(np.float64)

            interp_vals = RegularGridInterpolator(
                (src_lat, src_lon), values_filled, method="linear", bounds_error=False, fill_value=0.0
            )
            interp_weight = RegularGridInterpolator(
                (src_lat, src_lon), weight_source, method="linear", bounds_error=False, fill_value=0.0
            )

            numer = interp_vals(query_points)   # (n_lat*n_lon, n_batch_time)
            weight = interp_weight(query_points)  # (n_lat*n_lon, n_batch_time)

            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(weight > 0, numer / np.where(weight > 0, weight, 1.0), np.nan)
            valid = (weight >= config.min_valid_weight) & np.isfinite(ratio)
            ratio = np.where(valid, ratio, np.nan)

            n_edge_out_of_hull += int(np.sum(weight == 0.0))

            # (n_lat*n_lon, n_batch_time) -> (n_batch_time, n_lat, n_lon)
            ratio = ratio.reshape(n_lat_t, n_lon_t, -1).transpose(2, 0, 1)
            valid = valid.reshape(n_lat_t, n_lon_t, -1).transpose(2, 0, 1)
            result_batches.append(ratio)
            mask_batches.append(valid)

        result = np.concatenate(result_batches, axis=0) if len(result_batches) > 1 else result_batches[0]
        valid_mask_arr = np.concatenate(mask_batches, axis=0) if len(mask_batches) > 1 else mask_batches[0]
        report["regrid_time_batch_size"] = batch_size
        report["n_regrid_batches"] = len(result_batches)
        report["n_target_points_with_zero_source_support_any_time"] = int(n_edge_out_of_hull)

    except Exception as exc:  # noqa: BLE001 - deliberately broad: ANY regridding failure is fatal, re-raised with context
        raise RuntimeError(
            f"Regridding to the official {config.target_resolution_deg} deg target grid FAILED "
            f"({type(exc).__name__}: {exc}). This is FATAL BY DESIGN -- the official pipeline "
            f"must never silently fall back to the native grid. Common causes: missing scipy "
            f"(pip install scipy), non-monotonic/duplicate source coordinates, or insufficient "
            f"source coverage for the requested region."
        ) from exc

    regridded_da = xr.DataArray(
        result.astype(np.float32),
        dims=(coords.time, coords.latitude, coords.longitude),
        coords={
            coords.time: da[coords.time].values,
            coords.latitude: new_lat,
            coords.longitude: new_lon,
        },
    )
    mask_da = xr.DataArray(
        valid_mask_arr.astype(np.uint8),
        dims=(coords.time, coords.latitude, coords.longitude),
        coords={
            coords.time: da[coords.time].values,
            coords.latitude: new_lat,
            coords.longitude: new_lon,
        },
    )

    n_total = int(regridded_da.size)
    n_valid = int(valid_mask_arr.sum())
    n_invalid = n_total - n_valid
    report["target_cells_total"] = n_total
    report["target_cells_invalid_after_regrid"] = n_invalid
    report["target_cells_invalid_fraction"] = (n_invalid / n_total) if n_total else None
    if n_total and n_valid == 0:
        raise RuntimeError(
            "Regridding produced an ENTIRELY invalid output -- every target-grid cell had "
            "insufficient source support. This almost always means the requested region does "
            "not overlap the source file's real coverage at all, or min_valid_weight is set "
            "impossibly high. Fatal by design, not silently shipped."
        )
    if n_invalid > 0:
        report["warnings"].append(
            f"{n_invalid}/{n_total} ({100.0 * n_invalid / n_total:.2f}%) target-grid cells are "
            f"invalid after regridding (insufficient source support, outside native coverage, "
            f"or already-invalid source cells)."
        )

    return regridded_da, mask_da, report


def verify_final_grid(da: xr.DataArray, coords: DetectedCoords3D, config, tolerance_deg: Optional[float] = None) -> Dict[str, object]:
    """
    Explicit post-regrid validation: calculates actual spacing from the
    resulting coordinates rather than trusting configuration alone.
    Raises ValueError with the specific violation.
    """
    tolerance_deg = config.final_grid_tolerance_deg if tolerance_deg is None else tolerance_deg
    lat = np.asarray(da[coords.latitude].values, dtype=float)
    lon = np.asarray(da[coords.longitude].values, dtype=float)
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
    if len(lat) != 101 or len(lon) != 241:
        raise ValueError(f"Final output grid is {len(lat)} x {len(lon)}, expected exactly 101 x 241.")

    expected_dims = {coords.time, coords.latitude, coords.longitude}
    if set(da.dims) != expected_dims:
        raise ValueError(f"Final output has unexpected dimensions {da.dims}, expected exactly {sorted(expected_dims)}.")

    return report


# ---------------------------------------------------------------------
# Coordinate standardization
# ---------------------------------------------------------------------

def standardize_output_coordinates(
    da: xr.DataArray, mask_da: xr.DataArray, coords: DetectedCoords3D, config
) -> Tuple[xr.DataArray, xr.DataArray, "DetectedCoords3D", Dict[str, str]]:
    """Rename detected source coordinate names to the standardized
    output names in config. Pure relabeling -- .rename() does not
    touch values -- and never modifies the source file."""
    rename_map: Dict[str, str] = {}
    target_names = {
        coords.time: config.output_time_name,
        coords.latitude: config.output_latitude_name,
        coords.longitude: config.output_longitude_name,
    }
    for source_name, output_name in target_names.items():
        if source_name != output_name:
            rename_map[source_name] = output_name

    renamed_da = da.rename(rename_map) if rename_map else da
    renamed_mask = mask_da.rename(rename_map) if rename_map else mask_da
    new_coords = DetectedCoords3D(
        time=config.output_time_name,
        latitude=config.output_latitude_name,
        longitude=config.output_longitude_name,
    )
    return renamed_da, renamed_mask, new_coords, rename_map


# ---------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------

def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Call write_fn(tmp_path) to write to a temp file in the SAME
    directory as `path`, then atomically replace `path` with it."""
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
# Raw-input-file fingerprint (never-modified check)
# ---------------------------------------------------------------------

def fingerprint_file(path: Path) -> Dict[str, object]:
    """Cheap (size + mtime) fingerprint captured before opening and
    compared again after the run, to catch (not guarantee against) any
    accidental modification of the raw input. The primary guarantee is
    architectural (the raw file is only ever opened via
    xr.open_dataset, never in write mode, and no code path writes to
    config.explicit_input_path/default_input_candidate) -- this is a
    best-effort, non-fatal courtesy check on top of that."""
    st = os.stat(path)
    return {"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_file_unchanged(before: Dict[str, object], after: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    if before["size_bytes"] != after["size_bytes"]:
        warnings.append(
            f"RAW INPUT FILE SIZE CHANGED during this run ({before['size_bytes']} -> "
            f"{after['size_bytes']} bytes) -- investigate immediately, this should never happen."
        )
    if before["mtime_ns"] != after["mtime_ns"]:
        warnings.append(
            f"Raw input file modification time changed during this run ({before['mtime_ns']} -> "
            f"{after['mtime_ns']}) -- investigate; this pipeline never opens the raw input in "
            f"write mode."
        )
    return warnings


# ---------------------------------------------------------------------
# Output self-verification (reopen and check)
# ---------------------------------------------------------------------

def verify_output_file(nc_path: Path, config, expected_shape: Tuple[int, int, int]) -> Dict[str, object]:
    """
    Reopen the just-written NetCDF with a FRESH xr.open_dataset call
    and check it actually contains what was intended. Raises ValueError
    with the specific mismatch rather than assuming success just
    because to_netcdf() didn't raise.
    """
    ds = xr.open_dataset(nc_path)
    try:
        report: Dict[str, object] = {"checked_file": str(nc_path), "warnings": []}
        if "sss" not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no 'sss' variable; found {list(ds.data_vars)}.")
        da = ds["sss"]
        expected_dims = (config.output_time_name, config.output_latitude_name, config.output_longitude_name)
        if tuple(da.dims) != expected_dims:
            raise ValueError(f"Reopened dims {da.dims} != expected {expected_dims}.")
        if tuple(da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened shape {da.shape} != expected {expected_shape}.")
        if "sss_valid_mask" not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no 'sss_valid_mask' variable.")
        mask_da = ds["sss_valid_mask"]
        if tuple(mask_da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened sss_valid_mask shape {mask_da.shape} != expected {expected_shape}.")

        # Mask-consistency check: valid_mask==1 must correspond exactly
        # to finite sss values (never "valid but NaN" or vice versa).
        finite = np.isfinite(da.values)
        mask_bool = mask_da.values.astype(bool)
        if not np.array_equal(finite, mask_bool):
            n_mismatch = int(np.sum(finite != mask_bool))
            raise ValueError(
                f"Reopened sss_valid_mask is INCONSISTENT with finite(sss) in {n_mismatch} cell(s) "
                f"-- mask must exactly match np.isfinite(sss)."
            )

        report["reopened_dtype"] = str(da.dtype)
        report["reopened_units"] = da.attrs.get("units")
        report["reopened_mask_dtype"] = str(mask_da.dtype)
        report["reopened_dims"] = list(da.dims)
        report["reopened_shape"] = list(da.shape)
        report["mask_consistent_with_isfinite"] = True

        coords_reopened = DetectedCoords3D(
            time=config.output_time_name, latitude=config.output_latitude_name, longitude=config.output_longitude_name
        )
        grid_report = verify_final_grid(da, coords_reopened, config)
        report["reopened_grid"] = grid_report

        report["ok"] = True
        return report
    finally:
        ds.close()
