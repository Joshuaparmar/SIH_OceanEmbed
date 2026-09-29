"""
netcdf_utils.py (SSS, year-agnostic)
=====================================
Reusable, file-agnostic helpers for inspecting, validating, and
regridding a CNR-style Sea Surface Salinity NetCDF file (variable
'sos' or CF-equivalent), for ANY year's file, not just 2025.

See config_sss.py's module docstring CHANGELOG for what changed in
0.2.0 vs 0.1.0. The regridding algorithm itself (regrid_weighted_linear)
is UNCHANGED from 0.1.0 -- it was already year-agnostic (it operates
purely on whatever (time, lat, lon) array and coordinates it's given).
What's new here: CF-metadata-based variable detection with ambiguity
checking, longitude-convention detection/normalization, descending-
coordinate normalization, year detection, explicit mask-validation
checks, representative-timestep selection for diagnostics, and
{year}-templated output path construction.
"""

from __future__ import annotations

import calendar
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("preprocess_sss")


@dataclass
class DetectedCoords4D:
    time: str
    depth: str
    latitude: str
    longitude: str


@dataclass
class DetectedCoords3D:
    time: str
    latitude: str
    longitude: str


# ---------------------------------------------------------------------
# Input discovery (single-file + batch)
# ---------------------------------------------------------------------

def find_input_file(config) -> Tuple[Optional[Path], str]:
    """Single-file mode: resolve config.explicit_input_path. There is
    no year-specific default any more -- a year-agnostic pipeline
    cannot guess which year's file is wanted."""
    if not config.explicit_input_path:
        return None, "no --input given"
    p = config.resolve(config.explicit_input_path)
    if p.exists():
        return p, f"explicit path (--input): {p}"
    return None, f"explicit path (--input, NOT FOUND): {p}"


def discover_input_files(input_dir: Path, pattern: str) -> List[Path]:
    """Batch mode: find candidate NetCDF files under input_dir, sorted
    for deterministic processing order. Does not open or validate them
    -- that happens per-file in the caller so one bad file doesn't stop
    discovery of the rest."""
    if not input_dir.exists():
        raise FileNotFoundError(f"--input-dir does not exist: {input_dir}")
    files = sorted(input_dir.glob(pattern))
    files = [f for f in files if f.is_file() and f.suffix.lower() == ".nc"]
    return files


def open_dataset(path: Path, use_dask_chunking: bool = True) -> xr.Dataset:
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
# Variable detection -- name candidates AND CF metadata, cross-checked
# ---------------------------------------------------------------------

def detect_variable(ds: xr.Dataset, config) -> Tuple[str, Dict[str, object]]:
    """
    Detect the SSS variable using TWO independent signals:
      1. name candidates (config.sss_var_candidates)
      2. CF metadata: standard_name == config.cf_standard_name, or
         config.cf_long_name_substring found in long_name (case-insensitive)

    Returns (variable_name, report). Raises ValueError if:
      - neither signal finds anything (report the real variable list)
      - the two signals disagree on MULTIPLE different variables
        (ambiguous -- never guessed)
      - either signal alone matches more than one variable (ambiguous)

    This is deliberately stricter than "first name match wins" (0.1.0's
    behavior) specifically so a differently-organized future file
    (e.g. one that also carries a 'so' potential-density-adjacent
    variable, or an unrelated 'salinity_error' variable) cannot be
    silently mis-selected.
    """
    name_matches = [c for c in config.sss_var_candidates if c in ds.data_vars]

    cf_matches: List[str] = []
    for v in ds.data_vars:
        std_name = str(ds[v].attrs.get("standard_name", "")).strip().lower()
        long_name = str(ds[v].attrs.get("long_name", "")).strip().lower()
        if std_name == config.cf_standard_name.lower():
            cf_matches.append(v)
        elif config.cf_long_name_substring.lower() in long_name:
            cf_matches.append(v)
    cf_matches = list(dict.fromkeys(cf_matches))  # de-dup, preserve order

    report: Dict[str, object] = {"name_matches": name_matches, "cf_matches": cf_matches}

    combined = list(dict.fromkeys(name_matches + cf_matches))
    if len(combined) == 0:
        raise ValueError(
            f"No SSS variable detected. Name candidates tried: {config.sss_var_candidates}; "
            f"CF standard_name checked: {config.cf_standard_name!r}; CF long_name substring "
            f"checked: {config.cf_long_name_substring!r}. Actual variables in file: "
            f"{list(ds.data_vars)} with attrs "
            f"{ {v: dict(ds[v].attrs) for v in ds.data_vars} }."
        )
    if len(combined) > 1:
        raise ValueError(
            f"Ambiguous SSS variable detection: name-candidate matches={name_matches}, "
            f"CF-metadata matches={cf_matches} -- combined {combined} refer to more than one "
            f"variable. Refusing to guess which is the real SSS variable. Narrow "
            f"config.sss_var_candidates or pass the exact variable via a future --variable "
            f"override to resolve this explicitly."
        )
    return combined[0], report


def detect_coordinate_name(ds: xr.Dataset, candidates: List[str], logical_name: str) -> str:
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
# Coordinate validation
# ---------------------------------------------------------------------

def validate_coordinates(ds: xr.Dataset, coords: DetectedCoords4D) -> Dict[str, object]:
    report: Dict[str, object] = {"warnings": []}

    time_vals = pd.to_datetime(ds[coords.time].values)
    depth_vals = np.asarray(ds[coords.depth].values, dtype=float)
    lat_vals = np.asarray(ds[coords.latitude].values, dtype=float)
    lon_vals = np.asarray(ds[coords.longitude].values, dtype=float)

    if lat_vals.ndim != 1:
        raise ValueError(f"Latitude coordinate must be 1-D; got ndim={lat_vals.ndim}.")
    if lon_vals.ndim != 1:
        raise ValueError(f"Longitude coordinate must be 1-D; got ndim={lon_vals.ndim}.")

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
            report["warnings"].append(f"'{label}' coordinate is descending.")

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
        if not report["latitude_grid_regular"]:
            report["warnings"].append("Latitude spacing is irregular.")
    else:
        report["latitude_resolution_deg"] = None
        report["latitude_grid_regular"] = None
    if len(lon_vals) > 1:
        lon_diffs = np.diff(np.sort(lon_vals))
        report["longitude_resolution_deg"] = float(np.median(np.abs(lon_diffs)))
        report["longitude_grid_regular"] = bool(np.allclose(lon_diffs, lon_diffs[0], atol=1e-6))
        if not report["longitude_grid_regular"]:
            report["warnings"].append("Longitude spacing is irregular.")
    else:
        report["longitude_resolution_deg"] = None
        report["longitude_grid_regular"] = None

    return report


def validate_singleton_depth(ds: xr.Dataset, coords: DetectedCoords4D, config) -> Dict[str, object]:
    depth_vals = np.asarray(ds[coords.depth].values, dtype=float)
    report: Dict[str, object] = {"n_depth": int(len(depth_vals)), "depth_values_m": depth_vals.tolist()}

    if len(depth_vals) != 1:
        raise ValueError(
            f"Expected exactly one depth level (singleton surface dimension), found "
            f"{len(depth_vals)}: {depth_vals.tolist()}. This pipeline is built for a "
            f"single-surface-depth SSS product only -- refusing to guess which depth "
            f"is 'the' surface."
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
            f"surface data."
        )
    report["confirmed_surface"] = True
    return report


# ---------------------------------------------------------------------
# Coordinate normalization -- descending order, longitude convention
# ---------------------------------------------------------------------

def normalize_coordinate_order(
    ds: xr.Dataset, coords: DetectedCoords4D, config
) -> Tuple[xr.Dataset, Dict[str, object]]:
    """
    Sort latitude/longitude ascending if descending AND
    config.normalize_descending_coordinates is True. If it's False and
    a descending coordinate is found, raises rather than silently
    proceeding with a coordinate order the rest of the pipeline assumes
    is ascending.

    Every transformation actually applied is recorded and returned, per
    the brief's "do not silently alter coordinates without recording
    the transformation in metadata."
    """
    report: Dict[str, object] = {"transformations": []}
    for logical, name in (("latitude", coords.latitude), ("longitude", coords.longitude)):
        vals = ds[name].values
        if len(vals) > 1 and vals[0] > vals[-1]:
            if not config.normalize_descending_coordinates:
                raise ValueError(
                    f"'{name}' ({logical}) coordinate is descending and "
                    f"normalize_descending_coordinates=False. Refusing to proceed with an "
                    f"unsorted coordinate the rest of the pipeline assumes is ascending."
                )
            ds = ds.sortby(name)
            report["transformations"].append(f"sorted '{name}' ascending (was descending)")
            logger.info("Sorted '%s' ascending (was descending).", name)
    return ds, report


def detect_and_normalize_longitude_convention(
    ds: xr.Dataset, coords: DetectedCoords4D, config
) -> Tuple[xr.Dataset, Dict[str, object]]:
    """
    Detect whether the source longitude is 0-360 or -180-180, and shift
    it ONLY IF the requested region (config.lon_min/lon_max, both in
    the 0-360-positive style used by this project, e.g. 45-105) would
    otherwise have zero overlap with the source's actual convention.

    If the source is already compatible (any overlap exists), nothing
    is changed. If normalize_longitude_convention=False and a
    conversion would have been needed, raises rather than silently
    producing a zero-overlap subset later.
    """
    report: Dict[str, object] = {"transformation": "none needed"}
    lon_vals = ds[coords.longitude].values
    lon_min_actual, lon_max_actual = float(lon_vals.min()), float(lon_vals.max())
    source_convention = "-180 to 180" if lon_min_actual < 0 else "0 to 360"
    report["source_convention_detected"] = source_convention

    overlap_exists = not (lon_max_actual < config.lon_min or lon_min_actual > config.lon_max)
    if overlap_exists:
        return ds, report

    if not config.normalize_longitude_convention:
        raise ValueError(
            f"Source longitude range [{lon_min_actual}, {lon_max_actual}] ({source_convention}) "
            f"does not overlap the requested region [{config.lon_min}, {config.lon_max}], and "
            f"normalize_longitude_convention=False -- refusing to silently proceed with zero "
            f"spatial overlap."
        )

    # Requested region is 0-360-style positive (45-105); if source is
    # -180..180 and shifting by +360 for negative values would create
    # overlap, apply that shift (the only conversion direction this
    # project's region requires).
    if source_convention == "-180 to 180":
        shifted = xr.where(ds[coords.longitude] < 0, ds[coords.longitude] + 360, ds[coords.longitude])
        ds = ds.assign_coords({coords.longitude: shifted})
        ds = ds.sortby(coords.longitude)
        report["transformation"] = "shifted longitude from -180..180 to 0..360 to overlap requested region"
        logger.info("Longitude convention shifted: -180..180 -> 0..360 (recorded in metadata).")
    else:
        raise ValueError(
            f"Source longitude [{lon_min_actual}, {lon_max_actual}] (0-360 convention) does not "
            f"overlap requested region [{config.lon_min}, {config.lon_max}], and no supported "
            f"automatic conversion applies. This needs an explicit region/config decision, not "
            f"an automatic guess."
        )
    return ds, report


# ---------------------------------------------------------------------
# Year detection
# ---------------------------------------------------------------------

def detect_year(ds: xr.Dataset, coords: DetectedCoords4D, input_path: Path, config) -> Dict[str, object]:
    """
    Derive the processing year from the DATA (dominant/most-common
    calendar year across the time coordinate's actual timestamps) --
    never from the filename alone, since a filename can be wrong or
    misleading. The filename (if it contains a 4-digit year) is used
    only as an independent CROSS-CHECK, reported as a warning on
    mismatch, never as the source of truth.

    If config.expected_year is explicitly set, it's validated against
    the detected year (mismatch -> warning, not silently overridden).

    Also reports leap-year-aware completeness: whether the file's
    actual time span covers a full calendar year (365 or 366 days,
    correctly using calendar.isleap()) and, if not, that it's a
    partial year -- without rejecting it (see config
    require_complete_temporal_coverage for the opt-in strict mode).

    SAFETY DEFAULT: if the file spans MORE THAN ONE calendar year, this
    raises ValueError by default (config.allow_multi_year=False) rather
    than silently assigning the whole file to its dominant year -- see
    config_sss.py's allow_multi_year docstring. This check happens
    BEFORE any further year-based work, so a mixed-year file is refused
    outright, not partially processed.
    """
    time_vals = pd.to_datetime(ds[coords.time].values)
    if len(time_vals) == 0:
        raise ValueError("Time coordinate is empty; cannot detect a processing year.")

    years, counts = np.unique(time_vals.year, return_counts=True)
    dominant_year = int(years[np.argmax(counts)])
    n_years_present = int(len(years))

    report: Dict[str, object] = {
        "detected_year": dominant_year,
        "n_distinct_years_in_file": n_years_present,
        "years_present": years.tolist(),
        "first_timestamp": str(time_vals.min()),
        "last_timestamp": str(time_vals.max()),
        "n_time_steps": int(len(time_vals)),
        "warnings": [],
    }

    if n_years_present > 1:
        year_counts_str = "\n".join(f"  {int(y)}: {int(c)} timestamp(s)" for y, c in zip(years, counts))
        if not config.allow_multi_year:
            raise ValueError(
                f"Input file contains multiple calendar years (file: {input_path.name}):\n"
                f"{year_counts_str}\n\n"
                f"Dominant year (by timestamp count): {dominant_year}.\n\n"
                f"This generalized SSS pipeline expects one calendar year's data per input "
                f"file. Refusing to assign mixed-year data to a single output year. Split the "
                f"file by year, or pass --allow-multi-year to explicitly opt into assigning it "
                f"to the dominant year ({dominant_year}) -- an advanced, non-standard override "
                f"not used by normal OceanEmbed year-by-year preprocessing."
            )
        report["warnings"].append(
            f"MULTI-YEAR FILE ALLOWED VIA --allow-multi-year: time coordinate spans "
            f"{n_years_present} distinct calendar years {years.tolist()} (counts: "
            f"{dict(zip(years.tolist(), counts.tolist()))}) -- the ENTIRE file is being "
            f"assigned to its dominant year ({dominant_year}) for output naming. This is a "
            f"non-standard, advanced override; the file is NOT split by year."
        )

    is_leap = calendar.isleap(dominant_year)
    expected_days = 366 if is_leap else 365
    report["is_leap_year"] = is_leap
    report["expected_days_in_year"] = expected_days

    year_start = pd.Timestamp(f"{dominant_year}-01-01")
    year_end = pd.Timestamp(f"{dominant_year}-12-31")
    full_year_calendar_days = pd.date_range(year_start, year_end, freq="D")
    present_days = pd.DatetimeIndex(time_vals).normalize().unique()
    present_days_in_year = present_days[(present_days >= year_start) & (present_days <= year_end)]
    missing_days_in_year = sorted(set(full_year_calendar_days) - set(present_days_in_year))

    n_present_in_year = len(present_days_in_year)
    report["n_days_present_in_detected_year"] = int(n_present_in_year)
    report["n_missing_days_in_detected_year"] = int(len(missing_days_in_year))
    report["missing_days_in_detected_year_sample"] = [str(d.date()) for d in missing_days_in_year[:20]]

    if n_present_in_year == expected_days and len(missing_days_in_year) == 0:
        report["coverage_status"] = "complete"
    elif n_present_in_year == 0:
        report["coverage_status"] = "empty"
    else:
        report["coverage_status"] = "partial"

    # Filename cross-check -- informational only, never authoritative.
    m = re.search(r"(19|20)\d{2}", input_path.stem)
    filename_year = int(m.group(0)) if m else None
    report["filename_year"] = filename_year
    if filename_year is not None and filename_year != dominant_year:
        report["warnings"].append(
            f"Filename suggests year {filename_year} but the time coordinate's dominant year is "
            f"{dominant_year} -- using the DATA (dominant year), not the filename. Investigate if "
            f"this is unexpected."
        )

    if config.expected_year is not None and config.expected_year != dominant_year:
        report["warnings"].append(
            f"--year override ({config.expected_year}) does not match the data-detected dominant "
            f"year ({dominant_year}). Using the override ({config.expected_year}) for output "
            f"naming since it was explicitly given, but this mismatch is unusual -- verify the "
            f"correct file was passed."
        )

    report["year_used_for_output"] = config.expected_year if config.expected_year is not None else dominant_year
    return report


def verify_temporal_coverage(year_report: Dict[str, object], config) -> None:
    if not config.require_complete_temporal_coverage:
        return
    if year_report["coverage_status"] != "complete":
        raise ValueError(
            f"require_complete_temporal_coverage=True but detected year "
            f"{year_report['detected_year']} coverage status is "
            f"'{year_report['coverage_status']}' ({year_report['n_days_present_in_detected_year']}/"
            f"{year_report['expected_days_in_year']} days present, "
            f"{year_report['n_missing_days_in_detected_year']} missing). Omit --require-complete-year "
            f"(the default already allows a partial year) if a partial year is intended."
        )


# ---------------------------------------------------------------------
# Region subsetting (time is no longer restricted to a requested
# window -- year-agnostic pipeline preserves whatever dates are
# actually present)
# ---------------------------------------------------------------------

def subset_region(ds: xr.Dataset, coords: DetectedCoords4D, config) -> Tuple[xr.Dataset, Dict[str, object]]:
    """Select the configured lat/lon window using boolean masks over
    coordinate VALUES. Time is NOT subset here -- the actual available
    dates in the file are preserved as-is (see module docstring)."""
    lat_vals = ds[coords.latitude].values
    lon_vals = ds[coords.longitude].values

    lat_mask = (lat_vals >= config.lat_min) & (lat_vals <= config.lat_max)
    lon_mask = (lon_vals >= config.lon_min) & (lon_vals <= config.lon_max)

    report: Dict[str, object] = {
        "warnings": [],
        "requested_lat_range": (config.lat_min, config.lat_max),
        "requested_lon_range": (config.lon_min, config.lon_max),
        "n_lat_matched": int(lat_mask.sum()),
        "n_lon_matched": int(lon_mask.sum()),
    }

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        raise ValueError(
            f"Requested region has ZERO spatial overlap with the file. lat matched="
            f"{lat_mask.sum()}, lon matched={lon_mask.sum()}. File covers "
            f"lat[{lat_vals.min():.4f},{lat_vals.max():.4f}] "
            f"lon[{lon_vals.min():.4f},{lon_vals.max():.4f}]."
        )

    actual_lat = lat_vals[lat_mask]
    actual_lon = lon_vals[lon_mask]
    full_lat = actual_lat.min() <= config.lat_min + 1e-6 and actual_lat.max() >= config.lat_max - 1e-6
    full_lon = actual_lon.min() <= config.lon_min + 1e-6 and actual_lon.max() >= config.lon_max - 1e-6

    report["actual_lat_range"] = (float(actual_lat.min()), float(actual_lat.max()))
    report["actual_lon_range"] = (float(actual_lon.min()), float(actual_lon.max()))
    report["full_region_coverage"] = bool(full_lat and full_lon)

    if not (full_lat and full_lon):
        msg = (
            f"Native spatial coverage does not reach the exact requested boundary. Actual lat "
            f"[{actual_lat.min():.4f}, {actual_lat.max():.4f}] vs requested "
            f"[{config.lat_min}, {config.lat_max}]; actual lon "
            f"[{actual_lon.min():.4f}, {actual_lon.max():.4f}] vs requested "
            f"[{config.lon_min}, {config.lon_max}]. The outermost target grid row/column at the "
            f"boundary will be marked INVALID after regridding because the requested target "
            f"coordinate lies outside the native source coordinate range -- this reflects "
            f"unsupported source coverage, not necessarily land or physically missing ocean data."
        )
        logger.info(msg)
        report["warnings"].append(msg)
        if not config.allow_partial_spatial_overlap:
            raise ValueError("Partial spatial overlap and allow_partial_spatial_overlap=False. " + msg)

    subset = ds.isel({
        coords.latitude: np.where(lat_mask)[0],
        coords.longitude: np.where(lon_mask)[0],
    })
    return subset, report


# ---------------------------------------------------------------------
# Missing-value / plausibility inspection (unchanged from 0.1.0)
# ---------------------------------------------------------------------

def inspect_missing_values(da) -> Dict[str, object]:
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
    finite_count = missing_report.get("valid_count", 0)
    n_implausible = implausible_report.get("n_implausible", 0)
    if not finite_count:
        return
    fraction = n_implausible / finite_count
    if fraction > config.max_implausible_fraction:
        raise ValueError(
            f"{n_implausible}/{finite_count} finite values ({fraction:.1%}) fall outside the "
            f"plausible [{config.implausible_low}, {config.implausible_high}] range, exceeding "
            f"max_implausible_fraction={config.max_implausible_fraction:.1%}."
        )


def check_units(da: xr.DataArray, config) -> Dict[str, object]:
    """Report units and build the explicit unit-interpretation metadata
    block required by the generalization brief -- never modifies
    numerical values, and 'user_confirmed' is only ever set from an
    EXPLICIT config.assume_psu_equivalent=True, never inferred."""
    units = da.attrs.get("units")
    warnings: List[str] = []
    if units is None:
        warnings.append("SSS variable has no 'units' attribute in the file.")
    elif str(units).strip() == ".001":
        warnings.append(
            "SSS variable units attribute is '.001' -- unusual, preserved verbatim, NOT "
            "automatically converted (no x1000/÷1000) or relabeled as PSU."
        )
    else:
        warnings.append(f"SSS variable units attribute is {units!r} (not the expected '.001').")

    unit_block: Dict[str, object] = {
        "source_units": units,
        "unit_conversion": "none",
        "unit_documentation_reference": config.unit_documentation_reference,
        "unit_interpretation_notes": config.unit_interpretation_notes,
    }
    if config.assume_psu_equivalent:
        unit_block["interpreted_units"] = "PSU-equivalent"
        unit_block["unit_interpretation_status"] = "user_confirmed"
    else:
        unit_block["interpreted_units"] = None
        unit_block["unit_interpretation_status"] = "pending_verification"

    return {"warnings": warnings, "unit_block": unit_block}


# ---------------------------------------------------------------------
# Regridding -- algorithm UNCHANGED from 0.1.0 (already year-agnostic)
# ---------------------------------------------------------------------

def build_target_grid(config) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
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
    """NaN-aware normalized weighted bilinear interpolation onto the
    explicit official target grid. See config_sss.py min_valid_weight
    docstring for what the support-weight fraction actually means.
    Unchanged algorithm from 0.1.0 -- operates purely on the given
    array/coordinates, with no year-specific assumptions."""
    from scipy.interpolate import RegularGridInterpolator

    report: Dict[str, object] = {"warnings": []}

    lat_vals = da[coords.latitude].values
    lon_vals = da[coords.longitude].values
    for label, vals in ((coords.latitude, lat_vals), (coords.longitude, lon_vals)):
        if not np.issubdtype(vals.dtype, np.number):
            raise ValueError(f"Regridding requires a numeric '{label}' coordinate; got dtype {vals.dtype}.")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Regridding requires finite '{label}' coordinate values.")
        if len(np.unique(vals)) != len(vals):
            raise ValueError(f"Regridding requires unique '{label}' coordinate values; duplicates found.")
        if len(vals) > 1 and not np.all(np.diff(vals) > 0):
            raise ValueError(
                f"Regridding requires an ascending '{label}' coordinate at this stage "
                f"(normalize_coordinate_order should have handled this already)."
            )

    src_lat = lat_vals
    src_lon = lon_vals
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
        "receives interpolation weight 0 from both, converted to NaN/invalid via the "
        "divide-by-zero guard, never a guessed or edge-extended value. Cells invalid for this "
        "reason reflect unsupported source coordinate coverage, not necessarily land."
    )
    report["missing_data_handling"] = (
        "source NaN cells are masked to weight=0 (not treated as valid zero-salinity "
        "observations) and excluded from the weighted average; a target cell's value is set "
        "to NaN and marked invalid unless at least min_valid_weight fraction of its bilinear "
        "interpolation stencil's WEIGHT (not pixel count) came from finite source cells."
    )

    lon_grid, lat_grid = np.meshgrid(new_lon, new_lat)
    query_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
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

            numer = interp_vals(query_points)
            weight = interp_weight(query_points)

            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(weight > 0, numer / np.where(weight > 0, weight, 1.0), np.nan)
            valid = (weight >= config.min_valid_weight) & np.isfinite(ratio)
            ratio = np.where(valid, ratio, np.nan)

            n_edge_out_of_hull += int(np.sum(weight == 0.0))

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
            f"({type(exc).__name__}: {exc}). Fatal by design -- never falls back to the native "
            f"grid. Common causes: missing scipy, non-monotonic/duplicate source coordinates, "
            f"or insufficient source coverage for the requested region."
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
            "insufficient source support. Fatal by design, not silently shipped."
        )
    if n_invalid > 0:
        report["warnings"].append(
            f"{n_invalid}/{n_total} ({100.0 * n_invalid / n_total:.2f}%) target-grid cells are "
            f"invalid after regridding."
        )

    return regridded_da, mask_da, report


def verify_final_grid(da: xr.DataArray, coords: DetectedCoords3D, config, tolerance_deg: Optional[float] = None) -> Dict[str, object]:
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
        raise ValueError(f"Final output latitude spacing {lat_spacing} != required {config.target_resolution_deg}.")
    if lon_spacing is None or abs(lon_spacing - config.target_resolution_deg) > tolerance_deg:
        raise ValueError(f"Final output longitude spacing {lon_spacing} != required {config.target_resolution_deg}.")
    if abs(lat.min() - config.lat_min) > tolerance_deg or abs(lat.max() - config.lat_max) > tolerance_deg:
        raise ValueError(f"Final output latitude range [{lat.min()},{lat.max()}] != requested [{config.lat_min},{config.lat_max}].")
    if abs(lon.min() - config.lon_min) > tolerance_deg or abs(lon.max() - config.lon_max) > tolerance_deg:
        raise ValueError(f"Final output longitude range [{lon.min()},{lon.max()}] != requested [{config.lon_min},{config.lon_max}].")

    expected_n_lat = round((config.lat_max - config.lat_min) / config.target_resolution_deg) + 1
    expected_n_lon = round((config.lon_max - config.lon_min) / config.target_resolution_deg) + 1
    if len(lat) != expected_n_lat or len(lon) != expected_n_lon:
        raise ValueError(f"Final output grid is {len(lat)} x {len(lon)}, expected exactly {expected_n_lat} x {expected_n_lon}.")

    expected_dims = {coords.time, coords.latitude, coords.longitude}
    if set(da.dims) != expected_dims:
        raise ValueError(f"Final output has unexpected dimensions {da.dims}, expected exactly {sorted(expected_dims)}.")

    return report


# ---------------------------------------------------------------------
# Explicit mask-consistency validation (both post-regrid and post-write)
# ---------------------------------------------------------------------

def validate_mask_consistency(sss: np.ndarray, mask: np.ndarray) -> None:
    """Exactly the three checks required by the generalization brief.
    Raises immediately on the first violated invariant."""
    if np.any((mask == 1) & ~np.isfinite(sss)):
        raise ValueError("Mask marks NaN values as valid.")
    if np.any((mask == 0) & np.isfinite(sss)):
        raise ValueError("Mask marks finite values as invalid.")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("Validity mask contains values other than 0 and 1.")


# ---------------------------------------------------------------------
# Coordinate standardization
# ---------------------------------------------------------------------

def standardize_output_coordinates(
    da: xr.DataArray, mask_da: xr.DataArray, coords: DetectedCoords3D, config
) -> Tuple[xr.DataArray, xr.DataArray, "DetectedCoords3D", Dict[str, str]]:
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
# Representative-timestep selection for diagnostics
# ---------------------------------------------------------------------

def select_representative_timestep(mask_da: xr.DataArray, coords: DetectedCoords3D) -> Tuple[int, Dict[str, object]]:
    """Pick a timestep for the diagnostic plot that actually HAS valid
    data, rather than blindly using index 0 (which could be entirely
    missing). Selects the index with the MAXIMUM valid-cell count;
    documents which index and why."""
    valid_counts = mask_da.sum(dim=[coords.latitude, coords.longitude]).values
    if len(valid_counts) == 0:
        raise ValueError("No timesteps available to select a representative timestep from.")
    best_idx = int(np.argmax(valid_counts))
    report = {
        "selected_time_index": best_idx,
        "selected_time_value": str(pd.Timestamp(mask_da[coords.time].values[best_idx])),
        "valid_cell_count_at_selected_index": int(valid_counts[best_idx]),
        "selection_method": "argmax of per-timestep valid-cell count (not necessarily index 0)",
    }
    if valid_counts[0] == 0 and best_idx != 0:
        report["note"] = "Timestep 0 was completely invalid; a different timestep was selected instead."
    return best_idx, report


# ---------------------------------------------------------------------
# Output path construction ({year} templating)
# ---------------------------------------------------------------------

def build_output_paths(config, year: int) -> Dict[str, Path]:
    if config.explicit_output_dir:
        output_dir = config.resolve(config.explicit_output_dir)
    else:
        output_dir = config.resolve(config.output_dir_template.format(year=year))
    return {
        "output_dir": output_dir,
        "nc_path": output_dir / config.output_nc_name_template.format(year=year),
        "npy_path": output_dir / config.output_npy_name_template.format(year=year),
        "mask_npy_path": output_dir / config.output_mask_npy_name_template.format(year=year),
        "metadata_path": output_dir / config.output_metadata_name_template.format(year=year),
        "dryrun_report_path": output_dir / config.dryrun_report_name_template.format(year=year),
        "diagnostic_plot_path": output_dir / config.diagnostic_plot_name_template.format(year=year),
    }


# ---------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------

def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
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
# Raw-input-file fingerprint
# ---------------------------------------------------------------------

def fingerprint_file(path: Path) -> Dict[str, object]:
    st = os.stat(path)
    return {"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_file_unchanged(before: Dict[str, object], after: Dict[str, object]) -> List[str]:
    warnings: List[str] = []
    if before["size_bytes"] != after["size_bytes"]:
        warnings.append(
            f"RAW INPUT FILE SIZE CHANGED during this run ({before['size_bytes']} -> "
            f"{after['size_bytes']} bytes) -- investigate immediately."
        )
    if before["mtime_ns"] != after["mtime_ns"]:
        warnings.append(
            f"Raw input file modification time changed during this run -- investigate."
        )
    return warnings


# ---------------------------------------------------------------------
# Output self-verification (reopen and check)
# ---------------------------------------------------------------------

def verify_output_file(nc_path: Path, config, expected_shape: Tuple[int, int, int]) -> Dict[str, object]:
    ds = xr.open_dataset(nc_path)
    try:
        report: Dict[str, object] = {"checked_file": str(nc_path), "warnings": []}
        var_name = config.output_variable_name
        mask_name = f"{config.output_variable_name}_valid_mask"
        if var_name not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no '{var_name}' variable; found {list(ds.data_vars)}.")
        da = ds[var_name]
        expected_dims = (config.output_time_name, config.output_latitude_name, config.output_longitude_name)
        if tuple(da.dims) != expected_dims:
            raise ValueError(f"Reopened dims {da.dims} != expected {expected_dims}.")
        if tuple(da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened shape {da.shape} != expected {expected_shape}.")
        if mask_name not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no '{mask_name}' variable.")
        mask_da = ds[mask_name]
        if tuple(mask_da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened {mask_name} shape {mask_da.shape} != expected {expected_shape}.")

        sss_vals = da.values
        mask_vals = mask_da.values
        validate_mask_consistency(sss_vals, mask_vals)  # raises on any of the 3 invariants

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
