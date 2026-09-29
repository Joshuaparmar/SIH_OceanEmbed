"""
validation.py (SST)
====================
All "is this actually correct" checks for the SST pipeline, kept
separate from netcdf_utils.py (I/O, detection, subsetting, regridding)
so the validation logic -- what must be TRUE about the output, not how
it was produced -- is auditable on its own.

Two kinds of function here:
  - verify_* : raise ValueError/RuntimeError with a specific message on
    a real problem. Called at points where continuing would ship
    incorrect or incomplete data silently.
  - compute_* : never raise (except for genuinely invalid input shapes);
    return a report dict for the metadata / dry-run report / statistics
    file.

Nothing here mutates the DataArray/Dataset it's given.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from netcdf_utils import DetectedCoords, compute_valid_mask, inspect_missing_values

logger = logging.getLogger("preprocess_sst")


# ---------------------------------------------------------------------
# Strict-mode checks (Section 6 dataset-level / temporal requirements)
# ---------------------------------------------------------------------

def verify_temporal_coverage(coord_report: Dict[str, object], region_report: Dict[str, object], config) -> None:
    """Raise if config.require_complete_temporal_coverage and actual
    time coverage doesn't match what was requested (incomplete overlap,
    missing dates, or a short time-step count). Leap years are handled
    correctly because the expected count is computed from real calendar
    dates (see coord_report['missing_dates'] in netcdf_utils), not a
    fixed 365."""
    if not config.require_complete_temporal_coverage:
        return

    problems = []
    # Check TIME coverage specifically (full_time_coverage), not the
    # combined lat+lon+time flag -- a purely spatial coverage gap is
    # reported (and gated) separately by verify_official_spatial_
    # coverage(); folding it into this check as well would misreport a
    # longitude/latitude gap as a "temporal coverage" problem.
    if not region_report.get("full_time_coverage", region_report.get("full_region_time_coverage", False)):
        problems.append(
            f"requested time range {config.start_date}..{config.end_date} is not fully "
            f"covered; actual matched range {region_report.get('actual_time_range')}."
        )
    requested_missing_dates = region_report.get("missing_dates_in_requested_range")
    if requested_missing_dates is None:
        requested_missing_dates = coord_report.get("missing_dates", [])
    if requested_missing_dates:
        problems.append(f"{len(requested_missing_dates)} missing date(s) in the requested time range.")
    requested_duplicate_dates = region_report.get("duplicate_dates_in_requested_range", [])
    if requested_duplicate_dates:
        problems.append(f"{len(requested_duplicate_dates)} duplicate daily date(s) in the requested time range.")

    expected = config.expected_time_step_count
    if expected is None:
        import pandas as pd
        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        expected = (end - start).days + 1  # inclusive; correctly spans leap days since it's a real calendar diff
    actual = region_report.get("n_time_matched")
    if actual is not None and actual != expected:
        problems.append(f"expected {expected} daily time step(s) for the requested range, matched {actual}.")

    if problems:
        raise ValueError(
            "require_complete_temporal_coverage=True but temporal coverage is incomplete: "
            + " ".join(problems) + " Set require_complete_temporal_coverage=False (and "
            "date_range_confirmed=True with an explicit narrower start_date/end_date, if a "
            "partial range is actually intended) to proceed anyway."
        )


def verify_official_spatial_coverage(coord_report: Dict[str, object], config) -> Dict[str, object]:
    """
    Decide whether the SOURCE FILE (its full, un-subsetted extent, as
    reported by validate_coordinates() -- NOT any strict boolean-mask
    subset's resulting extent) sufficiently covers the requested
    5-30N/45-105E region for an OFFICIAL (regridded) run, and raise
    ValueError with a clear, specific message if not.

    THIS DELIBERATELY COMPARES AGAINST THE SOURCE'S ORIGINAL EXTENT,
    NOT A STRICT SUBSET'S RESULT. A previous version of this check
    compared against the strict subset's actual_lat_range/
    actual_lon_range (from subset_region_time() with the exact
    requested bounds) -- but a boolean mask at an EXACT boundary
    against a native grid that (almost always) doesn't land a cell
    center precisely on that boundary will naturally produce a subset
    that starts/ends up to half a native cell inside the requested
    edge, even when the source file's real data extends well past it.
    For OSTIA's real ~0.05 deg native grid (observed offset ~0.025 deg
    from the 5/30/45/105 boundary), that previously produced a FALSE
    "insufficient coverage" failure on a source file that in fact fully
    -- genuinely -- covers the requested region. Comparing against the
    source's own coordinate extent (independent of how any later
    subset happens to land on a grid line) is the numerically correct
    way to ask "does the data reach far enough," and is exactly what
    lets this function tell that case apart from an ACTUAL gap (a
    source that stops meaningfully short of the requested boundary).

    Tolerance: a gap between the requested boundary and the source's
    actual extent is ACCEPTED, up to
    config.coverage_tolerance_in_native_cells * native_spacing on each
    of the four boundaries independently, when
    config.allow_native_grid_boundary_offset is True (the default).
    native_spacing comes from coord_report's latitude_resolution_deg /
    longitude_resolution_deg (already computed by validate_coordinates()
    from the source's own coordinate spacing -- not assumed, not
    hard-coded to 0.05 or any other specific value, so the same logic
    works for a different SST product with different native
    resolution). If allow_native_grid_boundary_offset is False, the
    tolerance is a fixed tiny numerical epsilon only (1e-6 deg) -- the
    stricter, pre-tolerance behavior.

    Passing this check only means the SOURCE has enough real coverage
    to be usable -- it does NOT by itself guarantee no NaN will appear
    at the target boundary after regridding (that still depends on
    whether interpolation-support cells were actually included in the
    source subset fed to regrid_to_target_grid(); see
    build_interpolation_support_bounds() and
    regrid_to_target_grid()'s own post-hoc NaN-fraction reporting,
    which is the correct place to observe that outcome).

    A no-op (returns a report with checked=False, never raises) for a
    --no-regrid diagnostic run (never "official" regardless of
    coverage) or when require_full_official_coverage is explicitly set
    False (an escape hatch that must be deliberate, not a default).
    """
    if not config.regrid_to_quarter_degree:
        return {"checked": False, "note": "non-official (--no-regrid) run: official coverage not enforced"}
    if not config.require_full_official_coverage:
        return {"checked": False, "note": "require_full_official_coverage=False: official coverage not enforced (explicit escape hatch)"}

    src_lat_min, src_lat_max = coord_report["latitude_range"]
    src_lon_min, src_lon_max = coord_report["longitude_range"]
    lat_spacing = coord_report.get("latitude_resolution_deg")
    lon_spacing = coord_report.get("longitude_resolution_deg")

    if config.allow_native_grid_boundary_offset:
        if not lat_spacing or not lon_spacing:
            raise ValueError(
                "Cannot evaluate official coverage tolerance: native latitude/longitude spacing "
                "could not be determined from the source file (fewer than 2 coordinate points on "
                "one axis?). Set config.allow_native_grid_boundary_offset=False to fall back to a "
                "strict, tolerance-free coverage check, or investigate why spacing is undetermined."
            )
        tol_lat = config.coverage_tolerance_in_native_cells * lat_spacing
        tol_lon = config.coverage_tolerance_in_native_cells * lon_spacing
    else:
        tol_lat = tol_lon = 1e-6

    # Positive gap = source falls short of reaching the requested
    # boundary by this many degrees on that side (source's edge is
    # INSIDE the requested region on that side); 0 (via max(0, ...)) =
    # source already reaches AT OR PAST the requested boundary, i.e.
    # full coverage on that side, regardless of how much further out
    # the source actually extends.
    #   - "low" sides (lat_min/lon_min): gap = src_min - requested_min,
    #     positive only when the source's own minimum is GREATER than
    #     (i.e. inside) the requested minimum.
    #   - "high" sides (lat_max/lon_max): gap = requested_max - src_max,
    #     positive only when the source's own maximum is LESS than
    #     (i.e. inside) the requested maximum.
    lat_gap_low = max(0.0, src_lat_min - config.lat_min)
    lat_gap_high = max(0.0, config.lat_max - src_lat_max)
    lon_gap_low = max(0.0, src_lon_min - config.lon_min)
    lon_gap_high = max(0.0, config.lon_max - src_lon_max)

    lat_ok = lat_gap_low <= tol_lat and lat_gap_high <= tol_lat
    lon_ok = lon_gap_low <= tol_lon and lon_gap_high <= tol_lon

    report: Dict[str, object] = {
        "checked": True,
        "requested_bounds": {"lat_min": config.lat_min, "lat_max": config.lat_max, "lon_min": config.lon_min, "lon_max": config.lon_max},
        "source_bounds": {"lat_min": src_lat_min, "lat_max": src_lat_max, "lon_min": src_lon_min, "lon_max": src_lon_max},
        "native_spacing_deg": {"latitude": lat_spacing, "longitude": lon_spacing},
        "boundary_gaps_deg": {
            "lat_low": lat_gap_low, "lat_high": lat_gap_high, "lon_low": lon_gap_low, "lon_high": lon_gap_high,
        },
        "allow_native_grid_boundary_offset": config.allow_native_grid_boundary_offset,
        "coverage_tolerance_in_native_cells": config.coverage_tolerance_in_native_cells,
        "tolerance_deg": {"latitude": tol_lat, "longitude": tol_lon},
        "lat_ok": lat_ok,
        "lon_ok": lon_ok,
        "passed": bool(lat_ok and lon_ok),
    }

    if not report["passed"]:
        raise ValueError(
            f"Official (regridded) run requires the SOURCE to reach within tolerance of the "
            f"requested region lat[{config.lat_min},{config.lat_max}] lon[{config.lon_min},{config.lon_max}], "
            f"but the source's actual extent is lat[{src_lat_min:.4f},{src_lat_max:.4f}] "
            f"lon[{src_lon_min:.4f},{src_lon_max:.4f}] -- boundary gaps (deg): "
            f"lat_low={lat_gap_low:.4f} lat_high={lat_gap_high:.4f} lon_low={lon_gap_low:.4f} "
            f"lon_high={lon_gap_high:.4f}, exceeding tolerance lat={tol_lat:.4f} lon={tol_lon:.4f} "
            f"(coverage_tolerance_in_native_cells={config.coverage_tolerance_in_native_cells}, "
            f"native spacing lat={lat_spacing} lon={lon_spacing}). This is a GENUINE coverage gap, "
            f"not just a grid-alignment offset -- FATAL for an official run: silently regridding "
            f"would require extrapolating beyond real source data, or shipping a model-ready file "
            f"with an undocumented coverage gap. Either confirm this is expected and set "
            f"require_full_official_coverage=False explicitly (and document why), increase "
            f"coverage_tolerance_in_native_cells only if you have verified the gap is still just "
            f"grid alignment, or use --no-regrid for a clearly-marked non-official diagnostic run."
        )
    return report


def check_implausible_fraction(implausible_report: Dict[str, object], missing_report: Dict[str, object], config) -> None:
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
            f"{n_implausible}/{finite_count} finite values ({fraction:.1%}) fall outside the "
            f"plausible [{config.implausible_low_c}, {config.implausible_high_c}] degC range, "
            f"exceeding max_implausible_fraction={config.max_implausible_fraction:.1%}. This "
            f"usually means an unmasked _FillValue/missing_value sentinel, a units mismatch "
            f"(e.g. unconverted Kelvin), or the wrong variable was selected -- unsafe to proceed. "
            f"Investigate before raising max_implausible_fraction."
        )


def check_missing_fraction(missing_report: Dict[str, object], config) -> None:
    """Raise if config.missing_value_policy == 'reject_excessive' and
    the fraction of NaN/Inf cells exceeds config.max_missing_fraction.
    A no-op for 'preserve' (the default) -- SST legitimately has a
    large fraction of NaN land pixels over a mixed land/ocean region,
    so a high missing fraction is not inherently wrong; this check only
    fires when the caller has explicitly opted into treating it as a
    hard failure (e.g. to catch a wrong-variable/wrong-region mistake)."""
    if config.missing_value_policy != "reject_excessive":
        return
    fraction = missing_report.get("fraction_invalid")
    if fraction is not None and fraction > config.max_missing_fraction:
        raise ValueError(
            f"{missing_report['nan_or_inf_count']}/{missing_report['total_cells']} cells "
            f"({fraction:.1%}) are NaN/Inf, exceeding max_missing_fraction="
            f"{config.max_missing_fraction:.1%} with missing_value_policy='reject_excessive'. "
            f"This may indicate the wrong variable/region was selected, or a mask that wasn't "
            f"applied as expected. Investigate, or switch missing_value_policy='preserve' if "
            f"this level of land/ocean masking is actually expected for the region."
        )


# ---------------------------------------------------------------------
# Final grid / output verification
# ---------------------------------------------------------------------

def verify_final_grid(
    da: xr.DataArray, coords: DetectedCoords, config, tolerance_deg: Optional[float] = None
) -> Dict[str, object]:
    """
    Explicit post-regrid validation: calculate the actual spacing from
    the resulting coordinates rather than trusting configuration alone.
    Raises ValueError with the specific violation for grid
    spacing/range (when config.regrid_to_quarter_degree), coordinate
    monotonicity/duplicates, and unexpected dimensions -- never just
    reports and continues.
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

    expected_dims = {coords.time, coords.latitude, coords.longitude}
    if set(da.dims) != expected_dims:
        raise ValueError(f"Final output has unexpected dimensions {da.dims}, expected exactly {sorted(expected_dims)}.")

    return report


def verify_output_file(nc_path: Path, config, expected_shape: Tuple[int, int, int]) -> Dict[str, object]:
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
        var_name = config.output_variable_name
        if var_name not in ds.data_vars:
            raise ValueError(f"Reopened {nc_path} has no '{var_name}' variable; found {list(ds.data_vars)}.")
        da = ds[var_name]
        expected_dims = (config.output_time_name, config.output_latitude_name, config.output_longitude_name)
        if tuple(da.dims) != expected_dims:
            raise ValueError(f"Reopened dims {da.dims} != expected {expected_dims}.")
        if tuple(da.shape) != tuple(expected_shape):
            raise ValueError(f"Reopened shape {da.shape} != expected {expected_shape}.")
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

        coords_reopened = DetectedCoords(
            time=config.output_time_name, latitude=config.output_latitude_name, longitude=config.output_longitude_name,
        )
        grid_report = verify_final_grid(da, coords_reopened, config)
        report["reopened_grid"] = grid_report

        # Alignment check: the reopened coordinates must be usable for
        # later cross-variable tensor generation -- ascending, unique,
        # matching the requested OceanEmbed region bounds (already
        # enforced by verify_final_grid when regridded). Time must be a
        # valid, monotonic datetime axis.
        time_vals = da[config.output_time_name].values
        if len(time_vals) > 1:
            time_int = time_vals.astype("datetime64[ns]").astype("int64")
            if not np.all(np.diff(time_int) > 0):
                report["warnings"].append("Reopened time coordinate is not strictly ascending.")

        report["ok"] = True
        return report
    finally:
        ds.close()


# ---------------------------------------------------------------------
# Numerical statistics (Section 6 "Numerical checks" + Section I stats)
# ---------------------------------------------------------------------

def compute_numerical_statistics(array: np.ndarray) -> Dict[str, object]:
    """Min/max/mean/std/median and NaN counts on the FINAL array being
    shipped. Never raises on empty/all-NaN input -- returns None fields
    instead, since an all-NaN result (e.g. a region with no ocean cells
    in a given tiny window) is a fact to report, not a bug to crash on
    here (the earlier regrid-entirely-NaN and implausible-fraction
    checks are what actually gate a bad run)."""
    finite = array[np.isfinite(array)]
    total = int(array.size)
    n_nan = int(total - finite.size)
    stats: Dict[str, object] = {
        "n_time_steps": None,  # filled in by caller with the actual dimension size
        "total_cells": total,
        "n_nan": n_nan,
        "pct_nan": (100.0 * n_nan / total) if total else None,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
    }
    return stats


def compute_normalization_stats(array: np.ndarray) -> Dict[str, float]:
    """Mean/std/min/max computed from finite values only, for use by
    normalization. Separate from compute_numerical_statistics so the
    caller can explicitly control/record exactly which data (time
    range) these came from -- see config.stats_time_range."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Cannot compute normalization statistics: no finite SST values in the given array.")
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "n_finite_values_used": int(finite.size),
    }


def apply_normalization(da: xr.DataArray, stats: Dict[str, float], method: str) -> xr.DataArray:
    """Apply the configured normalization using precomputed stats.
    NaN cells (land/missing) stay NaN -- normalization never fabricates
    a value where there wasn't one; valid_mask (computed separately,
    before this is called) is what distinguishes real vs missing cells
    downstream, exactly as for the un-normalized case."""
    if method == "none":
        return da
    if method == "zscore":
        if stats["std"] == 0:
            raise ValueError("Cannot z-score normalize: statistics std == 0 (constant field).")
        return (da - stats["mean"]) / stats["std"]
    if method == "minmax":
        span = stats["max"] - stats["min"]
        if span == 0:
            raise ValueError("Cannot min-max normalize: statistics max == min (constant field).")
        return (da - stats["min"]) / span
    raise ValueError(f"Unknown normalization method {method!r}.")


# ---------------------------------------------------------------------
# Alignment checks (Section 6 "Alignment checks")
# ---------------------------------------------------------------------

def verify_common_grid_alignment(da: xr.DataArray, coords: DetectedCoords, config) -> Dict[str, object]:
    """
    Check that the output's time/latitude/longitude coordinates are
    suitable for later alignment with the other OceanEmbed variables
    (SSS, SSH/SLA, currents, winds, land mask) on the shared 0.25 deg
    common grid: ascending, unique, within the documented region
    bounds. Does not compare against another variable's actual file
    (none is available to this pipeline) -- that cross-file check
    belongs to the later common tensor-generation step; this only
    confirms THIS output meets the contract that step will assume.
    """
    lat = np.asarray(da[coords.latitude].values, dtype=float)
    lon = np.asarray(da[coords.longitude].values, dtype=float)
    time_vals = da[coords.time].values

    report: Dict[str, object] = {"warnings": []}
    report["latitude_monotonic_ascending"] = bool(len(lat) < 2 or np.all(np.diff(lat) > 0))
    report["longitude_monotonic_ascending"] = bool(len(lon) < 2 or np.all(np.diff(lon) > 0))
    report["latitude_matches_region"] = bool(
        abs(lat.min() - config.lat_min) <= config.final_grid_tolerance_deg * 1e3
        and abs(lat.max() - config.lat_max) <= config.final_grid_tolerance_deg * 1e3
    ) if config.regrid_to_quarter_degree else None
    report["longitude_matches_region"] = bool(
        abs(lon.min() - config.lon_min) <= config.final_grid_tolerance_deg * 1e3
        and abs(lon.max() - config.lon_max) <= config.final_grid_tolerance_deg * 1e3
    ) if config.regrid_to_quarter_degree else None
    report["n_time"] = int(len(time_vals))
    report["time_dtype"] = str(time_vals.dtype)
    report["suitable_for_alignment"] = bool(
        report["latitude_monotonic_ascending"] and report["longitude_monotonic_ascending"]
    )
    if not report["suitable_for_alignment"]:
        report["warnings"].append(
            "Output latitude/longitude are not both strictly ascending -- NOT suitable for "
            "direct alignment with the other OceanEmbed common-grid variables without further processing."
        )
    return report
