"""
preprocess_sst.py
==================
SST preprocessing pipeline for SIH26066 / OceanEmbed
("OceanEmbed - Satellite Embedding-Based Deep Learning Framework for
Reconstruction of Subsurface Ocean Temperature from Surface Satellite
Observations").

Reads the raw OSTIA/CMEMS SST NetCDF (one file, several files, or a
directory), validates and reports its real structure, subsets to the
configured region/time, checks and (if needed) explicitly converts
units, regrids to the official OceanEmbed 0.25 x 0.25 deg common grid,
computes statistics and optional normalization, standardizes coordinate
names, and writes a clean, model-ready SST dataset -- WITHOUT touching
subsurface-temperature/SSS/SSH/currents/winds/land-mask preprocessing
or any model-training/DataLoader code.

Run from the project root:

    python preprocessing/sst/preprocess_sst.py [--dry-run]

All paths, region/time bounds, unit policy, regridding, missing-value
policy, and normalization are configurable via config_sst.py or CLI
flags (see --help) -- nothing here is hard-coded. Everything about the
actual file(s) (variable name, coordinate names, resolution, time
steps, units) is detected/reported at runtime rather than assumed.

Exit codes: 0 = success (or a successful dry run). 2 = no input file
found (re-run later, or pass --input/--input-glob). 1 = input found but
failed validation (the message says exactly what and why -- a
best-effort error report is still written to help debug). Any other
exception is NOT caught here on purpose -- a full traceback is more
useful than a swallowed error for a scientific pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from config_sst import CONFIG, PIPELINE_VERSION, SSTConfig
from netcdf_utils import (
    DetectedCoords,
    atomic_write,
    build_interpolation_support_bounds,
    build_target_grid,
    check_input_output_collision,
    check_units,
    compute_valid_mask,
    convert_units_if_needed,
    detect_all_coords,
    detect_optional_variable,
    detect_variable,
    find_input_files,
    flag_implausible_values,
    inspect_missing_values,
    open_dataset,
    regrid_to_target_grid,
    standardize_output_coordinates,
    subset_region_time,
    validate_coordinates,
)
from validation import (
    apply_normalization,
    check_implausible_fraction,
    check_missing_fraction,
    compute_normalization_stats,
    compute_numerical_statistics,
    verify_common_grid_alignment,
    verify_final_grid,
    verify_official_spatial_coverage,
    verify_output_file,
    verify_temporal_coverage,
)

logger = logging.getLogger("preprocess_sst")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _safe(value, default=float("nan")):
    """None-safe passthrough for %-formatting (avoids the `0.0 or X`
    falsy-zero trap when a stat can legitimately be 0.0)."""
    return default if value is None else value


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _write_json_report(path: Path, report: Dict[str, object], config: SSTConfig) -> None:
    """Best-effort JSON write (dry-run report, stats file, or failure
    report), atomic if config.atomic_writes. Never raises -- a failure
    to write a supplementary report must not mask the original
    success/error."""
    try:
        def _do_write(tmp_path: Path) -> None:
            with open(tmp_path, "w") as f:
                json.dump(report, f, indent=2, default=_json_default)
        if config.atomic_writes:
            atomic_write(path, _do_write)
        else:
            _do_write(path)
        logger.info("Wrote report: %s", path)
    except Exception as exc:  # noqa: BLE001 - reporting is best-effort
        logger.warning("Could not write report to %s (%s: %s).", path, type(exc).__name__, exc)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def _parse_iso_date(value: str, label: str) -> date:
    if not isinstance(value, str) or not _DATE_RE.match(value):
        raise ValueError(f"{label} must use YYYY-MM-DD format, got {value!r}.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid calendar date: {value!r}.") from exc


def _parse_year(value: object, label: str = "year") -> int:
    try:
        year = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a four-digit integer year, got {value!r}.") from exc
    if year < 1 or year > 9999:
        raise ValueError(f"{label} must be within 1..9999, got {year}.")
    return year


def _full_year_dates(year: int) -> Tuple[date, date]:
    return date(year, 1, 1), date(year, 12, 31)


def _inclusive_day_count(start: date, end: date) -> int:
    return (end - start).days + 1


def _is_full_calendar_year(start: date, end: date) -> bool:
    return start == date(start.year, 1, 1) and end == date(start.year, 12, 31)


def _date_selection_report(
    start: date,
    end: date,
    processing_year: Optional[int],
    source: str,
    date_range_confirmed: bool,
) -> Dict[str, object]:
    full_year = _is_full_calendar_year(start, end)
    return {
        "source": source,
        "processing_year": processing_year,
        "requested_start_date": start.isoformat(),
        "requested_end_date": end.isoformat(),
        "requested_day_count": _inclusive_day_count(start, end),
        "is_full_calendar_year_request": full_year,
        "expected_full_year_day_count": _inclusive_day_count(*_full_year_dates(start.year)) if full_year else None,
        "is_leap_year": _inclusive_day_count(*_full_year_dates(start.year)) == 366,
        "date_range_confirmed": date_range_confirmed,
    }


def resolve_cli_or_config_dates(config: SSTConfig, args: argparse.Namespace) -> Tuple[SSTConfig, Dict[str, object]]:
    """
    Resolve dates that are known before opening data. If neither CLI nor
    config supplies a date range, leave the config unresolved so the
    caller can infer an unambiguous year from dataset timestamps after
    coordinate detection.
    """
    explicit_start = args.start_date is not None
    explicit_end = args.end_date is not None
    if explicit_start != explicit_end:
        raise ValueError("--start-date and --end-date must be supplied together.")

    year = _parse_year(args.year, "--year") if args.year is not None else None
    if explicit_start:
        start = _parse_iso_date(args.start_date, "--start-date")
        end = _parse_iso_date(args.end_date, "--end-date")
        if end < start:
            raise ValueError(f"--end-date ({end}) must be on or after --start-date ({start}).")
        if start.year != end.year:
            raise ValueError(
                f"Date range {start}..{end} spans multiple years; run one calendar year/range at a time."
            )
        if year is not None and (start.year != year or end.year != year):
            raise ValueError(
                f"--year {year} conflicts with explicit date range {start}..{end}."
            )
        resolved_year = year if year is not None else start.year
        resolved = replace(
            config,
            processing_year=resolved_year,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            date_range_confirmed=True,
        )
        return resolved, _date_selection_report(start, end, resolved_year, "explicit_date_range", True)

    if year is not None:
        start, end = _full_year_dates(year)
        resolved = replace(
            config,
            processing_year=year,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            date_range_confirmed=True,
        )
        return resolved, _date_selection_report(start, end, year, "explicit_year", True)

    if config.start_date is not None or config.end_date is not None:
        if config.start_date is None or config.end_date is None:
            raise ValueError("config_sst.py start_date and end_date must be set together.")
        start = _parse_iso_date(config.start_date, "config start_date")
        end = _parse_iso_date(config.end_date, "config end_date")
        if end < start:
            raise ValueError(f"config end_date ({end}) must be on or after config start_date ({start}).")
        if start.year != end.year:
            raise ValueError(
                f"Configured date range {start}..{end} spans multiple years; run one year/range at a time."
            )
        resolved_year = config.processing_year or start.year
        if resolved_year != start.year:
            raise ValueError(
                f"config processing_year {resolved_year} conflicts with configured date range {start}..{end}."
            )
        resolved = replace(config, processing_year=resolved_year)
        return resolved, _date_selection_report(
            start, end, resolved_year, "config_default", bool(config.date_range_confirmed)
        )

    return config, {
        "source": "unresolved_pending_dataset_inference",
        "processing_year": None,
        "requested_start_date": None,
        "requested_end_date": None,
        "requested_day_count": None,
        "is_full_calendar_year_request": None,
        "expected_full_year_day_count": None,
        "is_leap_year": None,
        "date_range_confirmed": False,
    }


def infer_dates_from_dataset(config: SSTConfig, coord_report: Dict[str, object]) -> Tuple[SSTConfig, Dict[str, object]]:
    import pandas as pd

    first = coord_report.get("first_timestamp")
    last = coord_report.get("last_timestamp")
    if not first or not last:
        raise ValueError(
            "No --year or explicit date range was supplied, and the dataset has no timestamps to infer from."
        )
    first_day = pd.Timestamp(first).date()
    last_day = pd.Timestamp(last).date()
    years = sorted(int(y) for y in coord_report.get("time_years", []))
    if not years:
        years = sorted(set(range(first_day.year, last_day.year + 1)))
    if len(years) != 1:
        raise ValueError(
            "No --year or explicit date range was supplied, and dataset timestamps span multiple years "
            f"({first_day}..{last_day}). Pass --year YYYY or --start-date/--end-date."
        )
    year = years[0]
    start, end = _full_year_dates(year)
    resolved = replace(
        config,
        processing_year=year,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        date_range_confirmed=False,
    )
    return resolved, _date_selection_report(start, end, year, "inferred_from_dataset_timestamps", False)


def apply_default_year_output_dir(config: SSTConfig, args: argparse.Namespace) -> SSTConfig:
    if args.output_dir is not None or config.processing_year is None:
        return config
    return replace(config, output_dir=str(Path(config.output_dir) / str(config.processing_year)))


def run_output_dir(config: SSTConfig) -> Path:
    return config.resolve(
        config.output_dir if config.regrid_to_quarter_degree else f"{config.output_dir}/{config.native_diagnostic_subdir}"
    )


def update_report_config_summary(report: Dict[str, object], config: SSTConfig, date_selection: Dict[str, object]) -> None:
    report["config_summary"] = {
        "region": [config.lat_min, config.lat_max, config.lon_min, config.lon_max],
        "processing_year": config.processing_year,
        "requested_time_range": [config.start_date, config.end_date],
        "date_selection": date_selection,
        "date_range_confirmed": config.date_range_confirmed,
        "require_complete_temporal_coverage": config.require_complete_temporal_coverage,
        "regrid_to_quarter_degree": config.regrid_to_quarter_degree,
        "unit_conversion_policy": config.unit_conversion_policy,
        "missing_value_policy": config.missing_value_policy,
        "normalization_method": config.normalization_method,
    }

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess OSTIA/CMEMS SST NetCDF file(s) into a clean, model-ready, "
                    "0.25deg-regridded SST dataset for OceanEmbed."
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Explicit single input NetCDF path, or a directory of .nc files. "
                             "Overrides all discovery.")
    parser.add_argument("--input-glob", type=str, default=None,
                        help="Explicit glob (relative to --project-root) matching one or more "
                             "input files, opened together. Overrides sst_input_glob.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Root that relative paths resolve against (default: cwd).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Exact output directory. If omitted, runs with a selected year write under "
                             "data/processed/sst/<year> by default.")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--year", type=int, default=None,
                        help="Process one full calendar year (YYYY-01-01 through YYYY-12-31, end-date inclusive).")
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-range-confirmed", action="store_true",
                        help="For config/default or inferred date ranges, mark start-date/end-date as "
                             "a confirmed project decision in logs and metadata.")
    parser.add_argument("--regrid-method", type=str, default=None, choices=["linear", "nearest"])
    parser.add_argument("--no-regrid", action="store_true",
                        help="Disable regridding and produce a NON-OFFICIAL native-grid "
                             "DIAGNOSTIC output (written to a separate native_diagnostic "
                             "subdirectory, clearly marked in its metadata). The official "
                             "OceanEmbed output is regridded to 0.25 deg by default; this flag "
                             "is for development/testing only.")
    parser.add_argument("--allow-partial-time", action="store_true",
                        help="Allow incomplete temporal coverage instead of failing (default: "
                             "fail for an official run).")
    parser.add_argument("--allow-partial-region", action="store_true",
                        help="(default already True) Explicitly allow partial spatial overlap.")
    parser.add_argument("--reject-excessive-missing", action="store_true",
                        help="Fail the run if the fraction of NaN cells exceeds "
                             "max_missing_fraction (sets missing_value_policy='reject_excessive').")
    parser.add_argument("--max-implausible-fraction", type=float, default=None,
                        help="Fraction of finite values allowed outside the plausible SST range "
                             "before the run fails (default 0.05).")
    parser.add_argument("--unit-policy", type=str, default=None,
                        choices=["auto", "never", "force_k_to_c"],
                        help="Unit conversion policy (default 'auto': convert only if units "
                             "attribute says Kelvin).")
    parser.add_argument("--normalize", type=str, default=None, choices=["none", "zscore", "minmax"],
                        help="Normalization method to apply to the written SST values "
                             "(default 'none'). Statistics are always computed and saved "
                             "regardless of this flag.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional diagnostic PNG.")
    parser.add_argument("--no-atomic", action="store_true", help="Disable atomic temp-file writes (debugging only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect/validate/subset/regrid and report the would-be output "
                             "shape, but do not write sst_target.nc/.npy/mask/stats -- writes a "
                             "dry-run report instead.")
    parser.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SSTConfig:
    overrides: Dict[str, object] = {}
    simple_map = {
        "input": "explicit_input_path", "input_glob": "explicit_input_glob",
        "project_root": "project_root", "output_dir": "output_dir",
        "lat_min": "lat_min", "lat_max": "lat_max", "lon_min": "lon_min", "lon_max": "lon_max",
        "start_date": "start_date", "end_date": "end_date",
        "regrid_method": "regrid_method", "max_implausible_fraction": "max_implausible_fraction",
        "log_level": "log_level",
    }
    for arg_name, field_name in simple_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.date_range_confirmed:
        overrides["date_range_confirmed"] = True
    if args.no_regrid:
        overrides["regrid_to_quarter_degree"] = False
    if args.allow_partial_time:
        overrides["require_complete_temporal_coverage"] = False
    if args.allow_partial_region:
        overrides["allow_partial_spatial_overlap"] = True
    if args.reject_excessive_missing:
        overrides["missing_value_policy"] = "reject_excessive"
    if args.unit_policy is not None:
        overrides["unit_conversion_policy"] = args.unit_policy
    if args.normalize is not None:
        overrides["normalization_method"] = args.normalize
    if args.no_plot:
        overrides["save_diagnostic_plot"] = False
    if args.no_atomic:
        overrides["atomic_writes"] = False
    if args.dry_run:
        overrides["dry_run"] = True
    return replace(CONFIG, **overrides)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def print_dataset_summary(ds: xr.Dataset, paths: List[Path]) -> None:
    logger.info("=" * 70)
    logger.info("DATASET SUMMARY: %s", [str(p) for p in paths])
    logger.info("=" * 70)
    logger.info("Dimensions: %s", dict(ds.sizes))
    logger.info("Coordinates: %s", list(ds.coords))
    logger.info("Data variables: %s", list(ds.data_vars))
    for v in ds.data_vars:
        da = ds[v]
        logger.info("  %-24s dims=%-35s dtype=%s units=%s", v, str(da.dims), da.dtype, da.attrs.get("units"))
    for k, v in ds.attrs.items():
        logger.info("Global attr: %s = %s", k, v)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_outputs(
    da: xr.DataArray,
    mask: np.ndarray,
    coords: DetectedCoords,
    config: SSTConfig,
    metadata: Dict[str, object],
    out_dir: Path,
) -> Dict[str, str]:
    """Write sst_target.nc (+ optional .npy / mask .npy), the metadata
    JSON, and the statistics JSON to out_dir, atomically if
    config.atomic_writes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    values = da.values
    if config.fill_enabled:
        # Deliberately NOT implemented as a blind fill: fill_enabled is
        # off by default and, if ever turned on, must use a
        # scientifically justified, spatially-aware method (e.g.
        # nearest valid ocean neighbor) rather than a constant -- left
        # as a documented gap rather than a fabricated fill here, since
        # implementing land/ocean-aware fill was not exercised against
        # real data. The pipeline still writes correctly with
        # fill_enabled=False (the default and recommended setting).
        raise NotImplementedError(
            "fill_enabled=True is not implemented in this pass -- SST has a real land/ocean "
            "boundary and this pipeline deliberately does not fabricate a blind fill. Leave "
            "fill_enabled=False (the default); valid_mask distinguishes real vs missing cells."
        )
    saved_values = values.astype(np.float32)
    fill_note = "NaN preserved for land/invalid/out-of-coverage cells (fill_enabled=False); see valid_mask."

    assert mask.shape == saved_values.shape, f"Internal error: mask shape {mask.shape} != value shape {saved_values.shape}."

    var_name = config.output_variable_name
    out_ds = xr.Dataset(
        {
            var_name: (da.dims, saved_values),
            "valid_mask": (da.dims, mask.astype(np.uint8)),
        },
        coords={
            coords.time: da[coords.time],
            coords.latitude: da[coords.latitude],
            coords.longitude: da[coords.longitude],
        },
    )
    physical_units = metadata.get("output_units") or "unknown (see unit_conversion report in metadata)"
    norm_method = metadata.get("normalization_method", "none")
    if norm_method == "none":
        out_ds[var_name].attrs["units"] = physical_units
    else:
        # Normalized values are no longer in physical units -- labeling
        # them "degrees_C" (or whatever the pre-normalization unit was)
        # would be exactly the kind of silent/misleading unit label the
        # project brief prohibits. dimensionless is correct for both
        # zscore and minmax; the original physical unit is preserved
        # separately so it's never lost.
        out_ds[var_name].attrs["units"] = "1 (dimensionless)"
        out_ds[var_name].attrs["pre_normalization_units"] = physical_units
    out_ds[var_name].attrs["source_variable"] = metadata["variable_name"]
    out_ds[var_name].attrs["missing_value_handling"] = fill_note
    out_ds[var_name].attrs["normalization"] = norm_method
    out_ds["valid_mask"].attrs["description"] = (
        "1 = SST value is finite at this cell, 0 = NaN/Inf. IMPORTANT -- this is a SINGLE finite-"
        "value mask, not a verified land-sea mask: a 0 here can mean land, a missing ocean "
        "observation, a sea-ice/invalid region, OR (after regridding) a target cell that fell "
        "outside the source data's actual coverage -- these are NOT distinguished from each other "
        "by this variable's value. See 'land_ocean_mask_caveat' and 'source_land_sea_mask_variable' "
        "global attributes."
    )
    out_ds[coords.latitude].attrs.setdefault("units", "degrees_north")
    out_ds[coords.longitude].attrs.setdefault("units", "degrees_east")
    out_ds.attrs["source_dataset"] = "OSTIA L4 SST (Copernicus Marine / CMEMS), product METOFFICE-GLO-SST-L4-REP-OBS-SST"
    out_ds.attrs["source_variable"] = "analysed_sst (as detected: see source_variable in metadata)"
    out_ds.attrs["source_file"] = str(metadata["input_files"])
    out_ds.attrs["processing_script"] = "preprocess_sst.py"
    out_ds.attrs["pipeline_version"] = PIPELINE_VERSION
    out_ds.attrs["processed_at_utc"] = metadata["processed_at_utc"]
    out_ds.attrs["dimension_order"] = " -> ".join(da.dims)
    out_ds.attrs["spatial_resolution_status"] = metadata["spatial_resolution_status"]
    out_ds.attrs["is_official_output"] = "true" if metadata.get("is_official_output") else "FALSE - NON-OFFICIAL NATIVE-GRID DIAGNOSTIC"
    out_ds.attrs["regridding_method"] = metadata.get("regridding", {}).get("method", "none (native-grid diagnostic)")
    out_ds.attrs["extrapolation_policy"] = metadata.get("regridding", {}).get("extrapolation_policy", "n/a (not regridded)")
    out_ds.attrs["land_ocean_mask_caveat"] = (
        "valid_mask combines land, missing ocean observations, sea-ice/invalid regions, and "
        "post-regrid out-of-source-coverage cells into ONE finite-value mask; it is not a verified, "
        "authoritative land-sea mask. The regridding method (xarray.DataArray.interp, linear or "
        "nearest) is NOT a coastline-aware interpolation scheme -- it does not know where the true "
        "coastline is and can produce interpolation artifacts for cells near a land/ocean boundary. "
        "If a source categorical land/sea mask variable was present in the input file, its name is "
        "recorded below (source_land_sea_mask_variable) but it is NOT currently propagated into this "
        "output -- see README 'Land/ocean handling and mask semantics'."
    )
    out_ds.attrs["source_land_sea_mask_variable"] = metadata.get("optional_companion_variables", {}).get("mask") or "none detected"
    out_ds.attrs["unit_conversion"] = metadata.get("unit_conversion", {}).get("note", "n/a")
    out_ds.attrs["requested_lat_bounds"] = f"{config.lat_min} to {config.lat_max}"
    out_ds.attrs["requested_lon_bounds"] = f"{config.lon_min} to {config.lon_max}"
    fg = metadata.get("final_grid", {})
    out_ds.attrs["actual_lat_bounds"] = f"{fg.get('latitude_min')} to {fg.get('latitude_max')}"
    out_ds.attrs["actual_lon_bounds"] = f"{fg.get('longitude_min')} to {fg.get('longitude_max')}"

    written: Dict[str, str] = {}
    encoding = None
    if config.enable_netcdf_compression:
        comp = {"zlib": True, "complevel": config.netcdf_complevel}
        encoding = {var_name: {**comp, "dtype": "float32"}, "valid_mask": {**comp, "dtype": "uint8"}}

    nc_path = out_dir / config.output_nc_name
    if config.atomic_writes:
        atomic_write(nc_path, lambda tmp: out_ds.to_netcdf(tmp, encoding=encoding))
    else:
        out_ds.to_netcdf(nc_path, encoding=encoding)
    written["netcdf"] = str(nc_path)

    if config.write_npy_outputs:
        npy_path = out_dir / config.output_npy_name
        mask_path = out_dir / config.output_mask_npy_name
        if config.atomic_writes:
            atomic_write(npy_path, lambda tmp: np.save(str(tmp), saved_values))
            atomic_write(mask_path, lambda tmp: np.save(str(tmp), mask.astype(bool)))
        else:
            np.save(npy_path, saved_values)
            np.save(mask_path, mask.astype(bool))
        written["npy_values"] = str(npy_path)
        written["npy_mask"] = str(mask_path)

    meta_path = out_dir / config.output_metadata_name
    _write_json_report(meta_path, metadata, config)
    written["metadata_json"] = str(meta_path)

    stats_path = out_dir / config.stats_output_name
    _write_json_report(stats_path, metadata["statistics_report"], config)
    written["statistics_json"] = str(stats_path)

    return written


def make_diagnostic_plot(da: xr.DataArray, coords: DetectedCoords, config: SSTConfig) -> Optional[str]:
    """Best-effort diagnostic PNG. Never raises -- logs and returns None
    if matplotlib is unavailable or there's nothing valid to plot."""
    if not config.save_diagnostic_plot:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping diagnostic plot: matplotlib unavailable (%s).", exc)
        return None

    try:
        values = da.values  # (time, lat, lon)
        lat_vals = da[coords.latitude].values
        lon_vals = da[coords.longitude].values

        finite_counts = np.isfinite(values).reshape(values.shape[0], -1).sum(axis=1)
        if finite_counts.max() == 0:
            logger.warning("Skipping diagnostic plot: no valid (finite) values anywhere in the array.")
            return None
        t0 = int(np.argmax(finite_counts))
        t_last = len(values) - 1

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        im0 = axes[0].pcolormesh(lon_vals, lat_vals, values[t0], shading="auto")
        axes[0].set_title(f"t={t0} (first step with data)")
        axes[0].set_xlabel("longitude")
        axes[0].set_ylabel("latitude")
        fig.colorbar(im0, ax=axes[0], label="SST (as stored/converted)")

        im1 = axes[1].pcolormesh(lon_vals, lat_vals, values[t_last], shading="auto")
        axes[1].set_title(f"t={t_last} (last step)")
        axes[1].set_xlabel("longitude")
        axes[1].set_ylabel("latitude")
        fig.colorbar(im1, ax=axes[1], label="SST (as stored/converted)")

        fig.tight_layout()
        out_dir = config.resolve(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / config.diagnostic_plot_name
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        return str(plot_path)
    except Exception as exc:
        logger.warning("Skipping diagnostic plot due to an error (%s: %s).", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    setup_logging(config.log_level)
    try:
        config, date_selection = resolve_cli_or_config_dates(config, args)
        config = apply_default_year_output_dir(config, args)
    except ValueError as exc:
        logger.error("Date selection error: %s", exc)
        return 1

    logger.info("SST preprocessing (SIH26066 / OceanEmbed) v%s", PIPELINE_VERSION)
    if config.regrid_to_quarter_degree:
        logger.info("Mode: OFFICIAL -- output will be regridded to %s deg and validated as such "
                    "(fatal on any regridding failure).", config.target_resolution_deg)
    else:
        logger.warning("Mode: NON-OFFICIAL NATIVE-GRID DIAGNOSTIC (--no-regrid) -- output written to "
                    "'%s/' and clearly marked; this is NOT the official model-ready output.",
                    config.native_diagnostic_subdir)
    logger.info("project_root=%s  dry_run=%s", Path(config.project_root).resolve(), config.dry_run)
    if config.start_date and config.end_date:
        logger.info("Region: lat[%s,%s] lon[%s,%s], time[%s,%s] (%s)",
                    config.lat_min, config.lat_max, config.lon_min, config.lon_max,
                    config.start_date, config.end_date, date_selection["source"])
    else:
        logger.info("Region: lat[%s,%s] lon[%s,%s], time pending dataset-year inference",
                    config.lat_min, config.lat_max, config.lon_min, config.lon_max)
    if config.start_date and config.end_date and not config.date_range_confirmed:
        logger.warning(
            "start_date/end_date (%s..%s) were not explicitly confirmed by CLI/config; pass --year "
            "or --start-date/--end-date for explicit date selection.",
            config.start_date, config.end_date,
        )

    report: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "dry_run": config.dry_run,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    update_report_config_summary(report, config, date_selection)
    run_out_dir = run_output_dir(config)

    paths, source_label = find_input_files(config)
    report["input_files"] = [str(p) for p in paths]
    report["input_file_source"] = source_label
    if not paths:
        logger.error(
            "No SST input file found (tried sst_input_candidates=%s, glob '%s', resolved "
            "against project_root=%s).", config.sst_input_candidates, config.sst_input_glob,
            Path(config.project_root).resolve(),
        )
        logger.error("Re-run once the data is in place, or pass --input <path>/--input-glob <pattern>.")
        report["error"] = "no_input_file_found"
        report_name = config.dryrun_report_name if config.dry_run else "sst_error_report.json"
        _write_json_report(run_out_dir / report_name, report, config)
        return 2
    logger.info("Using %d input file(s) (%s): %s", len(paths), source_label, [str(p) for p in paths])

    # --- Explicit input/output path collision check, BEFORE opening
    # anything or writing any report. Raw input files must never be
    # overwritten or written into -- this is a hard fail, not a warning.
    try:
        check_input_output_collision(paths, run_out_dir)
    except ValueError as exc:
        logger.error("Input/output path collision: %s", exc)
        report["error"] = str(exc)
        # Deliberately do NOT write the error report into run_out_dir here --
        # if run_out_dir itself is the problem (e.g. equals the input
        # location), writing into it would repeat the exact mistake this
        # check exists to prevent. Report to stderr/log only in this one case.
        return 1

    try:
        ds = open_dataset(paths, use_dask_chunking=config.use_dask_chunking)
    except (FileNotFoundError, OSError) as exc:
        logger.error(str(exc))
        report["error"] = str(exc)
        _write_json_report(run_out_dir / "sst_error_report.json", report, config)
        return 2

    try:
        print_dataset_summary(ds, paths)

        var_name = detect_variable(ds, config.sst_var_candidates)
        coords = detect_all_coords(ds, config)
        mask_var = detect_optional_variable(ds, config.mask_var_candidates)
        error_var = detect_optional_variable(ds, config.error_var_candidates)
        ice_var = detect_optional_variable(ds, config.sea_ice_var_candidates)
        report["variable_name"] = var_name
        report["detected_coordinate_names"] = {"time": coords.time, "latitude": coords.latitude, "longitude": coords.longitude}
        report["optional_companion_variables"] = {"mask": mask_var, "analysis_error": error_var, "sea_ice_fraction": ice_var}
        logger.info("Detected SST variable: '%s'", var_name)
        logger.info("Detected coordinates: time='%s' latitude='%s' longitude='%s'", coords.time, coords.latitude, coords.longitude)
        logger.info("Optional companion variables found: mask=%s analysis_error=%s sea_ice_fraction=%s", mask_var, error_var, ice_var)

        expected_dims = {coords.time, coords.latitude, coords.longitude}
        actual_dims = set(ds[var_name].dims)
        if actual_dims != expected_dims:
            raise ValueError(
                f"Variable '{var_name}' has dims {ds[var_name].dims}, expected exactly "
                f"(time, latitude, longitude) = {sorted(expected_dims)}. If an extra dimension "
                f"(e.g. a singleton depth/band) is real and expected, handle it explicitly here "
                f"rather than letting it pass silently."
            )

        coord_report = validate_coordinates(ds, coords)
        report["coordinate_validation"] = coord_report
        all_warnings: List[str] = list(coord_report.get("warnings", []))
        logger.info("Native grid: %d lat x %d lon (~%.4f x %.4f deg), %d time steps",
                    coord_report["n_latitude"], coord_report["n_longitude"],
                    _safe(coord_report["latitude_resolution_deg"]), _safe(coord_report["longitude_resolution_deg"]),
                    coord_report["n_time"])
        logger.info("Time range: %s to %s (daily: %s, missing dates: %d, duplicates: %s)",
                    coord_report["first_timestamp"], coord_report["last_timestamp"],
                    coord_report["is_daily"], len(coord_report["missing_dates"]), coord_report["time_has_duplicate_timestamps"])
        logger.info("Longitude convention detected: %s", coord_report["longitude_convention"])

        if config.start_date is None or config.end_date is None:
            config, date_selection = infer_dates_from_dataset(config, coord_report)
            config = apply_default_year_output_dir(config, args)
            run_out_dir = run_output_dir(config)
            update_report_config_summary(report, config, date_selection)
            logger.info("Inferred processing year %s from dataset timestamps; requested range %s..%s",
                        config.processing_year, config.start_date, config.end_date)
            try:
                check_input_output_collision(paths, run_out_dir)
            except ValueError as exc:
                logger.error("Input/output path collision after date inference: %s", exc)
                report["error"] = str(exc)
                return 1

        # --- Official spatial coverage check: compares the SOURCE
        # FILE'S OWN (un-subsetted) coordinate extent against the
        # requested region, with a tolerance for native-grid boundary
        # alignment offset (e.g. OSTIA's ~0.05 deg grid not landing
        # exactly on 5/30/45/105) -- NOT a strict subset's resulting
        # extent, which would (incorrectly) fail on that same alignment
        # offset even though the source genuinely has the coverage.
        # Deliberately checked here, right after coord_report is
        # available and BEFORE any subsetting, since it no longer
        # depends on how a boolean-mask subset happens to land on the
        # native grid. See verify_official_spatial_coverage()'s
        # docstring in validation.py for the full rationale (this is a
        # correction of a real false-positive failure observed against
        # the actual 2025 OSTIA file's ~0.025 deg boundary offset).
        coverage_report = verify_official_spatial_coverage(coord_report, config)
        report["official_coverage_validation"] = coverage_report
        if coverage_report.get("checked"):
            logger.info(
                "Official coverage check PASSED: source lat[%.4f,%.4f] lon[%.4f,%.4f] vs requested "
                "lat[%s,%s] lon[%s,%s] -- boundary gaps (deg) lat_low=%.4f lat_high=%.4f lon_low=%.4f "
                "lon_high=%.4f, tolerance lat=%.4f lon=%.4f (%s native cell(s), native spacing "
                "lat=%s lon=%s)",
                coverage_report["source_bounds"]["lat_min"], coverage_report["source_bounds"]["lat_max"],
                coverage_report["source_bounds"]["lon_min"], coverage_report["source_bounds"]["lon_max"],
                config.lat_min, config.lat_max, config.lon_min, config.lon_max,
                coverage_report["boundary_gaps_deg"]["lat_low"], coverage_report["boundary_gaps_deg"]["lat_high"],
                coverage_report["boundary_gaps_deg"]["lon_low"], coverage_report["boundary_gaps_deg"]["lon_high"],
                coverage_report["tolerance_deg"]["latitude"], coverage_report["tolerance_deg"]["longitude"],
                coverage_report["coverage_tolerance_in_native_cells"],
                coverage_report["native_spacing_deg"]["latitude"], coverage_report["native_spacing_deg"]["longitude"],
            )
        else:
            logger.info("Official coverage check skipped: %s", coverage_report.get("note"))

        original_dims = dict(ds[var_name].sizes)
        report["original_dimensions"] = original_dims

        # --- Missingness BEFORE subsetting: a lazy, dask-aware count
        # over the file's full native-resolution extent (whatever that
        # is -- could be much larger than the OceanEmbed region). Best-
        # effort/non-fatal: informational only, and skipped without
        # failing the run if it errors (e.g. an unusually shaped or
        # very large multi-file open makes even a lazy full-domain
        # reduction impractical) -- this is a diagnostic, not a gate.
        try:
            missing_before_subsetting = inspect_missing_values(ds[var_name])
            report["missing_values_before_subsetting"] = missing_before_subsetting
            logger.info("Missingness before subsetting (full native extent): %d/%d cells NaN/Inf (%.2f%%)",
                        missing_before_subsetting["nan_or_inf_count"], missing_before_subsetting["total_cells"],
                        100.0 * _safe(missing_before_subsetting["fraction_invalid"], 0.0))
        except Exception as exc:  # noqa: BLE001 - diagnostic only, never fatal
            logger.warning("Could not compute pre-subsetting missingness (non-fatal, informational only): %s", exc)
            report["missing_values_before_subsetting"] = None

        # --- STRICT requested-region subset: kept for reporting and for
        # verify_temporal_coverage() below (time coverage is unaffected
        # by the lat/lon expansion discussed next, so the strict subset
        # remains the right source for temporal completeness). This is
        # NOT what gets fed to regridding anymore -- see the
        # interpolation-support subset immediately after.
        subset_ds, region_report = subset_region_time(ds, coords, config, bounds_label="requested (strict)")
        report["requested_region_time"] = region_report
        report["date_coverage"] = {
            "processing_year": config.processing_year,
            "requested_start_date": config.start_date,
            "requested_end_date": config.end_date,
            "actual_first_timestamp": region_report.get("actual_time_range", [None, None])[0],
            "actual_last_timestamp": region_report.get("actual_time_range", [None, None])[1],
            "n_timestamps": region_report.get("n_time_matched"),
            "expected_daily_timestamps": region_report.get("expected_daily_timestamps"),
            "expected_full_year_day_count": date_selection.get("expected_full_year_day_count"),
            "missing_dates": region_report.get("missing_dates_in_requested_range", []),
            "duplicate_dates": region_report.get("duplicate_dates_in_requested_range", []),
            "requested_period_fully_covered": region_report.get("requested_period_fully_covered"),
            "is_full_calendar_year_request": date_selection.get("is_full_calendar_year_request"),
            "is_leap_year": date_selection.get("is_leap_year"),
        }
        all_warnings.extend(region_report.get("warnings", []))
        logger.info("Original extent: lat%s lon%s", region_report["original_lat_range"], region_report["original_lon_range"])
        logger.info("Requested extent: lat%s lon%s time%s", region_report["requested_lat_range"],
                    region_report["requested_lon_range"], region_report["requested_time_range"])
        logger.info("Strict-subset actual extent: lat%s lon%s time%s -- dims=%s (this is the STRICT boundary "
                    "selection; the slightly larger interpolation-support extent used for regridding is logged next)",
                    region_report["actual_lat_range"], region_report["actual_lon_range"],
                    region_report["actual_time_range"], region_report["resulting_dimensions"])
        logger.info("Actual first/last selected timestamps: %s to %s (end-date handling is END-INCLUSIVE: "
                    "any timestamp on the requested end_date's calendar day is kept, regardless of its time-of-day)",
                    region_report["actual_time_range"][0], region_report["actual_time_range"][1])

        verify_temporal_coverage(coord_report, region_report, config)

        # --- INTERPOLATION-SUPPORT subset: the strict subset above
        # naturally stops up to half a native cell inside the requested
        # boundary whenever the native grid doesn't land exactly on it
        # (the normal case) -- regridding FROM that strict subset alone
        # would starve target boundary cells of real neighboring source
        # data even though the source file actually has it just outside
        # the strict window. This expanded subset (padded by
        # config.interpolation_support_cells native cells on each side,
        # via build_interpolation_support_bounds()) is what's actually
        # used as the source for regridding below. The FINAL OUTPUT
        # region is never affected by this -- build_target_grid() always
        # builds the target grid from config.lat_min/lat_max/lon_min/
        # lon_max directly, regardless of how much source data was used
        # to interpolate it.
        support_bounds = build_interpolation_support_bounds(
            config, coord_report.get("latitude_resolution_deg"), coord_report.get("longitude_resolution_deg")
        )
        report["interpolation_support_bounds"] = support_bounds
        support_ds, support_report = subset_region_time(
            ds, coords, config,
            lat_min=support_bounds["lat_min"], lat_max=support_bounds["lat_max"],
            lon_min=support_bounds["lon_min"], lon_max=support_bounds["lon_max"],
            bounds_label="interpolation_support",
        )
        report["interpolation_support_region"] = support_report
        all_warnings.extend(
            w for w in support_report.get("warnings", [])
            if w not in region_report.get("warnings", [])  # avoid duplicating the same partial-overlap warning twice
        )
        logger.info(
            "Interpolation-support extent (padded %s native cell(s), lat_pad=%.4fdeg lon_pad=%.4fdeg): "
            "lat%s lon%s -- dims=%s",
            support_bounds["support_cells"], support_bounds["lat_pad_deg"], support_bounds["lon_pad_deg"],
            support_report["actual_lat_range"], support_report["actual_lon_range"], support_report["resulting_dimensions"],
        )

        da = support_ds[var_name].transpose(coords.time, coords.latitude, coords.longitude)

        units, unit_warnings = check_units(ds[var_name])
        all_warnings.extend(unit_warnings)
        da, unit_report = convert_units_if_needed(da, units, config)
        report["unit_conversion"] = unit_report
        logger.info("Units: declared=%s -> %s", unit_report["declared_units"], unit_report["note"])

        # --- Missingness AFTER subsetting (still pre-regrid, still
        # potentially native-resolution and large) -- lazy, see above.
        missing_after_subsetting = inspect_missing_values(da)
        report["missing_values_after_subsetting"] = missing_after_subsetting

        # --- Physical-value safety check happens BEFORE regridding: an
        # unmasked fill-value sentinel would otherwise get smeared into
        # real neighbouring ocean cells by linear interpolation,
        # corrupting them before there's ever a chance to catch it.
        # inspect_missing_values / flag_implausible_values operate
        # lazily on the xr.DataArray here (NOT da.values) -- this is the
        # single biggest memory-safety fix versus the previous version:
        # `da` at this point is the SUBSETTED but NOT YET REGRIDDED
        # array, which for a full year at OSTIA's native ~0.05 deg
        # resolution over this region can be on the order of ~2*10^8
        # cells -- too large to comfortably materialize as one in-memory
        # float32 numpy array (roughly 800 MB-1 GB) just to count NaNs
        # or check a value range.
        missing_pre_regrid = missing_after_subsetting
        implausible_pre_regrid = flag_implausible_values(da, config.implausible_low_c, config.implausible_high_c)
        report["missing_values_pre_regrid"] = missing_pre_regrid
        report["sst_stats_finite_only_pre_regrid"] = implausible_pre_regrid
        if implausible_pre_regrid["n_implausible"] > 0:
            all_warnings.append(
                f"{implausible_pre_regrid['n_implausible']} finite value(s) fall outside the plausible "
                f"[{config.implausible_low_c}, {config.implausible_high_c}] degC range (checked pre-regrid)."
            )
        check_implausible_fraction(implausible_pre_regrid, missing_pre_regrid, config)
        check_missing_fraction(missing_pre_regrid, config)

        # --- Regridding: OFFICIAL and default-on. FATAL BY DESIGN on any
        # failure -- deliberately no try/except-and-continue here.
        # Processed in configurable time batches (config.regrid_time_
        # batch_size) so peak memory is bounded by one batch's worth of
        # native-resolution data, not the whole subsetted array at once
        # -- see regrid_to_target_grid()'s docstring in netcdf_utils.py.
        regrid_report: Dict[str, object] = {}
        if config.regrid_to_quarter_degree:
            da, regrid_report = regrid_to_target_grid(da, coords, config)
            da = da.transpose(coords.time, coords.latitude, coords.longitude)
            all_warnings.extend(regrid_report.get("warnings", []))
            spatial_status = (f"regridded to the official {config.target_resolution_deg} deg target grid "
                            f"via {config.regrid_method} interpolation, no extrapolation")
            is_official_output = True
        else:
            spatial_status = "NATIVE SST RESOLUTION -- NON-OFFICIAL DIAGNOSTIC OUTPUT (regridding explicitly disabled)"
            all_warnings.append(
                "regrid_to_quarter_degree=False: this is a non-official native-grid DIAGNOSTIC output, "
                "written to native_diagnostic_subdir, NOT the official 0.25 deg model-ready output."
            )
            is_official_output = False
            logger.warning(spatial_status)

        # --- Missingness AFTER regridding. By this point `da` is the
        # small, already-materialized regridded array (regrid_to_target_
        # grid always .compute()s each batch), so passing it either as
        # a DataArray or .values is equally cheap here -- passed as a
        # DataArray for API consistency with the pre-regrid calls above.
        missing_after_regridding = inspect_missing_values(da)
        implausible_final = flag_implausible_values(da, config.implausible_low_c, config.implausible_high_c)
        missing_final = missing_after_regridding
        report["missing_values_after_regridding"] = missing_after_regridding
        check_missing_fraction(missing_final, config)
        logger.info("Missingness before subsetting: %s | after subsetting: %s | after regridding: %d/%d (%.2f%%)",
                    "n/a (not computed)" if not report.get("missing_values_before_subsetting") else
                    f"{report['missing_values_before_subsetting']['nan_or_inf_count']}/{report['missing_values_before_subsetting']['total_cells']}",
                    f"{missing_pre_regrid['nan_or_inf_count']}/{missing_pre_regrid['total_cells']}",
                    missing_after_regridding["nan_or_inf_count"], missing_after_regridding["total_cells"],
                    100.0 * _safe(missing_after_regridding["fraction_invalid"], 0.0))

        mask = compute_valid_mask(da)

        # --- Statistics (ALWAYS computed, regardless of normalization
        # setting) and optional normalization. Statistics are computed
        # from stats_time_range if given, else from the full processed
        # array -- and this scope is recorded explicitly so it's never
        # ambiguous what the numbers describe.
        if config.stats_time_range:
            import pandas as pd
            t0, t1 = pd.Timestamp(config.stats_time_range[0]), pd.Timestamp(config.stats_time_range[1])
            stats_source_da = da.sel({coords.time: slice(t0, t1)})
            stats_scope_note = f"Computed from stats_time_range=[{config.stats_time_range[0]}, {config.stats_time_range[1]}] only."
        else:
            stats_source_da = da
            stats_scope_note = (
                "Computed from the FULL processed (subsetted/regridded) array -- no separate "
                "train/test split has been defined for this internal round. Recompute from a "
                "training-only time range (set stats_time_range) before this is used for real "
                "model training, to avoid leakage into a held-out test period."
            )
        normalization_stats = compute_normalization_stats(stats_source_da.values)
        statistics_report = {
            "variable": config.output_variable_name,
            "units": unit_report["output_units"],
            "scope": stats_scope_note,
            "mean": normalization_stats["mean"],
            "std": normalization_stats["std"],
            "min": normalization_stats["min"],
            "max": normalization_stats["max"],
            "n_finite_values_used": normalization_stats["n_finite_values_used"],
            "normalization_method_applied_to_output": config.normalization_method,
        }
        if config.normalization_method != "none":
            da = apply_normalization(da, normalization_stats, config.normalization_method)
            logger.info("Applied '%s' normalization using stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                        config.normalization_method, normalization_stats["mean"], normalization_stats["std"],
                        normalization_stats["min"], normalization_stats["max"])
        else:
            logger.info("normalization_method='none': statistics computed and saved, but not applied to output values.")

        # --- Standardize coordinate names for common-grid compatibility.
        da, coords, rename_map = standardize_output_coordinates(da, coords, config)
        if rename_map:
            logger.info("Standardized output coordinate names: %s", rename_map)
            all_warnings.append(f"Renamed coordinates for common-grid compatibility: {rename_map}")
        report["output_coordinate_rename_map"] = rename_map

        final_grid_report = verify_final_grid(da, coords, config)
        report["final_grid"] = final_grid_report
        logger.info("Final grid: lat[%.4f,%.4f] n=%d spacing=%.4fdeg  lon[%.4f,%.4f] n=%d spacing=%.4fdeg",
                    final_grid_report["latitude_min"], final_grid_report["latitude_max"], final_grid_report["n_latitude"],
                    _safe(final_grid_report["latitude_spacing_deg"]),
                    final_grid_report["longitude_min"], final_grid_report["longitude_max"], final_grid_report["n_longitude"],
                    _safe(final_grid_report["longitude_spacing_deg"]))

        alignment_report = verify_common_grid_alignment(da, coords, config)
        report["alignment_check"] = alignment_report
        all_warnings.extend(alignment_report.get("warnings", []))

        numeric_stats = compute_numerical_statistics(da.values)
        numeric_stats["n_time_steps"] = int(da.sizes[coords.time])
        report["numerical_checks"] = numeric_stats

        out_dir = run_out_dir
        expected_shape = tuple(da.sizes[d] for d in (coords.time, coords.latitude, coords.longitude))
        output_files = {
            "netcdf": str(out_dir / config.output_nc_name),
            "npy_values": str(out_dir / config.output_npy_name) if config.write_npy_outputs else None,
            "npy_mask": str(out_dir / config.output_mask_npy_name) if config.write_npy_outputs else None,
            "metadata_json": str(out_dir / config.output_metadata_name),
            "statistics_json": str(out_dir / config.stats_output_name),
            "diagnostic_plot": str(out_dir / config.diagnostic_plot_name) if config.save_diagnostic_plot else None,
        }

        metadata: Dict[str, object] = {
            **report,
            "output_files": output_files,
            "is_official_output": is_official_output,
            "processed_dimensions": dict(da.sizes),
            "dimension_order": list(da.dims),
            "expected_output_shape": list(expected_shape),
            "output_units": unit_report["output_units"],
            "spatial_resolution_status": spatial_status,
            "regridding": regrid_report,
            "final_grid": final_grid_report,
            "alignment_check": alignment_report,
            "sst_stats_finite_only_pre_regrid": implausible_pre_regrid,
            "sst_stats_finite_only_final": implausible_final,
            "missing_values_pre_regrid": missing_pre_regrid,
            "missing_values_final": missing_final,
            "numerical_checks": numeric_stats,
            "statistics_report": statistics_report,
            "normalization_method": config.normalization_method,
            "valid_mask_cell_count": int(mask.sum()),
            "invalid_masked_cell_count": int(mask.size - mask.sum()),
            "n_time_steps": int(da.sizes[coords.time]),
            "fill_enabled": config.fill_enabled,
            "warnings": all_warnings,
        }

        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info("Input(s):  %s", metadata["input_files"])
        logger.info("Variable: %s  units=%s", var_name, unit_report["output_units"])
        logger.info("Original dims: %s -> Processed dims: %s", original_dims, metadata["processed_dimensions"])
        logger.info("SST (finite values, final array): min=%.3f max=%.3f mean=%.3f std=%.3f",
                    _safe(implausible_final["min_finite"]), _safe(implausible_final["max_finite"]),
                    _safe(implausible_final["mean_finite"]), _safe(implausible_final["std_finite"]))
        logger.info("Valid cells: %d / %d (%.2f%% invalid/masked)", metadata["valid_mask_cell_count"], mask.size,
                    100.0 * metadata["invalid_masked_cell_count"] / mask.size if mask.size else 0.0)
        logger.info("Spatial resolution: %s", spatial_status)
        if all_warnings:
            logger.warning("%d warning(s):", len(all_warnings))
            for w in all_warnings:
                logger.warning("  - %s", w)
        else:
            logger.info("No warnings.")

        if config.dry_run:
            metadata["note"] = "DRY RUN: no sst_target.nc/.npy/mask/stats written. This is the structure/validation that WOULD be produced."
            dryrun_path = out_dir / config.dryrun_report_name
            _write_json_report(dryrun_path, metadata, config)
            logger.info("DRY RUN complete -- no processed outputs written. Report: %s", dryrun_path)
            logger.info("Would-be output shape (time, lat, lon): %s", expected_shape)
            return 0

        written = save_outputs(da, mask, coords, config, metadata, out_dir)
        logger.info("Wrote outputs:")
        for k, v in written.items():
            logger.info("  %s: %s", k, v)

        plot_path = make_diagnostic_plot(da, coords, config)
        if plot_path:
            logger.info("  diagnostic_plot: %s", plot_path)

        verify_report = verify_output_file(Path(written["netcdf"]), config, expected_shape)
        logger.info("Output reopening check: OK -- dtype=%s units=%s mask_dtype=%s shape=%s",
                    verify_report["reopened_dtype"], verify_report["reopened_units"],
                    verify_report["reopened_mask_dtype"], verify_report["reopened_shape"])
        for w in verify_report.get("warnings", []):
            logger.warning("  - %s", w)

        logger.info("Done.")
        return 0

    except ValueError as exc:
        logger.error("Validation error: %s", exc)
        report["error"] = str(exc)
        _write_json_report(run_out_dir / "sst_error_report.json", report, config)
        return 1
    finally:
        ds.close()


if __name__ == "__main__":
    sys.exit(main())
