"""
preprocess_temperature.py
==========================
Preprocessing pipeline for the GLORYS `thetao` (potential temperature)
target, for SIH26066 / OceanEmbed. ACTIVE dataset: 2025 ocean data (the
January 2026 prototype plan is discontinued -- see config_temperature.py).

Reads the raw 2025 GLORYS NetCDF (one file or several), validates and
reports its real structure, subsets to the configured region/time/
depths, standardizes coordinate names to match what
oceanembed_step5/config.py expects, and writes a clean, model-ready
subsurface temperature target dataset -- WITHOUT touching SST/SSS/SSH/
currents/winds/ARGO preprocessing or any model-training code.

Run from the project root (the directory containing preprocessing/ and
month_data/):

    python preprocessing/temperature/preprocess_temperature.py [--dry-run]

All paths, region/time bounds, depths, and strictness flags are
configurable via config_temperature.py or CLI flags (see --help) --
nothing here is hard-coded. Everything about the actual file(s)
(variable name, coordinate names, resolution, available depths, time
steps) is detected/reported at runtime rather than assumed.

Exit codes: 0 = success (or a successful dry run). 2 = no input file
found (e.g. the 2025 data isn't in place yet -- re-run later, or pass
--input/--input-glob). 1 = input found but failed validation (the
message says exactly what and why -- a best-effort error report is
still written to help debug). Any other exception is NOT caught here
on purpose -- it means an unexpected bug, and a full traceback is more
useful than a swallowed error for a scientific pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from config_temperature import CONFIG, PIPELINE_VERSION, TemperatureConfig
from netcdf_utils import (
    DetectedCoords,
    atomic_write,
    build_target_grid,
    check_implausible_fraction,
    check_units,
    compute_valid_mask,
    detect_all_coords,
    detect_variable,
    find_input_files,
    flag_implausible_values,
    inspect_missing_values,
    open_dataset,
    regrid_to_target_grid,
    select_depths,
    standardize_output_coordinates,
    subset_region_time,
    validate_coordinates,
    verify_depth_count,
    verify_final_grid,
    verify_output_file,
    verify_temporal_coverage,
)

logger = logging.getLogger("preprocess_temperature")


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


def _write_json_report(path: Path, report: Dict[str, object], config: TemperatureConfig) -> None:
    """Best-effort JSON write (dry-run report or failure report), atomic
    if config.atomic_writes. Never raises -- a failure to write a
    supplementary report must not mask the original success/error."""
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

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the active 2025 GLORYS thetao NetCDF file(s) into a "
                    "clean, model-ready subsurface temperature target dataset."
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Explicit single input NetCDF path. Overrides all discovery.")
    parser.add_argument("--input-glob", type=str, default=None,
                        help="Explicit glob (relative to --project-root) matching one or "
                             "more input files, opened together. Overrides year2025_input_glob.")
    parser.add_argument("--project-root", type=str, default=None,
                        help="Root that relative paths resolve against (default: cwd).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default from config_temperature.py).")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--date-range-confirmed", action="store_true",
                        help="Mark start-date/end-date as a confirmed project decision "
                             "rather than the assumed full-calendar-year default.")
    parser.add_argument("--depths", type=str, default=None,
                        help="Comma-separated requested depths in metres, e.g. '0,5,10,50'.")
    parser.add_argument("--depth-method", type=str, default=None, choices=["nearest", "exact"])
    parser.add_argument("--depth-tolerance", type=float, default=None,
                        help="Max |actual-requested| depth diff (m) before exclusion "
                             "(nearest method only).")
    parser.add_argument("--allow-missing-depths", action="store_true",
                        help="Allow the run to proceed even if not all requested depths "
                             "are matched (default: fail -- the target depth count is authoritative).")
    parser.add_argument("--allow-partial-time", action="store_true",
                        help="Allow incomplete temporal coverage instead of failing "
                             "(default: fail for an official run -- see config).")
    parser.add_argument("--max-implausible-fraction", type=float, default=None,
                        help="Fraction of finite values allowed outside the plausible "
                             "temperature range before the run fails (default 0.05).")
    parser.add_argument("--fill", action="store_true",
                        help="Fill NaN with config.fill_value in the saved output. OFF BY "
                             "DEFAULT and NOT RECOMMENDED: Step 5 computes its own mask via "
                             "np.isfinite() on this array, so filled cells would be silently "
                             "treated as valid observations downstream.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the optional diagnostic PNG.")
    parser.add_argument("--no-regrid", action="store_true",
                        help="Disable regridding and produce a NON-OFFICIAL native-grid "
                             "DIAGNOSTIC output (written to a separate native_diagnostic "
                             "subdirectory, clearly marked in its metadata). The official "
                             "OceanEmbed output is regridded to 0.25 deg by default; this "
                             "flag is for development/testing only.")
    parser.add_argument("--no-atomic", action="store_true",
                        help="Disable atomic temp-file writes (debugging only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect/validate/subset and report the would-be output shape, "
                             "but do not write temperature_target.nc/.npy/mask -- writes a "
                             "dryrun report instead.")
    parser.add_argument("--log-level", type=str, default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> TemperatureConfig:
    overrides: Dict[str, object] = {}
    simple_map = {
        "input": "explicit_input_path", "input_glob": "explicit_input_glob",
        "project_root": "project_root", "output_dir": "output_dir",
        "lat_min": "lat_min", "lat_max": "lat_max", "lon_min": "lon_min", "lon_max": "lon_max",
        "start_date": "start_date", "end_date": "end_date",
        "depth_method": "depth_selection_method", "depth_tolerance": "depth_max_tolerance_m",
        "max_implausible_fraction": "max_implausible_fraction", "log_level": "log_level",
    }
    for arg_name, field_name in simple_map.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.depths is not None:
        overrides["requested_depths_m"] = [float(x) for x in args.depths.split(",") if x.strip()]
    if args.date_range_confirmed:
        overrides["date_range_confirmed"] = True
    if args.allow_missing_depths:
        overrides["require_all_requested_depths"] = False
    if args.allow_partial_time:
        overrides["require_complete_temporal_coverage"] = False
    if args.fill:
        overrides["fill_enabled"] = True
    if args.no_plot:
        overrides["save_diagnostic_plot"] = False
    if args.no_regrid:
        overrides["regrid_to_quarter_degree"] = False
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
        logger.info("  %-20s dims=%-35s dtype=%s units=%s",
                    v, str(da.dims), da.dtype, da.attrs.get("units"))
    for k, v in ds.attrs.items():
        logger.info("Global attr: %s = %s", k, v)


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_outputs(
    da: xr.DataArray,
    mask: np.ndarray,
    coords: DetectedCoords,
    config: TemperatureConfig,
    metadata: Dict[str, object],
    out_dir: Path,
) -> Dict[str, str]:
    """Write temperature_target.nc (+ optional .npy / mask .npy) and the
    metadata JSON to out_dir (official output_dir, or
    output_dir/native_diagnostic_subdir for a non-official diagnostic
    run -- decided by the caller), atomically if config.atomic_writes."""
    out_dir.mkdir(parents=True, exist_ok=True)

    values = da.values
    if config.fill_enabled:
        saved_values = np.where(mask, values, config.fill_value).astype(np.float32)
        fill_note = (f"NaN filled with {config.fill_value} (fill_enabled=True); "
                    f"valid_mask distinguishes real vs filled cells. NOTE: Step 5's own "
                    f"masking (np.isfinite on this array) will NOT see this distinction.")
    else:
        saved_values = values.astype(np.float32)
        fill_note = "NaN preserved for land/invalid cells (fill_enabled=False); see valid_mask."

    assert mask.shape == saved_values.shape, (
        f"Internal error: mask shape {mask.shape} != value shape {saved_values.shape}."
    )

    out_ds = xr.Dataset(
        {
            "temperature_target": (da.dims, saved_values),
            "valid_mask": (da.dims, mask.astype(np.uint8)),
        },
        coords={
            coords.time: da[coords.time],
            coords.depth: da[coords.depth],
            coords.latitude: da[coords.latitude],
            coords.longitude: da[coords.longitude],
        },
    )
    out_ds["temperature_target"].attrs["units"] = metadata.get("units") or "degrees_C (UNVERIFIED - see warnings)"
    out_ds["temperature_target"].attrs["source_variable"] = metadata["variable_name"]
    out_ds["temperature_target"].attrs["missing_value_handling"] = fill_note
    out_ds["valid_mask"].attrs["description"] = (
        "1 = valid ocean observation, 0 = NaN/Inf (land or missing) in the source file"
    )
    out_ds[coords.depth].attrs.setdefault("units", "m")
    out_ds[coords.latitude].attrs.setdefault("units", "degrees_north")
    out_ds[coords.longitude].attrs.setdefault("units", "degrees_east")
    out_ds.attrs["source_dataset"] = "GLORYS (Copernicus Marine)"
    out_ds.attrs["source_variable"] = "thetao"
    out_ds.attrs["source_file"] = str(metadata["input_files"])
    out_ds.attrs["processing_script"] = "preprocess_temperature.py"
    out_ds.attrs["pipeline_version"] = PIPELINE_VERSION
    out_ds.attrs["processed_at_utc"] = metadata["processed_at_utc"]
    out_ds.attrs["dimension_order"] = " -> ".join(da.dims)
    out_ds.attrs["spatial_resolution_status"] = metadata["spatial_resolution_status"]
    out_ds.attrs["is_official_output"] = "true" if metadata.get("is_official_output") else "FALSE - NON-OFFICIAL NATIVE-GRID DIAGNOSTIC"
    out_ds.attrs["regridding_method"] = metadata.get("regridding", {}).get("method", "none (native-grid diagnostic)")
    out_ds.attrs["extrapolation_policy"] = metadata.get("regridding", {}).get(
        "extrapolation_policy", "n/a (not regridded)"
    )
    out_ds.attrs["depth_selection_method"] = config.depth_selection_method
    # Full requested-vs-actual depth mapping recorded directly in the
    # NetCDF's own metadata (JSON-encoded string; netCDF attrs are
    # scalar/string), not just the separate metadata JSON -- this is
    # what "recorded in ... output NetCDF metadata" means concretely.
    depth_report_list = metadata.get("requested_vs_actual_depths", [])
    out_ds.attrs["depth_selection_report_json"] = json.dumps(depth_report_list)
    surface_mapped = [d for d in depth_report_list if d.get("status") == "SURFACE_MAPPED"]
    if surface_mapped:
        out_ds.attrs["surface_depth_mapping"] = "; ".join(
            f"nominal {d['requested_m']}m -> actual {d['actual_m']}m (shallowest available GLORYS "
            f"level; allow_surface_depth_mapping=True, surface_depth_max_m={config.surface_depth_max_m})"
            for d in surface_mapped
        )
    else:
        out_ds.attrs["surface_depth_mapping"] = "not applicable (no surface depth mapping occurred this run)"
    out_ds.attrs["requested_lat_bounds"] = f"{config.lat_min} to {config.lat_max}"
    out_ds.attrs["requested_lon_bounds"] = f"{config.lon_min} to {config.lon_max}"
    fg = metadata.get("final_grid", {})
    out_ds.attrs["actual_lat_bounds"] = f"{fg.get('latitude_min')} to {fg.get('latitude_max')}"
    out_ds.attrs["actual_lon_bounds"] = f"{fg.get('longitude_min')} to {fg.get('longitude_max')}"

    written: Dict[str, str] = {}

    encoding = None
    if config.enable_netcdf_compression:
        comp = {"zlib": True, "complevel": config.netcdf_complevel}
        encoding = {
            "temperature_target": {**comp, "dtype": "float32"},
            "valid_mask": {**comp, "dtype": "uint8"},
        }

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

    return written


def make_diagnostic_plot(da: xr.DataArray, coords: DetectedCoords, config: TemperatureConfig) -> Optional[str]:
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
        values = da.values  # (time, depth, lat, lon)
        depth_vals = da[coords.depth].values
        lat_vals = da[coords.latitude].values
        lon_vals = da[coords.longitude].values

        finite_counts = np.isfinite(values[0]).sum(axis=0)
        if finite_counts.max() == 0:
            logger.warning("Skipping diagnostic plot: no valid (finite) values at the first timestep.")
            return None
        iy, ix = np.unravel_index(np.argmax(finite_counts), finite_counts.shape)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

        im0 = axes[0].pcolormesh(lon_vals, lat_vals, values[0, 0], shading="auto")
        axes[0].set_title(f"t=0, depth={depth_vals[0]:.1f} m")
        axes[0].set_xlabel("longitude")
        axes[0].set_ylabel("latitude")
        fig.colorbar(im0, ax=axes[0], label="deg C (as stored)")

        deep_idx = len(depth_vals) - 1
        im1 = axes[1].pcolormesh(lon_vals, lat_vals, values[0, deep_idx], shading="auto")
        axes[1].set_title(f"t=0, depth={depth_vals[deep_idx]:.1f} m")
        axes[1].set_xlabel("longitude")
        axes[1].set_ylabel("latitude")
        fig.colorbar(im1, ax=axes[1], label="deg C (as stored)")

        profile = values[0, :, iy, ix]
        axes[2].plot(profile, depth_vals, marker="o")
        axes[2].invert_yaxis()
        axes[2].set_title(f"Depth profile at lat={lat_vals[iy]:.2f}, lon={lon_vals[ix]:.2f}")
        axes[2].set_xlabel("deg C (as stored)")
        axes[2].set_ylabel("depth (m)")

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

    logger.info("GLORYS temperature-target preprocessing (SIH26066 / OceanEmbed) v%s", PIPELINE_VERSION)
    logger.info("ACTIVE dataset: 2025. (Jan-2026 prototype is discontinued and not searched by default.)")
    if config.regrid_to_quarter_degree:
        logger.info("Mode: OFFICIAL -- output will be regridded to %s deg and validated as such (fatal on any regridding failure).",
                    config.target_resolution_deg)
    else:
        logger.warning("Mode: NON-OFFICIAL NATIVE-GRID DIAGNOSTIC (--no-regrid) -- output written to "
                    "'%s/' and clearly marked; this is NOT the official model-ready output.",
                    config.native_diagnostic_subdir)
    logger.info("project_root=%s  dry_run=%s", Path(config.project_root).resolve(), config.dry_run)
    logger.info("Region: lat[%s,%s] lon[%s,%s], time[%s,%s]%s",
                config.lat_min, config.lat_max, config.lon_min, config.lon_max,
                config.start_date, config.end_date,
                "" if config.date_range_confirmed else "  (ASSUMED full-year default, NOT confirmed -- see date_range_confirmed)")
    logger.info("Requested depths (m, %d values, authoritative -- require_all_requested_depths=%s): %s",
                len(config.requested_depths_m), config.require_all_requested_depths, config.requested_depths_m)
    if not config.date_range_confirmed:
        logger.warning(
            "start_date/end_date (%s..%s) are an ASSUMED full-calendar-year default, not a "
            "confirmed project decision -- the brief stated '2025 ocean data' without exact "
            "bounds. Confirm and pass --date-range-confirmed (or edit config) once known.",
            config.start_date, config.end_date,
        )

    report: Dict[str, object] = {
        "pipeline_version": PIPELINE_VERSION,
        "dry_run": config.dry_run,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_summary": {
            "requested_depths_m": config.requested_depths_m,
            "region": [config.lat_min, config.lat_max, config.lon_min, config.lon_max],
            "requested_time_range": [config.start_date, config.end_date],
            "date_range_confirmed": config.date_range_confirmed,
            "require_all_requested_depths": config.require_all_requested_depths,
            "require_complete_temporal_coverage": config.require_complete_temporal_coverage,
            "regrid_to_quarter_degree": config.regrid_to_quarter_degree,
        },
    }
    # Computed once, from CLI-known config alone (before any file is even
    # found), and reused everywhere a report gets written in this run --
    # official runs and --no-regrid diagnostic runs never share a
    # directory, including for early failure/dry-run reports.
    run_out_dir = config.resolve(
        config.output_dir if config.regrid_to_quarter_degree else f"{config.output_dir}/{config.native_diagnostic_subdir}"
    )

    paths, source_label = find_input_files(config)
    report["input_files"] = [str(p) for p in paths]
    report["input_file_source"] = source_label
    if not paths:
        logger.error(
            "No active 2025 input file found (tried year2025_input_candidates=%s, glob '%s', "
            "resolved against project_root=%s). The discontinued Jan-2026 prototype paths are "
            "deliberately NOT used as a fallback.",
            config.year2025_input_candidates, config.year2025_input_glob,
            Path(config.project_root).resolve(),
        )
        logger.error(
            "This is expected if the 2025 data isn't in this project layout yet -- re-run once "
            "it is, or pass --input <path> / --input-glob <pattern> to point at it directly."
        )
        report["error"] = "no_input_file_found"
        report_name = config.dryrun_report_name if config.dry_run else "temperature_error_report.json"
        _write_json_report(run_out_dir / report_name, report, config)
        return 2
    logger.info("Using %d input file(s) (%s): %s", len(paths), source_label, [str(p) for p in paths])

    try:
        ds = open_dataset(paths, use_dask_chunking=config.use_dask_chunking)
    except (FileNotFoundError, OSError) as exc:
        logger.error(str(exc))
        report["error"] = str(exc)
        _write_json_report(run_out_dir / "temperature_error_report.json", report, config)
        return 2

    try:
        print_dataset_summary(ds, paths)

        var_name = detect_variable(ds, config.temperature_var_candidates)
        coords = detect_all_coords(ds, config)
        report["variable_name"] = var_name
        report["detected_coordinate_names"] = {
            "time": coords.time, "depth": coords.depth,
            "latitude": coords.latitude, "longitude": coords.longitude,
        }
        logger.info("Detected temperature variable: '%s'", var_name)
        logger.info("Detected coordinates: time='%s' depth='%s' latitude='%s' longitude='%s'",
                    coords.time, coords.depth, coords.latitude, coords.longitude)

        expected_dims = {coords.time, coords.depth, coords.latitude, coords.longitude}
        actual_dims = set(ds[var_name].dims)
        if actual_dims != expected_dims:
            raise ValueError(
                f"Variable '{var_name}' has dims {ds[var_name].dims}, expected exactly "
                f"(time, depth, latitude, longitude) = {sorted(expected_dims)}. If this "
                f"extra/missing dimension is real and expected, handle it explicitly here "
                f"rather than letting it pass silently."
            )

        coord_report = validate_coordinates(ds, coords)
        report["coordinate_validation"] = coord_report
        all_warnings: List[str] = list(coord_report.get("warnings", []))
        logger.info("Native grid: %d lat x %d lon (~%.4f x %.4f deg), %d depth levels, %d time steps",
                    coord_report["n_latitude"], coord_report["n_longitude"],
                    _safe(coord_report["latitude_resolution_deg"]), _safe(coord_report["longitude_resolution_deg"]),
                    coord_report["n_depth"], coord_report["n_time"])
        logger.info("Time range: %s to %s (daily: %s, missing dates: %d, duplicates: %s)",
                    coord_report["first_timestamp"], coord_report["last_timestamp"],
                    coord_report["is_daily"], len(coord_report["missing_dates"]),
                    coord_report["time_has_duplicate_timestamps"])
        logger.info("Available depths (m): %s", coord_report["available_depths_m"])
        logger.info("Longitude convention detected: %s", coord_report["longitude_convention"])

        original_dims = dict(ds[var_name].sizes)
        report["original_dimensions"] = original_dims

        subset_ds, region_report = subset_region_time(ds, coords, config)
        report["requested_region_time"] = region_report
        all_warnings.extend(region_report.get("warnings", []))

        da = subset_ds[var_name].transpose(coords.time, coords.depth, coords.latitude, coords.longitude)

        depth_da, depth_report = select_depths(da, coords, config)
        depth_da = depth_da.transpose(coords.time, coords.depth, coords.latitude, coords.longitude)
        report["requested_vs_actual_depths"] = depth_report

        # --- Phase 3 strict-mode checks: fail clearly, never silently ship
        # an incomplete depth set or an incomplete time range as "complete".
        verify_depth_count(depth_report, config)
        verify_temporal_coverage(coord_report, region_report, config)

        # --- Phase 3 strict-mode checks: fail clearly, never silently ship
        # an incomplete depth set or an incomplete time range as "complete".
        verify_depth_count(depth_report, config)
        verify_temporal_coverage(coord_report, region_report, config)

        units, unit_warnings = check_units(ds[var_name])
        all_warnings.extend(unit_warnings)

        # --- Physical-value safety check happens BEFORE regridding, on
        # purpose: an unmasked fill-value sentinel (e.g. 1e20) would
        # otherwise get smeared into real neighbouring ocean cells by
        # linear interpolation, corrupting them before we ever get a
        # chance to catch it.
        missing_pre_regrid = inspect_missing_values(depth_da.values)
        implausible_pre_regrid = flag_implausible_values(depth_da.values, config.implausible_low_c, config.implausible_high_c)
        if implausible_pre_regrid["n_implausible"] > 0:
            all_warnings.append(
                f"{implausible_pre_regrid['n_implausible']} finite value(s) fall outside the plausible "
                f"[{config.implausible_low_c}, {config.implausible_high_c}] deg C range (checked pre-regrid)."
            )
        check_implausible_fraction(implausible_pre_regrid, missing_pre_regrid, config)

        # --- Regridding: OFFICIAL and default-on. FATAL BY DESIGN on
        # any failure -- deliberately NO try/except-and-continue here.
        # A run with regrid_to_quarter_degree=False (--no-regrid) is an
        # explicit, clearly-marked non-official native-grid DIAGNOSTIC
        # output, routed to a different subdirectory below so it can
        # never be mistaken for the official 0.25 deg output.
        regrid_report: Dict[str, object] = {}
        if config.regrid_to_quarter_degree:
            depth_da, regrid_report = regrid_to_target_grid(depth_da, coords, config)
            depth_da = depth_da.transpose(coords.time, coords.depth, coords.latitude, coords.longitude)
            all_warnings.extend(regrid_report.get("warnings", []))
            spatial_status = (f"regridded to the official {config.target_resolution_deg} deg target grid "
                            f"via {config.regrid_method} interpolation, no extrapolation")
            is_official_output = True
        else:
            spatial_status = "NATIVE GLORYS RESOLUTION -- NON-OFFICIAL DIAGNOSTIC OUTPUT (regridding explicitly disabled)"
            all_warnings.append(
                "regrid_to_quarter_degree=False: this is a non-official native-grid DIAGNOSTIC "
                "output, written to native_diagnostic_subdir, NOT the official 0.25 deg model-ready output."
            )
            is_official_output = False
            logger.warning(spatial_status)

        # Post-regrid (or post-depth-selection, in diagnostic mode)
        # stats -- these describe the array actually being shipped, not
        # the pre-regrid native one, so they're recomputed here rather
        # than reusing the pre-regrid numbers above.
        missing_final = inspect_missing_values(depth_da.values)
        implausible_final = flag_implausible_values(depth_da.values, config.implausible_low_c, config.implausible_high_c)

        mask = compute_valid_mask(depth_da.values)

        # --- Standardize coordinate names for Step 5 compatibility
        # (verified against oceanembed_step5/config.py's CoordNames
        # defaults: time/latitude/longitude/depth).
        depth_da, coords, rename_map = standardize_output_coordinates(depth_da, coords, config)
        if rename_map:
            logger.info("Standardized output coordinate names: %s", rename_map)
            all_warnings.append(f"Renamed coordinates for Step 5 compatibility: {rename_map}")
        report["output_coordinate_rename_map"] = rename_map

        # --- Explicit final-grid validation before writing anything:
        # calculated from the resulting coordinates, not trusted from
        # config. Raises clearly (depth count/ordering always; grid
        # spacing/range only when regrid_to_quarter_degree=True).
        final_grid_report = verify_final_grid(depth_da, coords, config)
        report["final_grid"] = final_grid_report
        logger.info(
            "Final grid: lat[%.4f,%.4f] n=%d spacing=%.4fdeg  lon[%.4f,%.4f] n=%d spacing=%.4fdeg",
            final_grid_report["latitude_min"], final_grid_report["latitude_max"], final_grid_report["n_latitude"],
            _safe(final_grid_report["latitude_spacing_deg"]),
            final_grid_report["longitude_min"], final_grid_report["longitude_max"], final_grid_report["n_longitude"],
            _safe(final_grid_report["longitude_spacing_deg"]),
        )

        out_dir = run_out_dir
        expected_shape = tuple(depth_da.sizes[d] for d in (coords.time, coords.depth, coords.latitude, coords.longitude))
        output_files = {
            "netcdf": str(out_dir / config.output_nc_name),
            "npy_values": str(out_dir / config.output_npy_name) if config.write_npy_outputs else None,
            "npy_mask": str(out_dir / config.output_mask_npy_name) if config.write_npy_outputs else None,
            "metadata_json": str(out_dir / config.output_metadata_name),
            "diagnostic_plot": str(out_dir / config.diagnostic_plot_name) if config.save_diagnostic_plot else None,
        }

        metadata: Dict[str, object] = {
            **report,
            "output_files": output_files,
            "is_official_output": is_official_output,
            "processed_dimensions": dict(depth_da.sizes),
            "dimension_order": list(depth_da.dims),
            "expected_output_shape": list(expected_shape),
            "units": units,
            "spatial_resolution_status": spatial_status,
            "regridding": regrid_report,
            "final_grid": final_grid_report,
            "temperature_stats_finite_only_pre_regrid": implausible_pre_regrid,
            "temperature_stats_finite_only_final": implausible_final,
            "missing_values_pre_regrid": missing_pre_regrid,
            "missing_values_final": missing_final,
            "valid_mask_cell_count": int(mask.sum()),
            "invalid_masked_cell_count": int(mask.size - mask.sum()),
            "n_time_steps": int(depth_da.sizes[coords.time]),
            "n_depth_levels": int(depth_da.sizes[coords.depth]),
            "fill_enabled": config.fill_enabled,
            "step5_target_variable_expected": "temperature_target",
            "step5_target_paths_default": "data/temperature_SIH15_depths.nc (STALE -- see final report)",
            "warnings": all_warnings,
        }

        logger.info("=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        logger.info("Input(s):  %s", metadata["input_files"])
        logger.info("Variable: %s  units=%s", var_name, units)
        logger.info("Original dims: %s -> Processed dims: %s", original_dims, metadata["processed_dimensions"])
        logger.info("Selected depths (m): %s -- count=%d (must be %d)",
                    [d["actual_m"] for d in depth_report if d["actual_m"] is not None],
                    metadata["n_depth_levels"], len(config.requested_depths_m))
        logger.info("Temperature (finite values, final array): min=%.3f max=%.3f mean=%.3f",
                    _safe(implausible_final["min_finite"]), _safe(implausible_final["max_finite"]), _safe(implausible_final["mean_finite"]))
        logger.info("Valid cells: %d / %d (%.2f%% invalid/masked)",
                    metadata["valid_mask_cell_count"], mask.size,
                    100.0 * metadata["invalid_masked_cell_count"] / mask.size if mask.size else 0.0)
        logger.info("Spatial resolution: %s", spatial_status)
        if all_warnings:
            logger.warning("%d warning(s):", len(all_warnings))
            for w in all_warnings:
                logger.warning("  - %s", w)
        else:
            logger.info("No warnings.")

        if config.dry_run:
            metadata["note"] = "DRY RUN: no temperature_target.nc/.npy/mask written. This is the structure/validation that WOULD be produced."
            dryrun_path = out_dir / config.dryrun_report_name
            _write_json_report(dryrun_path, metadata, config)
            logger.info("DRY RUN complete -- no processed outputs written. Report: %s", dryrun_path)
            logger.info("Would-be output shape (time, depth, lat, lon): %s", expected_shape)
            return 0

        written = save_outputs(depth_da, mask, coords, config, metadata, out_dir)
        logger.info("Wrote outputs:")
        for k, v in written.items():
            logger.info("  %s: %s", k, v)

        plot_path = make_diagnostic_plot(depth_da, coords, config)
        if plot_path:
            logger.info("  diagnostic_plot: %s", plot_path)

        # --- Output self-verification: reopen what was just written and
        # check it, rather than assuming to_netcdf() succeeding means the
        # file is actually correct.
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
        _write_json_report(run_out_dir / "temperature_error_report.json", report, config)
        return 1
    finally:
        ds.close()


if __name__ == "__main__":
    sys.exit(main())
