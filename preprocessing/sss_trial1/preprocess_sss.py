"""
preprocess_sss.py
==================
SIH26066 / OceanEmbed -- Sea Surface Salinity (SSS) preprocessing,
2025 internal round only.

Pipeline (mirrors preprocess_temperature.py's stage order):
  1. Resolve input file (config default or --input)
  2. Open (lazy, CF-decoded) + log dataset summary
  3. Detect variable ('sos') and coordinate names
  4. Validate coordinates (numeric, sorted, no NaN, daily cadence)
  5. Validate the singleton surface depth -> drop it
  6. Subset to the requested region/time window
  7. Missing-value + implausible-value inspection (pre-regrid)
  8. Unit check (report only, never relabel)
  9. Regrid: weighted_linear onto the official 0.25 deg / 101x241 grid
 10. Missing-value inspection (post-regrid) + final-grid verification
 11. --dry-run stops here (writes a dry-run report only)
 12. Write outputs (NetCDF + optional .npy + metadata JSON), atomically
 13. Reopen and validate the written NetCDF from disk
 14. Confirm the raw input file's fingerprint is unchanged
 15. Diagnostic plot
 16. Summarize

Usage (from the project root, PowerShell):
  python .\\preprocessing\\sss_preprocess_trial1\\preprocess_sss.py `
    --input ".\\data\\raw\\sss\\test\\2025\\sss_2025.nc" `
    --output-dir ".\\data\\processed\\sss\\2025" `
    --dry-run
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from config_sss import CONFIG, PIPELINE_VERSION, SSSConfig
import netcdf_utils as ncu

logger = logging.getLogger("preprocess_sss")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH26066 OceanEmbed -- SSS preprocessing (2025 internal round). "
                    "Regrids native ~0.125 deg CNR SSS ('sos') onto the official 0.25 deg "
                    "OceanEmbed grid using NaN-aware weighted-linear interpolation."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to the raw SSS NetCDF file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for processed outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report; write nothing.")
    parser.add_argument(
        "--region", type=str, default=None,
        help="Override region as 'lat_min,lat_max,lon_min,lon_max' (degrees). Default: 5,30,45,105.",
    )
    parser.add_argument(
        "--target-resolution", type=float, default=None,
        help="Override target grid resolution in degrees. Default: 0.25.",
    )
    parser.add_argument(
        "--date-range", type=str, default=None,
        help="Override time window as 'YYYY-MM-DD,YYYY-MM-DD'. Default: 2025-01-01,2025-12-31.",
    )
    parser.add_argument(
        "--regrid-method", type=str, default=None, choices=["weighted_linear"],
        help="Regridding method. Only 'weighted_linear' is implemented (see config_sss.py for why).",
    )
    parser.add_argument(
        "--min-valid-weight", type=float, default=None,
        help="Minimum fraction of a target cell's interpolation weight that must come from "
             "finite source cells for that cell to be marked valid. Default: 0.5.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output NetCDF.")
    parser.add_argument("--allow-partial-time", action="store_true", help="Do not fail on incomplete 2025 coverage.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the diagnostic plot.")
    parser.add_argument("--no-npy", action="store_true", help="Skip writing .npy companion outputs.")
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SSSConfig:
    config = SSSConfig()
    if args.input:
        config.explicit_input_path = args.input
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.region:
        try:
            lat_min, lat_max, lon_min, lon_max = (float(x) for x in args.region.split(","))
        except ValueError as exc:
            raise ValueError(f"--region must be 'lat_min,lat_max,lon_min,lon_max', got {args.region!r}.") from exc
        config.lat_min, config.lat_max, config.lon_min, config.lon_max = lat_min, lat_max, lon_min, lon_max
    if args.target_resolution is not None:
        config.target_resolution_deg = args.target_resolution
    if args.date_range:
        try:
            start, end = args.date_range.split(",")
        except ValueError as exc:
            raise ValueError(f"--date-range must be 'YYYY-MM-DD,YYYY-MM-DD', got {args.date_range!r}.") from exc
        config.start_date, config.end_date = start.strip(), end.strip()
        config.date_range_confirmed = False  # an override is no longer the brief-confirmed range
        config.expected_time_step_count = None  # let the pipeline infer/report it instead of enforcing 365
    if args.regrid_method:
        config.regrid_method = args.regrid_method
    if args.min_valid_weight is not None:
        config.min_valid_weight = args.min_valid_weight
    if args.overwrite:
        config.overwrite = True
    if args.allow_partial_time:
        config.require_complete_temporal_coverage = False
    if args.no_plot:
        config.save_diagnostic_plot = False
    if args.no_npy:
        config.write_npy_outputs = False
    config.dry_run = args.dry_run
    config.log_level = args.log_level
    config.__post_init__()
    return config


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


# =============================================================================
# Helpers
# =============================================================================

def _lib_versions() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in ("xarray", "numpy", "scipy", "pandas", "netCDF4", "h5netcdf"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def print_dataset_summary(ds: xr.Dataset, input_path: Path) -> None:
    logger.info("=" * 70)
    logger.info("INPUT: %s", input_path)
    try:
        size_mb = input_path.stat().st_size / (1024 * 1024)
        logger.info("Input file size: %.1f MB", size_mb)
    except OSError:
        pass
    logger.info("Dimensions: %s", dict(ds.sizes))
    logger.info("Data variables: %s", list(ds.data_vars))
    logger.info("Coordinates: %s", list(ds.coords))
    logger.info("Global attrs (subset): %s", {k: ds.attrs.get(k) for k in
                 ("Conventions", "title", "institution", "product_version") if k in ds.attrs})
    logger.info("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)

    run_started_at = datetime.now(timezone.utc)
    logger.info("SSS preprocessing pipeline v%s starting (dry_run=%s)", PIPELINE_VERSION, config.dry_run)

    # ---- 1-2. Resolve + open input -----------------------------------
    input_path, source_label = ncu.find_input_file(config)
    if input_path is None:
        logger.error("Could not locate an input file. %s", source_label)
        return 2
    logger.info("Using input file (%s)", source_label)
    raw_fingerprint_before = ncu.fingerprint_file(input_path)

    ds = ncu.open_dataset(input_path, use_dask_chunking=config.use_dask_chunking)
    try:
        print_dataset_summary(ds, input_path)

        # ---- 3. Detect variable + coordinates -------------------------
        var_name = ncu.detect_variable(ds, config.sss_var_candidates)
        coords4 = ncu.detect_all_coords(ds, config)
        logger.info("Detected SSS variable: '%s'", var_name)
        logger.info(
            "Detected coordinates: time='%s' depth='%s' latitude='%s' longitude='%s'",
            coords4.time, coords4.depth, coords4.latitude, coords4.longitude,
        )
        da_full = ds[var_name]
        logger.info("Variable attrs: %s", dict(da_full.attrs))
        logger.info("Variable encoding (post-CF-decode): %s", {
            k: da_full.encoding.get(k) for k in ("_FillValue", "scale_factor", "add_offset", "dtype")
        })

        # ---- 4. Coordinate validation -----------------------------------
        coord_report = ncu.validate_coordinates(ds, coords4)
        for w in coord_report["warnings"]:
            logger.warning(w)
        logger.info(
            "Source grid: lat[%.4f, %.4f] step=%.4f deg (%d pts, regular=%s) | "
            "lon[%.4f, %.4f] step=%.4f deg (%d pts, regular=%s)",
            coord_report["latitude_range"][0], coord_report["latitude_range"][1],
            coord_report["latitude_resolution_deg"], coord_report["n_latitude"], coord_report["latitude_grid_regular"],
            coord_report["longitude_range"][0], coord_report["longitude_range"][1],
            coord_report["longitude_resolution_deg"], coord_report["n_longitude"], coord_report["longitude_grid_regular"],
        )
        logger.info(
            "Time coverage: %s .. %s (%d steps, daily=%s, missing_dates=%d)",
            coord_report["first_timestamp"], coord_report["last_timestamp"], coord_report["n_time"],
            coord_report["is_daily"], len(coord_report["missing_dates"]),
        )
        if config.date_range_confirmed:
            if coord_report["first_timestamp"] is None or not str(coord_report["first_timestamp"]).startswith(config.start_date):
                logger.warning(
                    "date_range_confirmed=True but file's first timestamp (%s) does not start with "
                    "the confirmed start_date (%s) -- the file may not be the brief-inspected 2025 file.",
                    coord_report["first_timestamp"], config.start_date,
                )

        # ---- 5. Depth validation + drop ---------------------------------
        depth_report = ncu.validate_singleton_depth(ds, coords4, config)
        logger.info(
            "Depth validated: %s m (expected %s m, diff=%.2e m) -- confirmed surface.",
            depth_report["depth_value_m"], depth_report["expected_surface_depth_m"],
            depth_report["difference_from_expected_m"],
        )

        # ---- 6. Region/time subset --------------------------------------
        subset_ds, region_report = ncu.subset_region_time(ds, coords4, config)
        for w in region_report["warnings"]:
            logger.info(w)
        ncu.verify_temporal_coverage(coord_report, region_report, config)

        da4 = subset_ds[var_name].isel({coords4.depth: 0})
        coords3 = ncu.DetectedCoords3D(time=coords4.time, latitude=coords4.latitude, longitude=coords4.longitude)
        logger.info("Dropped singleton depth dimension. Shape now: %s %s", da4.dims, da4.shape)

        # ---- 7. Missing-value + implausible-value inspection (pre-regrid)
        pre_missing = ncu.inspect_missing_values(da4)
        logger.info(
            "PRE-REGRID missing-value stats: %d/%d valid (%.4f%%), %.4f%% invalid.",
            pre_missing["valid_count"], pre_missing["total_cells"],
            100 * pre_missing["fraction_valid"], 100 * pre_missing["fraction_invalid"],
        )
        pre_by_time = ncu.inspect_missing_by_time(da4, coords3)
        logger.info(
            "Per-timestep valid fraction: constant across all timesteps=%s (min=%.5f%%, max=%.5f%%), "
            "completely-missing timesteps=%d",
            pre_by_time["valid_fraction_constant_across_time"],
            100 * (pre_by_time["valid_fraction_min"] or 0), 100 * (pre_by_time["valid_fraction_max"] or 0),
            len(pre_by_time["completely_missing_times"]),
        )
        if pre_by_time["completely_missing_times"]:
            logger.warning("Completely-missing timesteps: %s", pre_by_time["completely_missing_times"][:10])

        implausible_report = ncu.flag_implausible_values(da4, config.implausible_low, config.implausible_high)
        logger.info(
            "Finite value range: [%.4f, %.4f], mean=%.4f, implausible-count=%d",
            implausible_report["min_finite"] or float("nan"), implausible_report["max_finite"] or float("nan"),
            implausible_report["mean_finite"] or float("nan"), implausible_report["n_implausible"],
        )
        ncu.check_implausible_fraction(implausible_report, pre_missing, config)

        # ---- 8. Unit check -----------------------------------------------
        units_value, unit_warnings = ncu.check_units(da4)
        for w in unit_warnings:
            logger.warning(w)

        # ---- 9. Regridding -------------------------------------------------
        if not config.regrid_to_quarter_degree:
            raise ValueError("regrid_to_quarter_degree=False is not supported by this pipeline (explicit regridding is mandatory).")
        logger.info(
            "Regridding: %s -> target %.2f deg grid, method=%s, min_valid_weight=%.2f",
            f"{coord_report['latitude_resolution_deg']:.4f}deg x {coord_report['longitude_resolution_deg']:.4f}deg",
            config.target_resolution_deg, config.regrid_method, config.min_valid_weight,
        )
        regridded_da, mask_da, regrid_report = ncu.regrid_weighted_linear(da4, coords3, config)
        for w in regrid_report["warnings"]:
            logger.info(w)
        logger.info(
            "Target grid built: lat[%.2f,%.2f] n=%d step=%.4f | lon[%.2f,%.2f] n=%d step=%.4f",
            regrid_report["target_grid"]["lat_min"], regrid_report["target_grid"]["lat_max"],
            regrid_report["target_grid"]["n_lat"], regrid_report["target_grid"]["lat_spacing_deg"],
            regrid_report["target_grid"]["lon_min"], regrid_report["target_grid"]["lon_max"],
            regrid_report["target_grid"]["n_lon"], regrid_report["target_grid"]["lon_spacing_deg"],
        )
        logger.info("Extrapolation policy: %s", regrid_report["extrapolation_policy"])

        # ---- 10. Post-regrid inspection + final-grid verification --------
        post_missing = ncu.inspect_missing_values(regridded_da)
        logger.info(
            "POST-REGRID stats: %d/%d valid (%.4f%%), %.4f%% invalid.",
            post_missing["valid_count"], post_missing["total_cells"],
            100 * post_missing["fraction_valid"], 100 * post_missing["fraction_invalid"],
        )
        grid_report = ncu.verify_final_grid(regridded_da, coords3, config)
        logger.info(
            "Final grid verified: %d x %d, lat step=%.4f deg, lon step=%.4f deg.",
            grid_report["n_latitude"], grid_report["n_longitude"],
            grid_report["latitude_spacing_deg"], grid_report["longitude_spacing_deg"],
        )

        # ---- Standardize output coordinate names --------------------------
        regridded_da, mask_da, out_coords, rename_map = ncu.standardize_output_coordinates(
            regridded_da, mask_da, coords3, config
        )
        if rename_map:
            logger.info("Standardized coordinate names: %s", rename_map)

        expected_shape = (
            regrid_report["target_grid"]["n_lat"] if False else da4.sizes[coords3.time],
            grid_report["n_latitude"], grid_report["n_longitude"],
        )

        # ---- Assemble metadata (used by both dry-run report and real output)
        metadata = {
            "project_name": "OceanEmbed",
            "project_id": "SIH26066",
            "pipeline_stage": "sss_preprocessing",
            "pipeline_version": PIPELINE_VERSION,
            "processing_timestamp_utc": run_started_at.isoformat(),
            "source": {
                "file_path": str(input_path),
                "file_size_bytes": raw_fingerprint_before["size_bytes"],
                "variable_name": var_name,
                "variable_attrs": dict(da_full.attrs),
                "variable_encoding": {
                    k: da_full.encoding.get(k) for k in ("_FillValue", "scale_factor", "add_offset", "dtype")
                },
                "original_dimensions": dict(subset_ds.sizes),
                "native_resolution_deg": {
                    "latitude": coord_report["latitude_resolution_deg"],
                    "longitude": coord_report["longitude_resolution_deg"],
                },
                "native_coordinate_range": {
                    "latitude": coord_report["latitude_range"],
                    "longitude": coord_report["longitude_range"],
                },
                "depth": depth_report,
                "input_time_range": {
                    "requested_start": config.start_date,
                    "requested_end": config.end_date,
                    "actual_matched_range": region_report.get("actual_time_range"),
                    "date_range_confirmed_from_brief": config.date_range_confirmed,
                    "n_time_steps_matched": region_report["n_time_matched"],
                },
                "global_attrs_subset": {
                    k: ds.attrs.get(k) for k in
                    ("Conventions", "grid_resolution", "title", "institution", "product_version", "software_version")
                    if k in ds.attrs
                },
            },
            "units": {
                "source_units_attr": units_value,
                "unit_interpretation": (
                    "Source units attribute is preserved verbatim and NOT auto-relabeled as PSU or any "
                    "other unit. assume_psu_equivalent={} -- see README 'Unit handling' for the "
                    "unresolved interpretation question that must be confirmed by the project before "
                    "any downstream code assumes PSU equivalence."
                ).format(config.assume_psu_equivalent),
            },
            "target_grid": {
                "region": {"lat_min": config.lat_min, "lat_max": config.lat_max,
                            "lon_min": config.lon_min, "lon_max": config.lon_max},
                "resolution_deg": config.target_resolution_deg,
                **regrid_report["target_grid"],
            },
            "regridding": {
                "method": regrid_report["method"],
                "min_valid_weight": regrid_report["min_valid_weight"],
                "extrapolation_policy": regrid_report["extrapolation_policy"],
                "missing_data_handling": regrid_report["missing_data_handling"],
                "target_cells_invalid_after_regrid": regrid_report["target_cells_invalid_fraction"],
                "n_target_points_with_zero_source_support_any_time": regrid_report.get(
                    "n_target_points_with_zero_source_support_any_time"
                ),
                "boundary_note": (
                    "Native coverage (lat 5.0625-29.9375, lon 45.0625-104.9375) sits strictly inside "
                    "the requested region boundary (lat 5-30, lon 45-105) by half the native grid "
                    "spacing on each side. With extrapolation disabled, the outermost target row "
                    "(lat=5.00 and lat=30.00) and outermost target column (lon=45.00 and lon=105.00) "
                    "are expected to be entirely invalid/NaN -- this is deliberate conservative "
                    "behavior, not a defect."
                ),
            },
            "missing_data_policy": (
                "Missing/invalid source cells are NEVER filled with zero, a mean, or any fabricated "
                "value. They are preserved as NaN in 'sss' and marked 0 in 'sss_valid_mask'. A target "
                "cell is only marked valid (1) if its interpolation support (fraction of finite "
                "contributing source cells) meets or exceeds min_valid_weight."
            ),
            "pre_regrid_missing_value_stats": pre_missing,
            "pre_regrid_missing_by_time_summary": {
                "valid_fraction_constant_across_time": pre_by_time["valid_fraction_constant_across_time"],
                "valid_fraction_min": pre_by_time["valid_fraction_min"],
                "valid_fraction_max": pre_by_time["valid_fraction_max"],
                "completely_missing_times": pre_by_time["completely_missing_times"],
            },
            "post_regrid_missing_value_stats": post_missing,
            "implausible_value_check": implausible_report,
            "final_grid": grid_report,
            "validity_mask_meaning": {
                "sss_valid_mask": "1 = valid, finite, regridded SSS value with sufficient source support "
                                    "(interpolation weight >= min_valid_weight). 0 = invalid -- missing "
                                    "source data, land, outside native grid coverage (no extrapolation), "
                                    "or insufficient interpolation support."
            },
            "library_versions": _lib_versions(),
            "validation_status": "NOT YET VALIDATED (populated after reopen-and-validate step)",
            "config_used": {
                k: v for k, v in vars(config).items()
            },
        }

        # ---- 11. Dry run stop ---------------------------------------------
        if config.dry_run:
            logger.info("DRY RUN: no processed output written.")
            output_dir = config.resolve(config.output_dir)
            report_path = output_dir / config.dryrun_report_name
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(metadata, f, indent=2, default=_json_default)
            logger.info("Dry-run report written to: %s", report_path)
            logger.info(
                "Expected final output shape: (%d, %d, %d)",
                expected_shape[0], expected_shape[1], expected_shape[2],
            )
            logger.info("Dry run complete.")
            return 0

        # ---- 12. Write outputs ----------------------------------------------
        output_dir = config.resolve(config.output_dir)
        nc_path = output_dir / config.output_nc_name
        if nc_path.exists() and not config.overwrite:
            raise FileExistsError(
                f"{nc_path} already exists and --overwrite was not given. Refusing to silently "
                f"clobber an existing processed output."
            )

        out_ds = xr.Dataset(
            data_vars={
                "sss": regridded_da,
                "sss_valid_mask": mask_da,
            }
        )
        out_ds["sss"].attrs = {
            "long_name": "sea surface salinity, regridded to 0.25 deg OceanEmbed grid",
            "standard_name": "sea_surface_salinity",
            "units": units_value if units_value is not None else "unknown",
            "source_variable": var_name,
            "source_depth_m": depth_report["depth_value_m"],
            "regrid_method": regrid_report["method"],
            "min_valid_weight": config.min_valid_weight,
        }
        out_ds["sss_valid_mask"].attrs = {
            "long_name": "SSS regridded-value validity mask",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "invalid valid",
            "description": metadata["validity_mask_meaning"]["sss_valid_mask"],
        }
        out_ds.attrs = {
            "title": "OceanEmbed SSS preprocessed (2025, 0.25 deg)",
            "project_id": "SIH26066",
            "project_name": "OceanEmbed",
            "pipeline_version": PIPELINE_VERSION,
            "source_file": str(input_path),
            "processing_timestamp_utc": run_started_at.isoformat(),
            "Conventions": "CF-1.7",
        }

        encoding = {
            "sss": {"_FillValue": np.nan, "dtype": "float32"},
            "sss_valid_mask": {"dtype": "uint8"},
        }
        if config.enable_netcdf_compression:
            encoding["sss"].update({"zlib": True, "complevel": config.netcdf_complevel})
            encoding["sss_valid_mask"].update({"zlib": True, "complevel": config.netcdf_complevel})

        def _write_nc(tmp_path: Path) -> None:
            out_ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding or None)

        logger.info("Writing processed NetCDF to: %s", nc_path)
        ncu.atomic_write(nc_path, _write_nc)

        if config.write_npy_outputs:
            npy_path = output_dir / config.output_npy_name
            mask_npy_path = output_dir / config.output_mask_npy_name
            logger.info("Writing companion .npy arrays: %s, %s", npy_path, mask_npy_path)
            ncu.atomic_write(npy_path, lambda p: np.save(p, out_ds["sss"].values))
            ncu.atomic_write(mask_npy_path, lambda p: np.save(p, out_ds["sss_valid_mask"].values))

        # ---- 13. Reopen and validate ------------------------------------
        logger.info("Reopening saved output to validate from disk...")
        validation_report = ncu.verify_output_file(nc_path, config, expected_shape)
        metadata["validation_status"] = "PASSED: reopened output matches expected structure." if validation_report.get("ok") else "FAILED"
        logger.info("Reopen validation PASSED: dims=%s shape=%s dtype=%s mask_consistent=%s",
                    validation_report["reopened_dims"], validation_report["reopened_shape"],
                    validation_report["reopened_dtype"], validation_report["mask_consistent_with_isfinite"])

        # ---- 14. Raw-file fingerprint check -----------------------------
        raw_fingerprint_after = ncu.fingerprint_file(input_path)
        fp_warnings = ncu.verify_file_unchanged(raw_fingerprint_before, raw_fingerprint_after)
        if fp_warnings:
            for w in fp_warnings:
                logger.error(w)
        else:
            logger.info("Confirmed: raw input file unchanged (size and mtime match pre-run fingerprint).")
        metadata["raw_input_file_unchanged"] = (len(fp_warnings) == 0)

        # ---- Metadata JSON (final, with validation results) -------------
        metadata_path = output_dir / config.output_metadata_name
        ncu.atomic_write(metadata_path, lambda p: p.write_text(json.dumps(metadata, indent=2, default=_json_default)))
        logger.info("Metadata written to: %s", metadata_path)

        # ---- 15. Diagnostic plot -----------------------------------------
        if config.save_diagnostic_plot:
            try:
                _save_diagnostic_plot(out_ds, output_dir / config.diagnostic_plot_name, config)
            except Exception as exc:  # noqa: BLE001 - diagnostic plot failure must not fail the pipeline
                logger.warning("Diagnostic plot failed (non-fatal): %s: %s", type(exc).__name__, exc)

        logger.info("=" * 70)
        logger.info("SSS preprocessing complete.")
        logger.info("Output NetCDF: %s", nc_path)
        logger.info("Output metadata: %s", metadata_path)
        logger.info("=" * 70)
        return 0

    finally:
        ds.close()


def _save_diagnostic_plot(out_ds: xr.Dataset, path: Path, config: SSSConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sss = out_ds["sss"]
    mask = out_ds["sss_valid_mask"]
    t0 = 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].pcolormesh(sss[config.output_longitude_name], sss[config.output_latitude_name], sss.isel({config.output_time_name: t0}), shading="auto")
    axes[0].set_title(f"SSS, timestep 0 ({np.asarray(sss[config.output_time_name].values[t0])})")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    fig.colorbar(im0, ax=axes[0], label=f"SSS ({sss.attrs.get('units', '?')})")

    im1 = axes[1].pcolormesh(mask[config.output_longitude_name], mask[config.output_latitude_name], mask.isel({config.output_time_name: t0}), shading="auto", vmin=0, vmax=1, cmap="gray")
    axes[1].set_title("Validity mask, timestep 0 (1=valid, 0=invalid)")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    fig.colorbar(im1, ax=axes[1])

    valid_frac_over_time = mask.mean(dim=[config.output_latitude_name, config.output_longitude_name])
    axes[2].plot(np.arange(len(valid_frac_over_time)), valid_frac_over_time.values)
    axes[2].set_title("Domain-mean valid fraction over time")
    axes[2].set_xlabel("Time step index"); axes[2].set_ylabel("Valid fraction")
    axes[2].set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Diagnostic plot written to: %s", path)


if __name__ == "__main__":
    sys.exit(main())
