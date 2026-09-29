"""
preprocess_sss.py (year-agnostic)
===================================
SIH26066 / OceanEmbed -- Sea Surface Salinity (SSS) preprocessing.
Works for ANY year's raw SSS file, not just 2025 -- see config_sss.py
CHANGELOG for what changed from the 2025-only 0.1.0 pipeline.

Single-file usage (PowerShell):
  python .\\preprocessing\\sss\\preprocess_sss.py `
    --input ".\\data\\raw\\sss\\test\\2024\\sss_2024.nc" `
    --output-dir ".\\data\\processed\\sss\\2024"

If --output-dir is omitted, the output directory is derived
automatically from the DETECTED year as data/processed/sss/<year>/.

Batch usage:
  python .\\preprocessing\\sss\\preprocess_sss.py `
    --input-dir ".\\data\\raw\\sss\\test" `
    --output-root ".\\data\\processed\\sss"

Dry run (either mode): add --dry-run.
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

import numpy as np
import xarray as xr

from config_sss import PIPELINE_VERSION, SSSConfig
import netcdf_utils as ncu

logger = logging.getLogger("preprocess_sss")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIH26066 OceanEmbed -- year-agnostic SSS preprocessing. Regrids native "
                    "~0.125 deg CNR SSS ('sos') onto the official 0.25 deg OceanEmbed grid "
                    "using NaN-aware weighted-linear interpolation, for any year's raw file."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--input", type=str, default=None, help="Path to a single raw SSS NetCDF file.")
    mode.add_argument("--input-dir", type=str, default=None, help="Batch mode: directory to search for raw SSS NetCDF files.")

    parser.add_argument("--output-dir", type=str, default=None, help="Single-file mode: explicit output directory (overrides the {year} template).")
    parser.add_argument("--output-root", type=str, default=None, help="Batch mode: root directory under which data/processed/sss/<year>/ subdirs are created.")
    parser.add_argument("--input-glob", type=str, default="**/*.nc", help="Batch mode: glob pattern (relative to --input-dir) for discovering files. Default: '**/*.nc'.")

    parser.add_argument("--dry-run", action="store_true", help="Validate and report; write nothing.")
    parser.add_argument("--year", type=int, default=None, help="Override the year used for output naming (normally auto-detected from the time coordinate).")
    parser.add_argument("--region", type=str, default=None, help="Override region as 'lat_min,lat_max,lon_min,lon_max' (degrees). Default: 5,30,45,105.")
    parser.add_argument("--target-resolution", type=float, default=None, help="Override target grid resolution in degrees. Default: 0.25.")
    parser.add_argument("--regrid-method", type=str, default=None, choices=["weighted_linear"], help="Regridding method. Only 'weighted_linear' is implemented.")
    parser.add_argument("--min-valid-weight", type=float, default=None, help="Minimum interpolation support-weight fraction for a target cell to be valid. Default: 0.5.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing outputs.")
    parser.add_argument("--require-complete-year", action="store_true", help="Fail if the detected year's calendar coverage is not complete (365/366 days). Default: allow partial years.")
    parser.add_argument("--allow-multi-year", action="store_true", help="ADVANCED/NON-STANDARD: allow an input file spanning more than one calendar year, assigning the whole file to its dominant year (not split). Default OFF -- normal OceanEmbed year-by-year preprocessing must not use this.")
    parser.add_argument("--no-plot", action="store_true", help="Skip the diagnostic plot.")
    parser.add_argument("--no-npy", action="store_true", help="Skip writing .npy companion outputs.")
    parser.add_argument("--assume-psu-equivalent", action="store_true", help="Add a documented metadata note interpreting '.001' as PSU-equivalent. NEVER changes numerical values. Default: off.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SSSConfig:
    config = SSSConfig()
    if args.input:
        config.explicit_input_path = args.input
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.explicit_output_dir = args.output_dir
    if args.output_root:
        config.output_root = args.output_root
    if args.input_glob:
        config.input_glob_pattern = args.input_glob
    if args.year is not None:
        config.expected_year = args.year
    if args.region:
        try:
            lat_min, lat_max, lon_min, lon_max = (float(x) for x in args.region.split(","))
        except ValueError as exc:
            raise ValueError(f"--region must be 'lat_min,lat_max,lon_min,lon_max', got {args.region!r}.") from exc
        config.lat_min, config.lat_max, config.lon_min, config.lon_max = lat_min, lat_max, lon_min, lon_max
    if args.target_resolution is not None:
        config.target_resolution_deg = args.target_resolution
    if args.regrid_method:
        config.regrid_method = args.regrid_method
    if args.min_valid_weight is not None:
        config.min_valid_weight = args.min_valid_weight
    if args.overwrite:
        config.overwrite = True
    if args.require_complete_year:
        config.require_complete_temporal_coverage = True
    if args.allow_multi_year:
        config.allow_multi_year = True
    if args.no_plot:
        config.save_diagnostic_plot = False
    if args.no_npy:
        config.write_npy_outputs = False
    if args.assume_psu_equivalent:
        config.assume_psu_equivalent = True
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
# Core per-file pipeline (shared by single-file and batch modes)
# =============================================================================

def process_one_file(input_path: Path, config: SSSConfig, run_started_at: datetime) -> dict:
    """
    Runs the full pipeline (or dry-run) for exactly one raw SSS file.
    Returns a small result summary dict on success. Raises on any
    validation/regridding/write/verification failure -- the caller
    (main() for single-file, run_batch() for batch mode) decides
    whether that's fatal to the whole process or just to this one file.
    """
    logger.info("SSS preprocessing pipeline v%s (dry_run=%s) -- input: %s", PIPELINE_VERSION, config.dry_run, input_path)
    raw_fingerprint_before = ncu.fingerprint_file(input_path)

    ds = ncu.open_dataset(input_path, use_dask_chunking=config.use_dask_chunking)
    try:
        print_dataset_summary(ds, input_path)

        var_name, var_detect_report = ncu.detect_variable(ds, config)
        coords4 = ncu.detect_all_coords(ds, config)
        logger.info("Detected SSS variable: '%s' (name_matches=%s, cf_matches=%s)",
                    var_name, var_detect_report["name_matches"], var_detect_report["cf_matches"])
        logger.info(
            "Detected coordinates: time='%s' depth='%s' latitude='%s' longitude='%s'",
            coords4.time, coords4.depth, coords4.latitude, coords4.longitude,
        )
        da_full = ds[var_name]
        logger.info("Variable attrs: %s", dict(da_full.attrs))
        logger.info("Variable encoding (post-CF-decode): %s", {
            k: da_full.encoding.get(k) for k in ("_FillValue", "scale_factor", "add_offset", "dtype")
        })

        coord_report = ncu.validate_coordinates(ds, coords4)
        for w in coord_report["warnings"]:
            logger.warning(w)
        logger.info(
            "Source grid: lat[%.4f, %.4f] step=%s deg (%d pts, regular=%s) | "
            "lon[%.4f, %.4f] step=%s deg (%d pts, regular=%s)",
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

        # ---- Year detection (data-derived, filename cross-checked) ------
        year_report = ncu.detect_year(ds, coords4, input_path, config)
        for w in year_report["warnings"]:
            logger.warning(w)
        year = year_report["year_used_for_output"]
        logger.info(
            "Detected year: %d (leap=%s, coverage=%s, %d/%d days present, filename_year=%s)",
            year_report["detected_year"], year_report["is_leap_year"], year_report["coverage_status"],
            year_report["n_days_present_in_detected_year"], year_report["expected_days_in_year"],
            year_report["filename_year"],
        )
        ncu.verify_temporal_coverage(year_report, config)

        # ---- Depth validation + drop ---------------------------------
        depth_report = ncu.validate_singleton_depth(ds, coords4, config)
        logger.info(
            "Depth validated: %s m (expected %s m, diff=%.2e m) -- confirmed surface.",
            depth_report["depth_value_m"], depth_report["expected_surface_depth_m"],
            depth_report["difference_from_expected_m"],
        )

        # ---- Coordinate normalization (descending order, lon convention)
        ds, order_report = ncu.normalize_coordinate_order(ds, coords4, config)
        for t in order_report["transformations"]:
            logger.info("Coordinate transformation applied: %s", t)
        ds, lon_report = ncu.detect_and_normalize_longitude_convention(ds, coords4, config)
        logger.info("Longitude convention: %s -> %s", lon_report["source_convention_detected"], lon_report["transformation"])

        # ---- Region subset (spatial only -- time dates preserved as-is)
        subset_ds, region_report = ncu.subset_region(ds, coords4, config)
        for w in region_report["warnings"]:
            logger.info(w)

        da4 = subset_ds[var_name].isel({coords4.depth: 0})
        coords3 = ncu.DetectedCoords3D(time=coords4.time, latitude=coords4.latitude, longitude=coords4.longitude)
        logger.info("Dropped singleton depth dimension. Shape now: %s %s", da4.dims, da4.shape)

        # ---- Missing-value + implausible-value inspection (pre-regrid)
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

        # ---- Unit check -----------------------------------------------
        unit_check = ncu.check_units(da4, config)
        for w in unit_check["warnings"]:
            logger.warning(w)

        # ---- Regridding -------------------------------------------------
        if not config.regrid_to_quarter_degree:
            raise ValueError("regrid_to_quarter_degree=False is not supported (explicit regridding is mandatory).")
        logger.info(
            "Regridding: native grid -> target %.2f deg grid, method=%s, min_valid_weight=%.2f",
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

        # ---- Post-regrid inspection + explicit mask-consistency check ---
        post_missing = ncu.inspect_missing_values(regridded_da)
        logger.info(
            "POST-REGRID stats: %d/%d valid (%.4f%%), %.4f%% invalid.",
            post_missing["valid_count"], post_missing["total_cells"],
            100 * post_missing["fraction_valid"], 100 * post_missing["fraction_invalid"],
        )
        ncu.validate_mask_consistency(regridded_da.values, mask_da.values)
        logger.info("Mask-consistency check PASSED (no NaN marked valid, no finite marked invalid, mask is binary).")

        grid_report = ncu.verify_final_grid(regridded_da, coords3, config)
        logger.info(
            "Final grid verified: %d x %d, lat step=%.4f deg, lon step=%.4f deg.",
            grid_report["n_latitude"], grid_report["n_longitude"],
            grid_report["latitude_spacing_deg"], grid_report["longitude_spacing_deg"],
        )

        regridded_da, mask_da, out_coords, rename_map = ncu.standardize_output_coordinates(
            regridded_da, mask_da, coords3, config
        )
        if rename_map:
            logger.info("Standardized coordinate names: %s", rename_map)
        if config.output_variable_name != "sss":
            logger.info("Output variable name: '%s' (configured)", config.output_variable_name)

        expected_shape = (da4.sizes[coords3.time], grid_report["n_latitude"], grid_report["n_longitude"])

        paths = ncu.build_output_paths(config, year)

        metadata = {
            "project": "OceanEmbed",
            "problem_statement": "SIH26066",
            "processing_stage": "SSS preprocessing",
            "pipeline_version": PIPELINE_VERSION,
            "processing_timestamp_utc": run_started_at.isoformat(),
            "source_file": str(input_path),
            "source_variable": var_name,
            "output_variable": config.output_variable_name,
            "source": {
                "file_size_bytes": raw_fingerprint_before["size_bytes"],
                "variable_attrs": dict(da_full.attrs),
                "variable_encoding": {k: da_full.encoding.get(k) for k in ("_FillValue", "scale_factor", "add_offset", "dtype")},
                "original_dimensions": dict(subset_ds.sizes),
                "native_resolution_deg": {"latitude": coord_report["latitude_resolution_deg"], "longitude": coord_report["longitude_resolution_deg"]},
                "native_coordinate_range": {"latitude": coord_report["latitude_range"], "longitude": coord_report["longitude_range"]},
                "depth": depth_report,
                "coordinate_transformations": order_report["transformations"],
                "longitude_convention": lon_report,
                "global_attrs_subset": {k: ds.attrs.get(k) for k in
                    ("Conventions", "grid_resolution", "title", "institution", "product_version", "software_version") if k in ds.attrs},
            },
            "year_detection": year_report,
            "units": unit_check["unit_block"],
            "source_grid_resolution": {"latitude_deg": coord_report["latitude_resolution_deg"], "longitude_deg": coord_report["longitude_resolution_deg"]},
            "target_grid_resolution": config.target_resolution_deg,
            "region": {"lat_min": config.lat_min, "lat_max": config.lat_max, "lon_min": config.lon_min, "lon_max": config.lon_max},
            "target_grid": {**regrid_report["target_grid"]},
            "regridding_method": regrid_report["method"],
            "extrapolation_enabled": False,
            "min_valid_weight": config.min_valid_weight,
            "regridding": {
                "min_valid_weight": regrid_report["min_valid_weight"],
                "extrapolation_policy": regrid_report["extrapolation_policy"],
                "missing_data_handling": regrid_report["missing_data_handling"],
                "target_cells_invalid_after_regrid": regrid_report["target_cells_invalid_fraction"],
                "n_target_points_with_zero_source_support_any_time": regrid_report.get("n_target_points_with_zero_source_support_any_time"),
                "boundary_note": (
                    "Where the native source grid does not reach the requested region boundary, "
                    "the outermost target row/column is invalid because the requested target "
                    "coordinate lies outside the native source coordinate range -- this reflects "
                    "unsupported source coverage, not necessarily land or physically missing ocean data."
                ),
            },
            "missing_data_policy": (
                "Missing/invalid source cells are never filled with zero, a mean, or any fabricated "
                "value. Preserved as NaN in the SSS variable and marked 0 in the validity mask. A "
                "target cell is valid (1) only if interpolation support >= min_valid_weight."
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
                f"{config.output_variable_name}_valid_mask": "1 = valid, finite, regridded SSS value with sufficient "
                    "source support (interpolation weight >= min_valid_weight). 0 = invalid -- missing source data, "
                    "unsupported source coverage (no extrapolation), or insufficient interpolation support."
            },
            "library_versions": _lib_versions(),
            "validation_status": "NOT YET VALIDATED (populated after reopen-and-validate step)",
            "warnings_and_unresolved_questions": [
                "min_valid_weight default (0.5) has not been confirmed as a project decision.",
                "Unit interpretation of source '.001' is pending verification against official product documentation "
                "(see 'units' block above) -- see unit_interpretation_status.",
                "Output variable naming ('sss') was not checked against a downstream 'tensor pipeline' convention "
                "(none was available to inspect in this environment).",
            ],
            "config_used": {k: v for k, v in vars(config).items()},
        }

        if config.dry_run:
            logger.info("DRY RUN: no processed output written.")
            paths["output_dir"].mkdir(parents=True, exist_ok=True)
            with open(paths["dryrun_report_path"], "w") as f:
                json.dump(metadata, f, indent=2, default=_json_default)
            logger.info("Dry-run report written to: %s", paths["dryrun_report_path"])
            logger.info("Expected final output shape: (%d, %d, %d)", *expected_shape)
            logger.info("Dry run complete for %s (year=%d).", input_path.name, year)
            return {"input": str(input_path), "year": year, "status": "dry_run_ok", "output_dir": str(paths["output_dir"])}

        nc_path = paths["nc_path"]
        if nc_path.exists() and not config.overwrite:
            raise FileExistsError(
                f"{nc_path} already exists and --overwrite was not given. Refusing to silently clobber an existing processed output."
            )

        var_name_out = config.output_variable_name
        mask_name_out = f"{config.output_variable_name}_valid_mask"
        out_ds = xr.Dataset(data_vars={var_name_out: regridded_da, mask_name_out: mask_da})
        out_ds[var_name_out].attrs = {
            "long_name": "sea surface salinity, regridded to 0.25 deg OceanEmbed grid",
            "standard_name": "sea_surface_salinity",
            "units": unit_check["unit_block"]["source_units"] or "unknown",
            "source_variable": var_name,
            "source_depth_m": depth_report["depth_value_m"],
            "regrid_method": regrid_report["method"],
            "min_valid_weight": config.min_valid_weight,
            "unit_interpretation_status": unit_check["unit_block"]["unit_interpretation_status"],
        }
        out_ds[mask_name_out].attrs = {
            "long_name": "SSS regridded-value validity mask",
            "flag_values": np.array([0, 1], dtype=np.uint8),
            "flag_meanings": "invalid valid",
            "description": metadata["validity_mask_meaning"][mask_name_out],
        }
        out_ds[config.output_latitude_name].attrs = {"standard_name": "latitude", "units": "degrees_north"}
        out_ds[config.output_longitude_name].attrs = {"standard_name": "longitude", "units": "degrees_east"}
        out_ds[config.output_time_name].attrs = {"standard_name": "time"}
        out_ds.attrs = {
            "title": f"OceanEmbed SSS preprocessed ({year}, 0.25 deg)",
            "project": "OceanEmbed",
            "problem_statement": "SIH26066",
            "processing_stage": "SSS preprocessing",
            "pipeline_version": PIPELINE_VERSION,
            "source_file": str(input_path),
            "source_variable": var_name,
            "source_grid_resolution": f"{coord_report['latitude_resolution_deg']:.4f} deg (native)",
            "target_grid_resolution": f"{config.target_resolution_deg} deg",
            "region": f"lat[{config.lat_min},{config.lat_max}] lon[{config.lon_min},{config.lon_max}]",
            "regridding_method": regrid_report["method"],
            "extrapolation_enabled": "False",
            "min_valid_weight": config.min_valid_weight,
            "processing_timestamp_utc": run_started_at.isoformat(),
            "Conventions": "CF-1.7",
        }

        encoding = {
            var_name_out: {"_FillValue": np.nan, "dtype": "float32"},
            mask_name_out: {"dtype": "uint8"},
        }
        if config.enable_netcdf_compression:
            encoding[var_name_out].update({"zlib": True, "complevel": config.netcdf_complevel})
            encoding[mask_name_out].update({"zlib": True, "complevel": config.netcdf_complevel})

        def _write_nc(tmp_path: Path) -> None:
            out_ds.to_netcdf(tmp_path, engine="netcdf4", encoding=encoding)

        logger.info("Writing processed NetCDF to: %s", nc_path)
        ncu.atomic_write(nc_path, _write_nc)

        if config.write_npy_outputs:
            logger.info("Writing companion .npy arrays: %s, %s", paths["npy_path"], paths["mask_npy_path"])
            ncu.atomic_write(paths["npy_path"], lambda p: np.save(p, out_ds[var_name_out].values))
            ncu.atomic_write(paths["mask_npy_path"], lambda p: np.save(p, out_ds[mask_name_out].values))

        logger.info("Reopening saved output to validate from disk...")
        validation_report = ncu.verify_output_file(nc_path, config, expected_shape)
        metadata["validation_status"] = "PASSED: reopened output matches expected structure." if validation_report.get("ok") else "FAILED"
        logger.info("Reopen validation PASSED: dims=%s shape=%s dtype=%s mask_consistent=%s",
                    validation_report["reopened_dims"], validation_report["reopened_shape"],
                    validation_report["reopened_dtype"], validation_report["mask_consistent_with_isfinite"])

        raw_fingerprint_after = ncu.fingerprint_file(input_path)
        fp_warnings = ncu.verify_file_unchanged(raw_fingerprint_before, raw_fingerprint_after)
        if fp_warnings:
            for w in fp_warnings:
                logger.error(w)
        else:
            logger.info("Confirmed: raw input file unchanged (size and mtime match pre-run fingerprint).")
        metadata["raw_input_file_unchanged"] = (len(fp_warnings) == 0)

        ncu.atomic_write(paths["metadata_path"], lambda p: p.write_text(json.dumps(metadata, indent=2, default=_json_default)))
        logger.info("Metadata written to: %s", paths["metadata_path"])

        if config.save_diagnostic_plot:
            try:
                _save_diagnostic_plot(out_ds, paths["diagnostic_plot_path"], config, year, var_name_out, mask_name_out)
            except Exception as exc:  # noqa: BLE001 - diagnostic plot failure must not fail the pipeline
                logger.warning("Diagnostic plot failed (non-fatal): %s: %s", type(exc).__name__, exc)

        logger.info("SSS preprocessing complete for %s (year=%d). Output: %s", input_path.name, year, nc_path)
        return {"input": str(input_path), "year": year, "status": "ok", "output_nc": str(nc_path)}

    finally:
        ds.close()


def _save_diagnostic_plot(out_ds: xr.Dataset, path: Path, config: SSSConfig, year: int, var_name: str, mask_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sss = out_ds[var_name]
    mask = out_ds[mask_name]
    coords3 = ncu.DetectedCoords3D(time=config.output_time_name, latitude=config.output_latitude_name, longitude=config.output_longitude_name)
    t_idx, sel_report = ncu.select_representative_timestep(mask, coords3)
    logger.info("Diagnostic plot representative timestep: index=%d (%s)", t_idx, sel_report["selected_time_value"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].pcolormesh(sss[config.output_longitude_name], sss[config.output_latitude_name], sss.isel({config.output_time_name: t_idx}), shading="auto")
    axes[0].set_title(f"SSS, {year}, t={sel_report['selected_time_value']}")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    fig.colorbar(im0, ax=axes[0], label=f"SSS ({sss.attrs.get('units', '?')})")

    im1 = axes[1].pcolormesh(mask[config.output_longitude_name], mask[config.output_latitude_name], mask.isel({config.output_time_name: t_idx}), shading="auto", vmin=0, vmax=1, cmap="gray")
    axes[1].set_title(f"Validity mask, t={sel_report['selected_time_value']} (1=valid, 0=invalid)")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    fig.colorbar(im1, ax=axes[1])

    valid_frac_over_time = mask.mean(dim=[config.output_latitude_name, config.output_longitude_name])
    axes[2].plot(np.arange(len(valid_frac_over_time)), valid_frac_over_time.values)
    axes[2].axvline(t_idx, color="red", linestyle="--", linewidth=1, label="plotted timestep")
    axes[2].set_title(f"Domain-mean valid fraction over time ({year})")
    axes[2].set_xlabel("Time step index"); axes[2].set_ylabel("Valid fraction")
    axes[2].set_ylim(0, 1); axes[2].legend()

    fig.suptitle(f"OceanEmbed SSS diagnostic -- {year} -- region lat[{config.lat_min},{config.lat_max}] lon[{config.lon_min},{config.lon_max}]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info("Diagnostic plot written to: %s", path)


# =============================================================================
# Batch mode
# =============================================================================

def run_batch(config: SSSConfig, run_started_at: datetime) -> int:
    input_dir = config.resolve(config.input_dir)
    files = ncu.discover_input_files(input_dir, config.input_glob_pattern)
    logger.info("Batch mode: discovered %d candidate file(s) under %s (pattern=%s)", len(files), input_dir, config.input_glob_pattern)
    if not files:
        logger.error("No .nc files found under %s with pattern %s.", input_dir, config.input_glob_pattern)
        return 2

    results = []
    failures = []
    for f in files:
        file_config = SSSConfig(**{**vars(config)})
        file_config.explicit_input_path = str(f)
        file_config.explicit_output_dir = None  # always use {year} template in batch mode
        if config.output_root:
            file_config.output_dir_template = str(Path(config.output_root) / "{year}")
        try:
            result = process_one_file(f, file_config, run_started_at)
            results.append(result)
        except Exception as exc:  # noqa: BLE001 - batch mode must continue past one bad file, but the failure is reported explicitly and loudly, never swallowed
            logger.error("FAILED processing %s: %s: %s", f, type(exc).__name__, exc)
            failures.append({"input": str(f), "error_type": type(exc).__name__, "error": str(exc)})

    logger.info("=" * 70)
    logger.info("BATCH SUMMARY: %d succeeded, %d failed, %d total.", len(results), len(failures), len(files))
    for r in results:
        logger.info("  OK   year=%s  %s -> %s", r.get("year"), r["input"], r.get("output_nc") or r.get("output_dir"))
    for f in failures:
        logger.error("  FAIL %s: %s: %s", f["input"], f["error_type"], f["error"])
    logger.info("=" * 70)
    return 0 if not failures else 1


# =============================================================================
# Main
# =============================================================================

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    config = build_config(args)
    run_started_at = datetime.now(timezone.utc)

    if config.input_dir:
        return run_batch(config, run_started_at)

    input_path, source_label = ncu.find_input_file(config)
    if input_path is None:
        logger.error("Could not locate an input file. %s. Use --input <path> or --input-dir <dir> for batch mode.", source_label)
        return 2
    logger.info("Using input file (%s)", source_label)

    try:
        result = process_one_file(input_path, config, run_started_at)
    except Exception as exc:
        logger.error("Pipeline FAILED: %s: %s", type(exc).__name__, exc)
        raise
    logger.info("Result: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
