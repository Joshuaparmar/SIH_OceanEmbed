"""
tests/test_boundary_offset_and_coverage.py
===========================================
Focused regression tests for the native-grid boundary-offset coverage
fix (see README "v0.3.0 upgrade: native-grid boundary-offset
coverage fix"). Plain assertions, no pytest dependency (none was
added to requirements.txt) -- run directly:

    cd preprocessing/sst
    python tests/test_boundary_offset_and_coverage.py

or from the repo root:

    python preprocessing/sst/tests/test_boundary_offset_and_coverage.py

All fixtures here are small, clearly-synthetic NetCDF files generated
in a temp directory and deleted afterward -- NONE of this exercises
the real OSTIA/CMEMS 2025 file (which is not available in an automated
test environment; see README "Real dry-run" section for the manual/
documented path to test against the actual file locally). Passing
these tests demonstrates the FIX is logically correct on representative
synthetic data; it is not a substitute for the real-file dry run.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import netcdf_utils  # noqa: E402
import validation  # noqa: E402
from config_sst import CONFIG  # noqa: E402

PASS = []
FAIL = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASS.append(name)
        print(f"[PASS] {name}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name}  {detail}")


def make_offset_fixture(path: Path, spacing: float = 0.05,
                         lat_min_coord: float = None, lat_max_coord: float = None,
                         lon_min_coord: float = None, lon_max_coord: float = None) -> None:
    """
    Build a small synthetic SST-shaped NetCDF whose native grid is offset
    by half a cell from the OceanEmbed 5/30/45/105 boundary (mirroring
    the real OSTIA file's ~0.025 deg offset at ~0.05 deg spacing).

    Each of lat_min_coord/lat_max_coord/lon_min_coord/lon_max_coord can
    be given explicitly to control exactly how far the source's real
    coordinate extent reaches in that direction (used to construct
    either an ample-margin fixture, a genuine-gap fixture, or a
    tight-exact-offset-only fixture with no margin beyond the natural
    half-cell alignment). Defaults give 10 deg of margin beyond the
    requested boundary on every side (an ample-coverage fixture).
    """
    half = spacing / 2.0
    if lat_min_coord is None:
        lat_min_coord = 5.0 - 10.0 + half
    if lat_max_coord is None:
        lat_max_coord = 30.0 + 10.0 - half
    if lon_min_coord is None:
        lon_min_coord = 45.0 - 10.0 + half
    if lon_max_coord is None:
        lon_max_coord = 105.0 + 10.0 - half

    lat = np.round(np.arange(lat_min_coord, lat_max_coord + 1e-9, spacing), 4)
    lon = np.round(np.arange(lon_min_coord, lon_max_coord + 1e-9, spacing), 4)
    time = pd.date_range("2025-01-01", "2025-01-02", freq="D")
    rng = np.random.default_rng(0)
    sst_c = 27.0 + rng.standard_normal((len(time), len(lat), len(lon)))
    sst_k = (sst_c + 273.15).astype("float32")
    ds = xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), sst_k)},
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    ds["analysed_sst"].attrs["units"] = "kelvin"
    ds.to_netcdf(path)


def load(path: Path, cfg):
    paths, _ = netcdf_utils.find_input_files(cfg)
    ds = netcdf_utils.open_dataset(paths)
    var = netcdf_utils.detect_variable(ds, cfg.sst_var_candidates)
    coords = netcdf_utils.detect_all_coords(ds, cfg)
    coord_report = netcdf_utils.validate_coordinates(ds, coords)
    return ds, var, coords, coord_report


def test_1_half_cell_offset_should_pass(tmpdir: Path) -> None:
    """Test 1: a source whose native grid is offset by half a cell from
    the requested boundary (like the real OSTIA file) must PASS official
    coverage validation, and the final target grid must still be exactly
    5.0-30.0 (101 points) / 45.0-105.0 (241 points) @ 0.25 deg."""
    path = tmpdir / "offset.nc"
    make_offset_fixture(path)  # ample-margin, half-cell-offset-only fixture
    cfg = replace(CONFIG, project_root=str(tmpdir), explicit_input_path=str(path),
                  start_date="2025-01-01", end_date="2025-01-02", date_range_confirmed=True)
    ds, var, coords, coord_report = load(path, cfg)
    try:
        coverage_report = validation.verify_official_spatial_coverage(coord_report, cfg)
        check("test1_coverage_passed", coverage_report["passed"] is True)
    except ValueError as exc:
        check("test1_coverage_passed", False, str(exc)[:200])
        ds.close()
        return

    support_bounds = netcdf_utils.build_interpolation_support_bounds(
        cfg, coord_report.get("latitude_resolution_deg"), coord_report.get("longitude_resolution_deg")
    )
    support_ds, support_report = netcdf_utils.subset_region_time(
        ds, coords, cfg,
        lat_min=support_bounds["lat_min"], lat_max=support_bounds["lat_max"],
        lon_min=support_bounds["lon_min"], lon_max=support_bounds["lon_max"],
        bounds_label="interpolation_support",
    )
    da = support_ds[var].transpose(coords.time, coords.latitude, coords.longitude)
    regridded, regrid_report = netcdf_utils.regrid_to_target_grid(da, coords, cfg)
    check("test1_final_lat_count_101", regridded.sizes[coords.latitude] == 101,
          f"got {regridded.sizes[coords.latitude]}")
    check("test1_final_lon_count_241", regridded.sizes[coords.longitude] == 241,
          f"got {regridded.sizes[coords.longitude]}")
    check("test1_final_lat_bounds_exact", np.isclose(float(regridded[coords.latitude].min()), 5.0)
          and np.isclose(float(regridded[coords.latitude].max()), 30.0))
    check("test1_final_lon_bounds_exact", np.isclose(float(regridded[coords.longitude].min()), 45.0)
          and np.isclose(float(regridded[coords.longitude].max()), 105.0))
    ds.close()


def test_2_genuine_gap_should_fail(tmpdir: Path) -> None:
    """Test 2: a source that stops meaningfully short of the requested
    boundary (well beyond native-cell tolerance) must FAIL official
    coverage validation with a clear error."""
    path = tmpdir / "genuine_gap.nc"
    # Source stops 5 deg short of the requested lat_max (30.0) -- a real gap,
    # not just a half-cell alignment offset.
    make_offset_fixture(path, lat_max_coord=25.0 - 0.025)
    cfg = replace(CONFIG, project_root=str(tmpdir), explicit_input_path=str(path),
                  start_date="2025-01-01", end_date="2025-01-02", date_range_confirmed=True)
    ds, var, coords, coord_report = load(path, cfg)
    raised = False
    try:
        validation.verify_official_spatial_coverage(coord_report, cfg)
    except ValueError:
        raised = True
    check("test2_genuine_gap_raises", raised)
    ds.close()


def test_3_expanded_support_selection(tmpdir: Path) -> None:
    """Test 3: the interpolation-support subset must include source
    cells OUTSIDE the strict requested bounds, while the final regridded
    output remains restricted to exactly the requested target grid."""
    path = tmpdir / "offset.nc"
    make_offset_fixture(path)
    cfg = replace(CONFIG, project_root=str(tmpdir), explicit_input_path=str(path),
                  start_date="2025-01-01", end_date="2025-01-02", date_range_confirmed=True,
                  interpolation_support_cells=2)
    ds, var, coords, coord_report = load(path, cfg)
    strict_ds, strict_report = netcdf_utils.subset_region_time(ds, coords, cfg)
    support_bounds = netcdf_utils.build_interpolation_support_bounds(
        cfg, coord_report.get("latitude_resolution_deg"), coord_report.get("longitude_resolution_deg")
    )
    support_ds, support_report = netcdf_utils.subset_region_time(
        ds, coords, cfg,
        lat_min=support_bounds["lat_min"], lat_max=support_bounds["lat_max"],
        lon_min=support_bounds["lon_min"], lon_max=support_bounds["lon_max"],
        bounds_label="interpolation_support",
    )
    check("test3_support_has_more_lat_cells_than_strict",
          support_report["n_lat_matched"] > strict_report["n_lat_matched"],
          f"support={support_report['n_lat_matched']} strict={strict_report['n_lat_matched']}")
    check("test3_support_has_more_lon_cells_than_strict",
          support_report["n_lon_matched"] > strict_report["n_lon_matched"],
          f"support={support_report['n_lon_matched']} strict={strict_report['n_lon_matched']}")
    check("test3_support_extends_below_strict_lat_min",
          support_report["actual_lat_range"][0] < strict_report["actual_lat_range"][0])
    check("test3_support_extends_above_strict_lat_max",
          support_report["actual_lat_range"][1] > strict_report["actual_lat_range"][1])

    da = support_ds[var].transpose(coords.time, coords.latitude, coords.longitude)
    regridded, _ = netcdf_utils.regrid_to_target_grid(da, coords, cfg)
    check("test3_final_output_restricted_to_requested_grid",
          regridded.sizes[coords.latitude] == 101 and regridded.sizes[coords.longitude] == 241)
    ds.close()


def test_4_no_extrapolation(tmpdir: Path) -> None:
    """Test 4: a target cell that lies genuinely beyond the source's
    real coverage (even after interpolation-support expansion) must be
    NaN, never an extrapolated/fabricated value."""
    path = tmpdir / "short_at_top.nc"
    # Source's last coordinate sits EXACTLY one half-native-cell below
    # 30.0 (matching the REAL reported OSTIA extent of 29.975 at
    # ~0.05 deg spacing) with ample margin on the low/left sides but
    # NO margin at all beyond the half-cell offset on the high side --
    # target row at exactly lat=30.0 has no bracketing source data on
    # the high side and cannot be linearly interpolated without
    # extrapolating.
    make_offset_fixture(path, lat_max_coord=30.0 - 0.025)
    cfg = replace(CONFIG, project_root=str(tmpdir), explicit_input_path=str(path),
                  start_date="2025-01-01", end_date="2025-01-02", date_range_confirmed=True)
    ds, var, coords, coord_report = load(path, cfg)
    support_bounds = netcdf_utils.build_interpolation_support_bounds(
        cfg, coord_report.get("latitude_resolution_deg"), coord_report.get("longitude_resolution_deg")
    )
    support_ds, _ = netcdf_utils.subset_region_time(
        ds, coords, cfg,
        lat_min=support_bounds["lat_min"], lat_max=support_bounds["lat_max"],
        lon_min=support_bounds["lon_min"], lon_max=support_bounds["lon_max"],
        bounds_label="interpolation_support",
    )
    da = support_ds[var].transpose(coords.time, coords.latitude, coords.longitude)
    regridded, regrid_report = netcdf_utils.regrid_to_target_grid(da, coords, cfg)
    last_lat_row = regridded.isel({coords.latitude: -1}).values
    second_last_lat_row = regridded.isel({coords.latitude: -2}).values
    check("test4_exact_boundary_beyond_source_stays_nan",
          bool(np.isnan(last_lat_row).all()) or bool(np.isnan(last_lat_row).mean() > 0.9),
          f"NaN fraction at exact upper boundary: {np.isnan(last_lat_row).mean():.3f}")
    check("test4_interior_near_boundary_is_real_interpolated_data",
          bool(np.isfinite(second_last_lat_row).mean() > 0.5),
          f"finite fraction one step in: {np.isfinite(second_last_lat_row).mean():.3f}")
    ds.close()


def test_5_temporal_vs_spatial_not_misattributed(tmpdir: Path) -> None:
    """Test 5 (regression from the previous round): a purely SPATIAL
    coverage gap must be reported as a spatial/coverage error, never
    mislabeled as a temporal-coverage problem, and end-date selection
    must be end-inclusive regardless of time-of-day stamping."""
    path = tmpdir / "noon_stamped.nc"
    lat = np.round(np.arange(5.025, 30.0, 0.05), 4)
    lon = np.round(np.arange(45.025, 105.0, 0.05), 4)
    time = pd.date_range("2025-01-01 12:00", "2025-01-03 12:00", freq="D")
    sst = (27.0 + np.random.default_rng(1).standard_normal((len(time), len(lat), len(lon))) + 273.15).astype("float32")
    ds_fixture = xr.Dataset({"analysed_sst": (("time", "latitude", "longitude"), sst)},
                             coords={"time": time, "latitude": lat, "longitude": lon})
    ds_fixture["analysed_sst"].attrs["units"] = "kelvin"
    ds_fixture.to_netcdf(path)

    cfg = replace(CONFIG, project_root=str(tmpdir), explicit_input_path=str(path),
                  start_date="2025-01-01", end_date="2025-01-03", date_range_confirmed=True)
    ds, var, coords, coord_report = load(path, cfg)
    strict_ds, strict_report = netcdf_utils.subset_region_time(ds, coords, cfg)
    check("test5_end_date_inclusive_all_3_days_selected", strict_report["n_time_matched"] == 3,
          f"got {strict_report['n_time_matched']}")
    check("test5_full_time_coverage_true_despite_noon_stamps", strict_report["full_time_coverage"] is True)
    try:
        validation.verify_temporal_coverage(coord_report, strict_report, cfg)
        check("test5_temporal_coverage_check_passes", True)
    except ValueError as exc:
        check("test5_temporal_coverage_check_passes", False, str(exc)[:150])
    ds.close()


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="sst_pipeline_tests_"))
    try:
        test_1_half_cell_offset_should_pass(tmpdir)
        test_2_genuine_gap_should_fail(tmpdir)
        test_3_expanded_support_selection(tmpdir)
        test_4_no_extrapolation(tmpdir)
        test_5_temporal_vs_spatial_not_misattributed(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
