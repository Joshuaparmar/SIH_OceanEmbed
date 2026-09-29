"""
Focused tests for year/date-range generalization in the SST pipeline.
Plain assertions, no pytest dependency -- run from the repo root:

    python preprocessing/sst_preprocess_trial3/test_date_generalization.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import netcdf_utils  # noqa: E402
import preprocess_sst  # noqa: E402
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


def resolve(argv):
    args = preprocess_sst.parse_args(argv)
    cfg = preprocess_sst.build_config(args)
    cfg, report = preprocess_sst.resolve_cli_or_config_dates(cfg, args)
    cfg = preprocess_sst.apply_default_year_output_dir(cfg, args)
    return args, cfg, report


def expect_date_error(name: str, argv) -> None:
    try:
        args = preprocess_sst.parse_args(argv)
        cfg = preprocess_sst.build_config(args)
        preprocess_sst.resolve_cli_or_config_dates(cfg, args)
    except ValueError:
        check(name, True)
        return
    check(name, False, "expected ValueError")


def make_fixture(path: Path, times) -> None:
    lat = np.array([5.0, 6.0], dtype="float32")
    lon = np.array([45.0, 46.0], dtype="float32")
    values = np.full((len(times), len(lat), len(lon)), 300.0, dtype="float32")
    ds = xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), values)},
        coords={"time": pd.DatetimeIndex(times), "latitude": lat, "longitude": lon},
    )
    ds["analysed_sst"].attrs["units"] = "kelvin"
    ds.to_netcdf(path)


def test_year_ranges() -> None:
    _, cfg_2025, report_2025 = resolve(["--year", "2025"])
    check("year_2025_start", cfg_2025.start_date == "2025-01-01")
    check("year_2025_end", cfg_2025.end_date == "2025-12-31")
    check("year_2025_expected_days", report_2025["expected_full_year_day_count"] == 365)

    _, cfg_2024, report_2024 = resolve(["--year", "2024"])
    check("leap_2024_start", cfg_2024.start_date == "2024-01-01")
    check("leap_2024_end", cfg_2024.end_date == "2024-12-31")
    check("leap_2024_expected_days", report_2024["expected_full_year_day_count"] == 366)

    _, cfg_2026, report_2026 = resolve(["--year", "2026"])
    check("year_2026_start", cfg_2026.start_date == "2026-01-01")
    check("year_2026_end", cfg_2026.end_date == "2026-12-31")
    check("year_2026_expected_days", report_2026["expected_full_year_day_count"] == 365)


def test_explicit_date_range_and_invalid_inputs() -> None:
    _, cfg, report = resolve(["--start-date", "2024-06-01", "--end-date", "2024-08-31"])
    check("explicit_range_start", cfg.start_date == "2024-06-01")
    check("explicit_range_end", cfg.end_date == "2024-08-31")
    check("explicit_range_end_inclusive_count", report["requested_day_count"] == 92)

    expect_date_error("invalid_end_before_start", ["--start-date", "2024-08-31", "--end-date", "2024-06-01"])
    expect_date_error("invalid_conflicting_year", ["--year", "2024", "--start-date", "2025-01-01", "--end-date", "2025-01-31"])
    expect_date_error("invalid_calendar_date", ["--start-date", "2024-02-30", "--end-date", "2024-03-01"])
    expect_date_error("invalid_partial_explicit_range", ["--start-date", "2024-06-01"])


def test_output_isolation(tmpdir: Path) -> None:
    _, cfg_2024, _ = resolve(["--year", "2024"])
    _, cfg_2025, _ = resolve(["--year", "2025"])
    check("output_2024_year_child", Path(cfg_2024.output_dir).parts[-1] == "2024", cfg_2024.output_dir)
    check("output_2025_year_child", Path(cfg_2025.output_dir).parts[-1] == "2025", cfg_2025.output_dir)
    check("output_years_are_separate", cfg_2024.output_dir != cfg_2025.output_dir)

    _, explicit_cfg, _ = resolve(["--year", "2024", "--output-dir", "custom/sst"])
    check("explicit_output_dir_not_rewritten", explicit_cfg.output_dir == "custom/sst", explicit_cfg.output_dir)

    raw_dir = tmpdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "sst_2024.nc"
    raw_file.write_bytes(b"not a real netcdf; collision guard only needs a path")
    try:
        netcdf_utils.check_input_output_collision([raw_file], raw_dir)
    except ValueError:
        check("raw_input_output_same_dir_rejected", True)
    else:
        check("raw_input_output_same_dir_rejected", False, "expected ValueError")
    try:
        netcdf_utils.check_input_output_collision([raw_file], tmpdir / "processed" / "sst" / "2024")
        netcdf_utils.check_input_output_collision([raw_file], tmpdir / "processed" / "sst" / "2025")
        check("year_outputs_do_not_collide_with_raw", True)
    except ValueError as exc:
        check("year_outputs_do_not_collide_with_raw", False, str(exc))


def test_multi_year_selection_and_inference(tmpdir: Path) -> None:
    path = tmpdir / "multi_year.nc"
    make_fixture(path, ["2024-12-31", "2025-01-01", "2025-01-02"])

    args, cfg, _ = resolve(["--input", str(path), "--year", "2025"])
    cfg = replace(cfg, project_root=str(tmpdir), explicit_input_path=str(path))
    ds = netcdf_utils.open_dataset([path], use_dask_chunking=False)
    try:
        coords = netcdf_utils.detect_all_coords(ds, cfg)
        subset, report = netcdf_utils.subset_region_time(ds, coords, cfg)
        check("multi_year_requested_year_selected", report["n_time_matched"] == 2, str(report))
        check("multi_year_requested_dates_complete", report["requested_period_fully_covered"] is False)
        check("multi_year_no_arbitrary_inference_with_multiple_years", True)
        coord_report = netcdf_utils.validate_coordinates(ds, coords)
        try:
            preprocess_sst.infer_dates_from_dataset(replace(CONFIG, start_date=None, end_date=None), coord_report)
        except ValueError:
            check("multi_year_inference_rejected", True)
        else:
            check("multi_year_inference_rejected", False, "expected ValueError")
    finally:
        ds.close()

    one_year_path = tmpdir / "one_year.nc"
    make_fixture(one_year_path, ["2024-06-01", "2024-06-02"])
    ds = netcdf_utils.open_dataset([one_year_path], use_dask_chunking=False)
    try:
        coords = netcdf_utils.detect_all_coords(ds, CONFIG)
        coord_report = netcdf_utils.validate_coordinates(ds, coords)
        inferred_cfg, inferred_report = preprocess_sst.infer_dates_from_dataset(
            replace(CONFIG, start_date=None, end_date=None), coord_report
        )
        check("one_year_inference_year", inferred_cfg.processing_year == 2024)
        check("one_year_inference_full_year_range", inferred_cfg.start_date == "2024-01-01"
              and inferred_cfg.end_date == "2024-12-31")
        check("one_year_inference_leap_count", inferred_report["expected_full_year_day_count"] == 366)
    finally:
        ds.close()


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp(prefix="sst_date_tests_"))
    try:
        test_year_ranges()
        test_explicit_date_range_and_invalid_inputs()
        test_output_isolation(tmpdir)
        test_multi_year_selection_and_inference(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
