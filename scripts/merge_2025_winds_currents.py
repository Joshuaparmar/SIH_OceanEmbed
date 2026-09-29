#!/usr/bin/env python3
"""
merge_2025_winds_currents.py

Consolidate daily 2025 wind and ocean-current NetCDF files into two
single time-series NetCDF files with EXACTLY ONE TIME ENTRY PER DAY.

Currents
--------
Each OSCAR current file already contains exactly one daily observation.
These are concatenated only -- no averaging, no value changes. Output
values are bit-identical to the originals.

Winds
-----
Each CCMP wind file contains four 6-hourly records for that day
(00/06/12/18 UTC). These are temporally aggregated to ONE daily record
per day:

    daily_uwnd = mean(uwnd over the day's valid 6-hourly records)
    daily_vwnd = mean(vwnd over the day's valid 6-hourly records)
    daily_ws   = mean(ws over the day's valid 6-hourly records)
                 (mean of the reported speed magnitudes -- NOT the
                 same quantity as the speed of the mean vector)
    ws_from_daily_mean_components = sqrt(daily_uwnd**2 + daily_vwnd**2)
                 (derived diagnostic, vector-consistent daily speed)
    daily_nobs = sum(nobs over the day)
                 (total observation count contributing to the day;
                 see rationale in AGGREGATION NOTES below)

Because this is a temporal aggregation, THE WIND OUTPUT VALUES ARE NOT
BIT-IDENTICAL to the raw 6-hourly files. This is expected and is
documented in the output's global attributes. The current output
values ARE bit-identical to the raw files (pure concatenation, no
aggregation).

The original raw files are never modified, deleted, or renamed.
No interpolation, regridding, resampling, normalization, land masking,
cropping, or tensor conversion is performed anywhere in this script.

AGGREGATION NOTES (nobs)
-------------------------
`nobs` in the CCMP files is documented as "number of observations used
to derive wind vector components" for each 6-hourly analysis window.
Summing across a day's four 6-hourly windows gives the total number of
satellite/model observations that contributed to that day's wind
field -- a natural, easily-interpreted daily quantity. (Averaging or
taking the min/max would answer a different, less useful question --
"typical observation density per synoptic time" rather than "how much
data informed this day" -- so sum was chosen and is documented on the
variable itself via its `daily_aggregation` attribute.)

TIMESTAMP CONVENTION
---------------------
Each output file has one timestamp per UTC calendar day, set to
00:00:00 UTC of that day (e.g. "2025-06-15T00:00:00"). For winds this
represents the aggregate of that day's 00/06/12/18 UTC records. For
currents this is the timestamp already provided by OSCAR (each OSCAR
file's original time value already represents a whole-day average --
see the file's own "comment" attribute -- so it is preserved as-is,
only normalized to a clean UTC-midnight representation).

Usage
-----
    # 1. Inspect + validate only (no merge). ALWAYS do this first for a
    #    full-year run and confirm 365/365 complete days before merging.
    python merge_2025_winds_currents.py \
        --winds-dir data/raw/winds/2025 \
        --currents-dir data/raw/currents/test/2025 \
        --output-dir data/raw/merged/2025 \
        --year 2025 --dry-run

    # 2. Actually merge (only after the dry-run confirms completeness)
    python merge_2025_winds_currents.py \
        --winds-dir data/raw/winds/2025 \
        --currents-dir data/raw/currents/test/2025 \
        --output-dir data/raw/merged/2025 \
        --year 2025

    # 3. If (and only if) some days are genuinely short on records and
    #    you have decided that's acceptable, override the hard stop:
    #    ... --allow-incomplete-days
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_2025")

DATE_RE = re.compile(r"(\d{8})")


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------

@dataclass
class Inventory:
    label: str
    files_by_date: dict = field(default_factory=dict)
    unparsed_files: list = field(default_factory=list)
    duplicate_dates: dict = field(default_factory=dict)

    @property
    def sorted_dates(self):
        return sorted(self.files_by_date.keys())

    @property
    def sorted_files(self):
        return [self.files_by_date[d] for d in self.sorted_dates]


# --------------------------------------------------------------------------
# STEP 1 -- Inventory
# --------------------------------------------------------------------------

def discover_files(directory: Path, label: str) -> Inventory:
    inv = Inventory(label=label)
    if not directory.exists():
        raise FileNotFoundError(f"{label}: directory does not exist: {directory}")

    nc_files = sorted(directory.rglob("*.nc"))
    log.info("%s: found %d .nc files under %s", label, len(nc_files), directory)

    seen: dict = {}
    for f in nc_files:
        m = DATE_RE.search(f.name)
        if not m:
            inv.unparsed_files.append(f)
            continue
        raw = m.group(1)
        try:
            d = date(int(raw[0:4]), int(raw[4:6]), int(raw[6:8]))
        except ValueError:
            inv.unparsed_files.append(f)
            continue
        if d in seen:
            inv.duplicate_dates.setdefault(d, [seen[d]]).append(f)
        else:
            seen[d] = f
    inv.files_by_date = seen

    if inv.unparsed_files:
        log.warning("%s: %d files had no parseable 8-digit date and were IGNORED: %s",
                    label, len(inv.unparsed_files), [f.name for f in inv.unparsed_files[:5]])
    if inv.duplicate_dates:
        log.warning("%s: duplicate dates detected for %d date(s): %s",
                    label, len(inv.duplicate_dates),
                    {d.isoformat(): [f.name for f in fs] for d, fs in inv.duplicate_dates.items()})
    return inv


def report_coverage(inv: Inventory, year: int) -> bool:
    dates = inv.sorted_dates
    if not dates:
        log.error("%s: no dated files found at all.", inv.label)
        return False

    log.info("%s: first file = %s (%s)", inv.label, inv.files_by_date[dates[0]].name, dates[0])
    log.info("%s: last file  = %s (%s)", inv.label, inv.files_by_date[dates[-1]].name, dates[-1])
    log.info("%s: %d distinct dates spanning %s to %s", inv.label, len(dates), dates[0], dates[-1])

    start = date(year, 1, 1)
    end = date(year, 12, 31)
    expected = set()
    d = start
    while d <= end:
        expected.add(d)
        d += timedelta(days=1)

    present_in_year = {d for d in dates if start <= d <= end}
    missing = sorted(expected - present_in_year)
    outside_year = sorted(d for d in dates if d < start or d > end)

    if missing:
        log.warning("%s: MISSING %d date(s) in %d (showing up to 10): %s",
                    inv.label, len(missing), year, [d.isoformat() for d in missing[:10]])
    else:
        log.info("%s: full coverage of %d-01-01 through %d-12-31 confirmed (%d/%d days).",
                  inv.label, year, year, len(present_in_year), len(expected))

    if outside_year:
        log.warning("%s: %d file(s) found with dates OUTSIDE %d: %s",
                    inv.label, len(outside_year), year, [d.isoformat() for d in outside_year[:10]])

    return (not missing) and (not inv.duplicate_dates)


# --------------------------------------------------------------------------
# STEP 2 -- Structural inspection (no hardcoded var names)
# --------------------------------------------------------------------------

def inspect_file(path: Path) -> dict:
    ds = xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    try:
        info = {
            "path": str(path),
            "dims": dict(ds.sizes),
            "coord_names": list(ds.coords),
            "data_vars": {},
            "global_attrs": dict(ds.attrs),
        }
        for name, da in ds.data_vars.items():
            info["data_vars"][name] = {
                "dims": da.dims, "dtype": str(da.dtype), "shape": da.shape,
                "attrs": dict(da.attrs),
            }
        for name in ds.coords:
            ca = ds.coords[name]
            info[f"coord::{name}"] = {
                "dims": ca.dims, "dtype": str(ca.dtype), "shape": ca.shape,
                "attrs": dict(ca.attrs),
                "min": float(np.nanmin(ca.values)) if np.issubdtype(ca.dtype, np.number) else None,
                "max": float(np.nanmax(ca.values)) if np.issubdtype(ca.dtype, np.number) else None,
            }
        return info
    finally:
        ds.close()


def log_inspection(info: dict, tag: str) -> None:
    log.info("--- Structure of %s (%s) ---", info["path"], tag)
    log.info("  dims: %s", info["dims"])
    log.info("  data variables: %s", list(info["data_vars"].keys()))
    for name, meta in info["data_vars"].items():
        log.info("    %s: dims=%s dtype=%s fill=%s units=%s",
                  name, meta["dims"], meta["dtype"],
                  meta["attrs"].get("_FillValue"), meta["attrs"].get("units"))
    for key in info:
        if key.startswith("coord::"):
            cname = key.split("::", 1)[1]
            meta = info[key]
            log.info("    coord %s: dims=%s dtype=%s range=[%s, %s] units=%s",
                      cname, meta["dims"], meta["dtype"], meta["min"], meta["max"],
                      meta["attrs"].get("units"))


def inspect_representative_files(inv: Inventory) -> list:
    dates = inv.sorted_dates
    if not dates:
        return []
    idx = sorted(set([0, len(dates) // 2, len(dates) - 1]))
    results = []
    for i in idx:
        path = inv.files_by_date[dates[i]]
        info = inspect_file(path)
        tag = {0: "first", len(dates) - 1: "last"}.get(i, "middle")
        log_inspection(info, tag)
        results.append(info)
    return results


def find_lat_lon_names(info: dict) -> tuple:
    lat_name = lon_name = None
    for key in info:
        if not key.startswith("coord::"):
            continue
        cname = key.split("::", 1)[1]
        attrs = info[key]["attrs"]
        std = (attrs.get("standard_name") or "").lower()
        axis = (attrs.get("axis") or "").lower()
        if std == "latitude" or axis == "y" or "lat" in cname.lower():
            lat_name = lat_name or cname
        if std == "longitude" or axis == "x" or "lon" in cname.lower():
            lon_name = lon_name or cname
    return lat_name, lon_name


# --------------------------------------------------------------------------
# STEP 3 -- Cross-file consistency validation
# --------------------------------------------------------------------------

def validate_consistency(inv: Inventory, lat_name: Optional[str] = None,
                          lon_name: Optional[str] = None) -> dict:
    """Checks variable set, SPATIAL dims (lat/lon grid size -- time length
    is intentionally excluded here since wind files may legitimately vary
    in time-dim length; that is checked separately by
    check_daily_record_counts), coordinate array equality, and duplicate
    time values across files."""
    files = inv.sorted_files
    if not files:
        raise RuntimeError(f"{inv.label}: no files to validate.")

    report = {
        "n_files": len(files),
        "var_set_consistent": True,
        "spatial_dims_consistent": True,
        "coords_consistent": True,
        "issues": [],
        "time_dim_sizes": {},
    }

    ref_vars = None
    ref_spatial_dims = None
    ref_lat = None
    ref_lon = None
    all_times = []
    all_time_units = set()

    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        try:
            vset = set(ds.data_vars)
            dims = dict(ds.sizes)
            spatial_dims = {k: v for k, v in dims.items() if k != "time"}
            report["time_dim_sizes"][f.name] = dims.get("time")

            if ref_vars is None:
                ref_vars = vset
            elif vset != ref_vars:
                report["var_set_consistent"] = False
                report["issues"].append(f"{f.name}: variable set {vset} differs from reference {ref_vars}")

            if ref_spatial_dims is None:
                ref_spatial_dims = spatial_dims
            elif spatial_dims != ref_spatial_dims:
                report["spatial_dims_consistent"] = False
                report["issues"].append(f"{f.name}: spatial dims {spatial_dims} differ from reference {ref_spatial_dims}")

            if lat_name and lat_name in ds.variables:
                lat = ds[lat_name].values
                if ref_lat is None:
                    ref_lat = lat
                elif not np.array_equal(lat, ref_lat):
                    report["coords_consistent"] = False
                    report["issues"].append(f"{f.name}: {lat_name} differs from reference")
            if lon_name and lon_name in ds.variables:
                lon = ds[lon_name].values
                if ref_lon is None:
                    ref_lon = lon
                elif not np.array_equal(lon, ref_lon):
                    report["coords_consistent"] = False
                    report["issues"].append(f"{f.name}: {lon_name} differs from reference")

            if "time" in ds.variables:
                t = ds["time"].values
                all_times.append(t)
                units = ds["time"].attrs.get("units")
                calendar = ds["time"].attrs.get("calendar", "standard")
                all_time_units.add((units, calendar))
        finally:
            ds.close()

    if len(all_time_units) > 1:
        report["issues"].append(f"Inconsistent time units/calendar across files: {all_time_units}")

    if all_times:
        concat_times = np.concatenate(all_times)
        n_total = len(concat_times)
        n_unique = len(np.unique(concat_times))
        report["n_time_steps_total"] = n_total
        report["n_time_steps_unique"] = n_unique
        report["duplicate_time_values"] = n_total - n_unique
        if n_total != n_unique:
            report["issues"].append(f"{n_total - n_unique} duplicate time value(s) found across files.")

    for issue in report["issues"]:
        log.warning("%s: %s", inv.label, issue)
    if not report["issues"]:
        log.info("%s: consistency check passed across all %d files.", inv.label, len(files))

    return report


def check_daily_record_counts(inv: Inventory, expected: Optional[int] = None) -> dict:
    """Wind-specific: for each daily file, how many time records does it
    actually contain? Compares against `expected` (auto-detected as the
    most common count across all files, if not given). Returns a report
    with per-date counts and a list of deviating dates."""
    counts = {}
    for d, f in inv.files_by_date.items():
        ds = xr.open_dataset(f, decode_times=False)
        counts[d] = int(ds.sizes.get("time", 0))
        ds.close()

    if expected is None:
        mode_count, _ = Counter(counts.values()).most_common(1)[0]
        expected = mode_count

    deviations = [
        {"date": d, "available": c, "expected": expected}
        for d, c in sorted(counts.items()) if c != expected
    ]

    log.info("%s: expected records/day = %d (auto-detected mode).", inv.label, expected)
    if deviations:
        log.warning("%s: %d day(s) deviate from the expected record count:", inv.label, len(deviations))
        for dev in deviations:
            log.warning("    %s: available=%d expected=%d", dev["date"].isoformat(),
                        dev["available"], dev["expected"])
    else:
        log.info("%s: all %d present days have exactly %d records -- daily completeness OK.",
                  inv.label, len(counts), expected)

    return {"expected": expected, "counts": counts, "deviations": deviations}


# --------------------------------------------------------------------------
# STEP 4a -- Wind daily aggregation
# --------------------------------------------------------------------------

def aggregate_wind_file_to_daily(path: Path, day: date) -> xr.Dataset:
    """Open one daily 6-hourly wind file and reduce it to a single daily
    record: mean(uwnd), mean(vwnd), mean(ws), sum(nobs), plus the derived
    ws_from_daily_mean_components diagnostic. Fill values are already NaN
    after CF decoding, so `.mean(skipna=True)` naturally excludes them at
    the per-grid-cell level."""
    ds = xr.open_dataset(path, chunks={})  # dask-backed, standard CF decode

    n_records = ds.sizes["time"]

    daily_uwnd = ds["uwnd"].mean(dim="time", skipna=True, keep_attrs=True)
    daily_vwnd = ds["vwnd"].mean(dim="time", skipna=True, keep_attrs=True)
    daily_ws = ds["ws"].mean(dim="time", skipna=True, keep_attrs=True)
    daily_ws_from_components = np.sqrt(daily_uwnd ** 2 + daily_vwnd ** 2)
    daily_nobs = ds["nobs"].sum(dim="time", skipna=True, keep_attrs=True)

    daily_uwnd.attrs.update({
        "daily_aggregation": "mean of valid (non-fill) 6-hourly values for this UTC day",
        "source_temporal_resolution": "6-hourly (00,06,12,18 UTC)",
        "n_records_used": n_records,
    })
    daily_vwnd.attrs.update({
        "daily_aggregation": "mean of valid (non-fill) 6-hourly values for this UTC day",
        "source_temporal_resolution": "6-hourly (00,06,12,18 UTC)",
        "n_records_used": n_records,
    })
    daily_ws.attrs.update({
        "daily_aggregation": (
            "mean of the reported 6-hourly wind-speed magnitudes for this UTC day. "
            "NOTE: this is NOT the same quantity as the speed of the daily-mean "
            "vector -- see 'ws_from_daily_mean_components' for that."
        ),
        "source_temporal_resolution": "6-hourly (00,06,12,18 UTC)",
        "n_records_used": n_records,
    })
    daily_ws_from_components.attrs = {
        "long_name": "daily wind speed computed from daily-mean u/v components",
        "units": "m s-1",
        "formula": "sqrt(daily_uwnd**2 + daily_vwnd**2)",
        "standard_name": "wind_speed",
        "note": "Derived diagnostic; distinct from 'ws' which is the mean of instantaneous speeds.",
    }
    daily_nobs.attrs.update({
        "daily_aggregation": (
            "sum of 'number of observations used to derive wind vector components' "
            "across the day's 6-hourly analyses -- represents total observation "
            "count contributing to this day's wind field."
        ),
        "source_temporal_resolution": "6-hourly (00,06,12,18 UTC)",
        "n_records_used": n_records,
    })

    day_ts = pd.Timestamp(day.isoformat())  # 00:00:00 UTC of that calendar day

    out = xr.Dataset(
        data_vars={
            "uwnd": daily_uwnd,
            "vwnd": daily_vwnd,
            "ws": daily_ws,
            "ws_from_daily_mean_components": daily_ws_from_components,
            "nobs": daily_nobs,
        },
    )
    out = out.expand_dims(time=[day_ts])
    out["n_records_per_day"] = xr.DataArray(
        np.array([n_records], dtype=np.int32), dims=["time"],
        attrs={"long_name": "number of 6-hourly records aggregated into this day"},
    )
    out.attrs = dict(ds.attrs)  # carry original global attrs forward as a base
    ds.close()
    return out


def merge_winds_daily(inv: Inventory, daily_check: dict) -> xr.Dataset:
    log.info("winds: aggregating %d daily files to one record/day each...", len(inv.sorted_files))
    pieces = [aggregate_wind_file_to_daily(inv.files_by_date[d], d) for d in inv.sorted_dates]
    merged = xr.concat(pieces, dim="time", combine_attrs="drop_conflicts")
    merged = merged.sortby("time")

    _, unique_idx = np.unique(merged["time"].values, return_index=True)
    if len(unique_idx) != merged.sizes["time"]:
        n_dupes = merged.sizes["time"] - len(unique_idx)
        log.warning("winds: dropping %d duplicate daily time step(s) (keeping first).", n_dupes)
        merged = merged.isel(time=np.sort(unique_idx))

    merged.attrs.update({
        "source_consolidation": (
            f"Consolidated from {len(inv.sorted_files)} original daily 6-hourly NetCDF "
            f"files ({inv.sorted_dates[0].isoformat()} to {inv.sorted_dates[-1].isoformat()}) "
            "by merge_2025_winds_currents.py."
        ),
        "aggregation_method": (
            "Wind fields were TEMPORALLY AGGREGATED from 6-hourly to daily: "
            "uwnd/vwnd/ws are the mean of the day's valid 6-hourly values; "
            "nobs is the sum across the day's 6-hourly values; "
            "ws_from_daily_mean_components = sqrt(daily_uwnd**2 + daily_vwnd**2) "
            "was added as a derived, vector-consistent daily speed diagnostic. "
            "Output values are NOT bit-identical to the raw 6-hourly source files."
        ),
        "source_temporal_resolution": "6-hourly (00,06,12,18 UTC)",
        "output_temporal_resolution": "daily (one record per UTC calendar day, timestamp = 00:00:00 UTC)",
        "expected_records_per_day": daily_check["expected"],
        "consolidation_created": datetime.now(timezone.utc).isoformat(),
    })
    return merged


# --------------------------------------------------------------------------
# STEP 4b -- Currents merge (unchanged: lossless concatenation)
# --------------------------------------------------------------------------

def merge_currents(inv: Inventory, time_chunks: int) -> xr.Dataset:
    files = [str(p) for p in inv.sorted_files]
    log.info("currents: opening %d files with open_mfdataset (combine='nested', concat_dim='time')...",
              len(files))

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        data_vars="minimal", coords="minimal", compat="equals", join="exact",
        decode_times=time_coder, chunks={"time": time_chunks},
        combine_attrs="drop_conflicts",
    )
    ds = ds.sortby("time")

    _, unique_idx = np.unique(ds["time"].values, return_index=True)
    if len(unique_idx) != ds.sizes["time"]:
        n_dupes = ds.sizes["time"] - len(unique_idx)
        log.warning("currents: dropping %d duplicate time step(s) (keeping first).", n_dupes)
        ds = ds.isel(time=np.sort(unique_idx))

    ds.attrs.update({
        "source_consolidation": (
            f"Consolidated from {len(files)} original daily NetCDF files "
            f"({inv.sorted_dates[0].isoformat()} to {inv.sorted_dates[-1].isoformat()}) "
            "by merge_2025_winds_currents.py. No scientific values were modified: "
            "files were concatenated along the time dimension only (lossless)."
        ),
        "aggregation_method": "None -- one native daily observation per file, concatenated as-is.",
        "output_temporal_resolution": "daily (one record per UTC calendar day, as provided by OSCAR)",
        "consolidation_created": datetime.now(timezone.utc).isoformat(),
    })
    return ds


# --------------------------------------------------------------------------
# STEP 5 -- Write
# --------------------------------------------------------------------------

def write_dataset(ds: xr.Dataset, out_path: Path, time_encoding: dict,
                   overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        alt = out_path.with_name(out_path.stem + "_v2" + out_path.suffix)
        log.warning("Output already exists at %s and --overwrite not set. "
                    "Writing to %s instead.", out_path, alt)
        out_path = alt

    encoding = {}
    for var in ds.data_vars:
        if var == "n_records_per_day":
            continue
        enc = {}
        fill = ds[var].encoding.get("_FillValue", ds[var].attrs.get("_FillValue"))
        if fill is not None:
            enc["_FillValue"] = fill
        enc["zlib"] = True
        enc["complevel"] = 4
        encoding[var] = enc
    encoding["time"] = time_encoding

    log.info("Writing merged dataset to %s ...", out_path)
    ds.to_netcdf(out_path, encoding=encoding, engine="netcdf4")
    log.info("Write complete: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


# --------------------------------------------------------------------------
# STEP 6 -- Validate merged outputs
# --------------------------------------------------------------------------

def validate_output(out_path: Path, inv: Inventory, first_file_info: dict, year: int,
                     wind_extra: Optional[dict] = None) -> None:
    log.info("=" * 78)
    log.info("VALIDATION REPORT: %s", out_path)
    log.info("=" * 78)

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(out_path, decode_times=time_coder)
    try:
        log.info("File size: %.1f MB", out_path.stat().st_size / 1e6)
        log.info("Dimensions: %s", dict(ds.sizes))
        data_vars = [v for v in ds.data_vars if v != "n_records_per_day"]
        log.info("Variables: %s", data_vars)

        t = ds["time"].values
        n_steps = len(t)
        log.info("Time steps: %d", n_steps)
        log.info("First timestamp: %s", t[0])
        log.info("Last timestamp:  %s", t[-1])
        dates_seen = sorted({pd.Timestamp(str(x)).date() for x in t})
        n_unique = len(dates_seen)
        log.info("Unique daily timestamps: %d", n_unique)
        log.info("Duplicate timestamps: %d", n_steps - n_unique)

        start, end = date(year, 1, 1), date(year, 12, 31)
        all_days = set()
        d = start
        while d <= end:
            all_days.add(d)
            d += timedelta(days=1)
        missing = sorted(all_days - set(dates_seen))
        log.info("Missing dates vs full %d coverage: %d%s", year, len(missing),
                  f" (first few: {[x.isoformat() for x in missing[:5]]})" if missing else "")
        exactly_365 = (n_unique == len(all_days)) and (n_steps == n_unique)
        log.info("Contains exactly %d unique daily entries with no duplicates: %s",
                  len(all_days), exactly_365)

        for cname in ds.coords:
            if cname == "time":
                continue
            vals = ds.coords[cname].values
            if np.issubdtype(vals.dtype, np.number):
                log.info("%s range: [%s, %s]  (%d points)", cname, vals.min(), vals.max(), len(vals))

        for var in data_vars:
            da = ds[var]
            fill = da.encoding.get("_FillValue", da.attrs.get("_FillValue"))
            vals = da.values
            if fill is not None and not np.isnan(fill):
                n_missing = int(np.sum(vals == fill))
            else:
                n_missing = int(np.sum(np.isnan(vals))) if np.issubdtype(vals.dtype, np.floating) else 0
            pct = 100.0 * n_missing / vals.size
            log.info("  %s: units=%s missing=%d/%d (%.3f%%)", var, da.attrs.get("units"),
                      n_missing, vals.size, pct)

        orig_vars = set(first_file_info["data_vars"].keys())
        merged_vars = set(data_vars)
        if wind_extra is not None:
            expected_extra = {"ws_from_daily_mean_components"}
            unexpected_new = merged_vars - orig_vars - expected_extra
            missing_from_merge = orig_vars - merged_vars
            if unexpected_new or missing_from_merge:
                log.warning("Variable set differs from original beyond documented additions: "
                            "unexpected_new=%s missing=%s", unexpected_new, missing_from_merge)
            else:
                log.info("Variable set matches original plus documented derived variable(s): %s",
                          sorted(merged_vars))
        else:
            if orig_vars != merged_vars:
                log.warning("VARIABLE MISMATCH vs original file: original=%s merged=%s",
                            orig_vars, merged_vars)
            else:
                log.info("Variable set matches original representative file: %s", sorted(orig_vars))

        if wind_extra is not None:
            log.info("--- Wind-specific aggregation report ---")
            log.info("  Original number of files: %d", wind_extra["n_files"])
            log.info("  Original total 6-hourly time steps: %d", wind_extra["n_raw_time_steps"])
            log.info("  Expected records/day: %d", wind_extra["expected_records_per_day"])
            log.info("  Days aggregated: %d", n_unique)
            n_deviant = len(wind_extra["deviations"])
            log.info("  Days with != expected record count: %d", n_deviant)
            if n_deviant:
                for dv in wind_extra["deviations"][:10]:
                    log.info("    %s: available=%d expected=%d",
                              dv["date"].isoformat(), dv["available"], dv["expected"])
            log.info("  uwnd/vwnd aggregated as daily means: True")
            log.info("  ws aggregated as daily mean of reported speeds: True")
            log.info("  ws_from_daily_mean_components (vector-consistent) present: %s",
                      "ws_from_daily_mean_components" in merged_vars)
            log.info("  nobs aggregation: sum across the day's 6-hourly records "
                      "(see module docstring for rationale)")
            log.info("  Aggregation method attr on file: %s", ds.attrs.get("aggregation_method"))
        else:
            log.info("Aggregation method attr on file: %s", ds.attrs.get("aggregation_method"))
    finally:
        ds.close()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def process_winds(directory: Path, year: int, output_path: Path, dry_run: bool,
                   overwrite: bool, allow_incomplete_days: bool,
                   expected_records_per_day: Optional[int]) -> bool:
    log.info("#" * 78)
    log.info("# WINDS (daily aggregation from 6-hourly)")
    log.info("#" * 78)

    inv = discover_files(directory, "winds")
    coverage_ok = report_coverage(inv, year)
    if not inv.sorted_dates:
        log.error("winds: nothing to do, no valid files found.")
        return False

    rep_infos = inspect_representative_files(inv)
    first_info = rep_infos[0]
    lat_name, lon_name = find_lat_lon_names(first_info)
    log.info("winds: identified coordinate names lat=%r lon=%r", lat_name, lon_name)

    consistency = validate_consistency(inv, lat_name=lat_name, lon_name=lon_name)
    if (not consistency["var_set_consistent"]) or (not consistency["spatial_dims_consistent"]):
        log.error("winds: STOPPING -- variable set or SPATIAL dimensions are inconsistent "
                   "across files. Refusing to merge silently.")
        return False

    daily_check = check_daily_record_counts(inv, expected=expected_records_per_day)
    if daily_check["deviations"] and not allow_incomplete_days:
        log.error("winds: STOPPING -- %d day(s) have an unexpected record count and "
                   "--allow-incomplete-days was not set. No aggregation or merge performed. "
                   "Re-run with --allow-incomplete-days to proceed anyway (affected days will "
                   "be averaged over whatever records ARE present).", len(daily_check["deviations"]))
        return False
    elif daily_check["deviations"]:
        log.warning("winds: proceeding DESPITE %d incomplete day(s) because "
                     "--allow-incomplete-days was set.", len(daily_check["deviations"]))

    if not coverage_ok:
        log.warning("winds: coverage is incomplete or has duplicate dates -- output will not "
                     "contain all 365 days. See missing-dates list above.")

    if dry_run:
        log.info("winds: --dry-run set, skipping aggregation/write/validate steps.")
        return coverage_ok and not daily_check["deviations"]

    merged = merge_winds_daily(inv, daily_check)

    time_encoding = {"units": "days since 1970-01-01", "calendar": "standard", "dtype": "int32"}
    final_path = write_dataset(merged, output_path, time_encoding=time_encoding, overwrite=overwrite)
    merged.close()

    n_raw_time_steps = sum(v for v in daily_check["counts"].values())
    wind_extra = {
        "n_files": len(inv.sorted_files),
        "n_raw_time_steps": n_raw_time_steps,
        "expected_records_per_day": daily_check["expected"],
        "deviations": daily_check["deviations"],
    }
    validate_output(final_path, inv, first_info, year, wind_extra=wind_extra)
    return True


def process_currents(directory: Path, year: int, output_path: Path, dry_run: bool,
                      overwrite: bool, time_chunks: int) -> bool:
    log.info("#" * 78)
    log.info("# CURRENTS (lossless concatenation, already one record/day)")
    log.info("#" * 78)

    inv = discover_files(directory, "currents")
    coverage_ok = report_coverage(inv, year)
    if not inv.sorted_dates:
        log.error("currents: nothing to do, no valid files found.")
        return False

    rep_infos = inspect_representative_files(inv)
    first_info = rep_infos[0]
    lat_name, lon_name = find_lat_lon_names(first_info)
    log.info("currents: identified coordinate names lat=%r lon=%r", lat_name, lon_name)

    consistency = validate_consistency(inv, lat_name=lat_name, lon_name=lon_name)
    if (not consistency["var_set_consistent"]) or (not consistency["spatial_dims_consistent"]):
        log.error("currents: STOPPING -- variable set or spatial dimensions are inconsistent "
                   "across files. Refusing to merge silently.")
        return False

    # currents should already be exactly one record/day; verify explicitly
    daily_check = check_daily_record_counts(inv, expected=1)
    if daily_check["deviations"]:
        log.error("currents: STOPPING -- %d file(s) do NOT contain exactly 1 time record "
                   "(unexpected for OSCAR daily files). Refusing to silently average currents.",
                   len(daily_check["deviations"]))
        return False

    if not coverage_ok:
        log.warning("currents: coverage is incomplete or has duplicate dates -- output will not "
                     "contain all 365 days. See missing-dates list above.")

    if dry_run:
        log.info("currents: --dry-run set, skipping merge/write/validate steps.")
        return coverage_ok

    merged = merge_currents(inv, time_chunks=time_chunks)

    sample_ds = xr.open_dataset(inv.sorted_files[0], decode_times=False)
    time_units = sample_ds["time"].attrs.get("units")
    time_calendar = sample_ds["time"].attrs.get("calendar", "standard")
    time_dtype = str(sample_ds["time"].dtype)
    sample_ds.close()
    time_encoding = {"units": time_units, "calendar": time_calendar, "dtype": time_dtype}

    final_path = write_dataset(merged, output_path, time_encoding=time_encoding, overwrite=overwrite)
    merged.close()
    validate_output(final_path, inv, first_info, year, wind_extra=None)
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Consolidate 2025 wind/current NetCDF files into single daily time-series files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--winds-dir", type=Path, required=True)
    p.add_argument("--currents-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--dry-run", action="store_true",
                    help="Inspect and validate only; do not aggregate, merge, or write anything.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--time-chunks", type=int, default=20,
                    help="Dask chunk size along time for the currents concat (default: 20).")
    p.add_argument("--allow-incomplete-days", action="store_true",
                    help="For winds: proceed with daily aggregation even if some days have "
                         "fewer/more than the expected number of 6-hourly records (averages "
                         "over whatever is present instead of stopping).")
    p.add_argument("--expected-records-per-day", type=int, default=None,
                    help="Override the auto-detected expected wind records/day (default: "
                         "auto-detect via mode across all files, normally 4).")
    p.add_argument("--skip-winds", action="store_true")
    p.add_argument("--skip-currents", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ok = True

    if not args.skip_winds:
        ok &= process_winds(
            directory=args.winds_dir, year=args.year,
            output_path=args.output_dir / f"{args.year}_winds.nc",
            dry_run=args.dry_run, overwrite=args.overwrite,
            allow_incomplete_days=args.allow_incomplete_days,
            expected_records_per_day=args.expected_records_per_day,
        )

    if not args.skip_currents:
        ok &= process_currents(
            directory=args.currents_dir, year=args.year,
            output_path=args.output_dir / f"{args.year}_currents.nc",
            dry_run=args.dry_run, overwrite=args.overwrite, time_chunks=args.time_chunks,
        )

    if ok:
        log.info("DONE. Outputs %s ready for the next preprocessing stage.",
                  "are" if not args.dry_run else "would be (dry-run, nothing written)")
    else:
        log.error("Completed with failures / incomplete data -- see warnings/errors above. "
                   "Full-year production merge should NOT proceed until this is clean.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
