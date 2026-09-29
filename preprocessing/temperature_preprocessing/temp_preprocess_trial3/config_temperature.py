"""
config_temperature.py
======================
Single source of truth for the GLORYS temperature-target preprocessing
stage. preprocess_temperature.py and netcdf_utils.py never hard-code a
file path, coordinate name, region bound, or depth list -- everything
dataset-specific lives here.

STATUS (this audit): the January 2026 15-day/monthly prototype plan is
DISCONTINUED. The active internal-round dataset is 2025 ocean data.
The old Jan-2026 paths are kept below (clearly marked OBSOLETE) for
history, not deleted, and are no longer searched by default.

IMPORTANT -- this config does NOT assume any coordinate/variable name,
filename, or date range is correct. Every *_candidates / *_glob field
is a discovery mechanism to try against the real file(s) at runtime,
not a verified fact. No real 2025 file was found in this environment
to confirm any of it against -- see README.md "What has and hasn't
actually been run".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Bump when this config's shape or preprocess_temperature.py's behavior
# changes in a way that could affect output -- recorded in every run's
# metadata JSON so an output file is traceable to the code that made it.
PIPELINE_VERSION = "0.3.0"  # 0.2.0 = 2025-active audit; 0.1.0 = initial Jan-2026-prototype version


@dataclass
class TemperatureConfig:
    # ------------------------------------------------------------------
    # 1. INPUT FILE(S) -- ACTIVE: 2025 internal-round dataset
    # ------------------------------------------------------------------
    # CHANGE ME candidates: exact filenames, if you know them. Left
    # empty by default ON PURPOSE -- no 2025 file was found in this
    # environment, and guessing a specific filename would risk exactly
    # the "fictional filename" the brief warns against. Use --input to
    # point at a known single file.
    year2025_input_candidates: List[str] = field(default_factory=list)

    # PRIMARY discovery mechanism: a glob, not a claimed filename. A
    # full year of daily GLORYS data is commonly distributed as several
    # files (e.g. one per month) rather than one -- this matches
    # multiple files and they are opened together (xr.open_mfdataset,
    # combine="by_coords"), the same pattern oceanembed_step5/
    # data_loader.py already uses for its own multi-file inputs. The
    # pattern below follows the naming convention established for the
    # (now-discontinued) Jan-2026 files -- thetao_<descriptor>.nc under
    # month_data/target_glorys/ -- applied to 2025. NOT verified against
    # a real file; adjust if the real naming differs.
    year2025_input_glob: str = "month_data/target_glorys/thetao_2025_*.nc"

    # Explicit override (--input on the CLI): a single exact path, used
    # as-is with no discovery/fallback. --input-glob (CLI) overrides
    # year2025_input_glob the same way, for a specific multi-file match.
    explicit_input_path: Optional[str] = None
    explicit_input_glob: Optional[str] = None

    # Project root that relative paths above are resolved against.
    project_root: str = "."

    # ------------------------------------------------------------------
    # 1b. OBSOLETE -- January 2026 prototype (DISCONTINUED)
    # ------------------------------------------------------------------
    # Kept for history only. NOT searched by default anymore -- do not
    # treat these as the active internal-round dataset. Still usable
    # manually via --input if you deliberately want to re-run the old
    # prototype for comparison.
    obsolete_jan2026_monthly_candidates: List[str] = field(
        default_factory=lambda: [
            "month_data/target_glorys/thetao_2026_Jan_31days.nc",
        ]
    )
    obsolete_jan2026_fifteenday_candidates: List[str] = field(
        default_factory=lambda: [
            "month_data/target_glorys/thetao_2026_Jan_15days.nc",
            "thetao_2026_Jan_15days.nc",
            "temperature_SIH15_depths.nc",
            "SIH15_model_ready.nc",
        ]
    )

    # ------------------------------------------------------------------
    # 2. VARIABLE / COORDINATE NAME CANDIDATES (source file detection)
    # ------------------------------------------------------------------
    temperature_var_candidates: List[str] = field(
        default_factory=lambda: ["thetao", "temperature", "water_temp", "votemper", "TEMP"]
    )
    time_name_candidates: List[str] = field(
        default_factory=lambda: ["time", "TIME", "Time"]
    )
    depth_name_candidates: List[str] = field(
        default_factory=lambda: ["depth", "deptht", "lev", "z", "Depth"]
    )
    latitude_name_candidates: List[str] = field(
        default_factory=lambda: ["latitude", "lat", "nav_lat", "Latitude"]
    )
    longitude_name_candidates: List[str] = field(
        default_factory=lambda: ["longitude", "lon", "nav_lon", "Longitude"]
    )

    # ------------------------------------------------------------------
    # 2b. OUTPUT COORDINATE NAMES -- Step 5 compatibility (verified)
    # ------------------------------------------------------------------
    # oceanembed_step5/config.py's CoordNames defaults to EXACTLY these
    # four strings for target_coords. The source file may call them
    # anything (see candidates above); this pipeline now STANDARDIZES
    # to these names when writing its output, regardless of the source
    # names, so Step 5 can read the output without editing its own
    # config.target_coords. This was verified by reading
    # oceanembed_step5/config.py directly, not assumed.
    output_time_name: str = "time"
    output_depth_name: str = "depth"
    output_latitude_name: str = "latitude"
    output_longitude_name: str = "longitude"

    # ------------------------------------------------------------------
    # 3. REGION / TIME
    # ------------------------------------------------------------------
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0

    # ASSUMPTION, not a confirmed project decision: "2025 ocean data"
    # was stated without exact start/end dates, so this defaults to the
    # full calendar year (standard for a train/val/test split). If the
    # real intended range is narrower, set start_date/end_date AND set
    # date_range_confirmed=True once that's been confirmed -- until
    # then, the pipeline logs a warning that this range is an assumed
    # default rather than a verified one.
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    date_range_confirmed: bool = False

    # Spatial overlap: lenient by default. Partial coverage near the
    # region's edges is common and rarely a real problem; the actual
    # overlap achieved is always reported either way.
    allow_partial_spatial_overlap: bool = True

    # Temporal coverage: STRICT by default for an official run -- the
    # brief is explicit that an incomplete date range must not be
    # silently processed and labeled complete. Set False only for a
    # deliberate partial-range run (and set date_range_confirmed=True
    # with the narrower start_date/end_date so that's explicit too).
    require_complete_temporal_coverage: bool = True
    # None = auto-computed as (end_date - start_date).days + 1 assuming
    # daily data. Set explicitly if the project intentionally expects a
    # different count.
    expected_time_step_count: Optional[int] = None

    # ------------------------------------------------------------------
    # 4. TARGET DEPTHS -- authoritative: exactly 15, not 13
    # ------------------------------------------------------------------
    requested_depths_m: List[float] = field(
        default_factory=lambda: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    )
    depth_selection_method: str = "nearest"  # "nearest" or "exact"
    # None = no distance limit (nearest match always kept, difference
    # always reported). Set a number (metres) to exclude a nearest
    # match that's too far from what was requested.
    depth_max_tolerance_m: Optional[float] = None
    # STRICT by default: the target depth count is authoritative at 15.
    # If any requested depth can't be matched (out of range, excluded
    # by tolerance, or no exact match in "exact" mode), the run fails
    # clearly rather than silently shipping fewer than 15 levels.
    require_all_requested_depths: bool = True

    # ------------------------------------------------------------------
    # 5. SPATIAL RESOLUTION / REGRIDDING
    # ------------------------------------------------------------------
    # ON by default as of this audit: "the final processed temperature
    # output must be regridded to 0.25 x 0.25 deg" is now an explicit
    # OceanEmbed project requirement, and the official pipeline must
    # not produce native-resolution output. Regridding failure for ANY
    # reason is FATAL (see netcdf_utils.regrid_to_target_grid) -- there
    # is deliberately no try/except-and-continue around it. Pass
    # --no-regrid for an explicit, clearly-marked non-official
    # native-grid DIAGNOSTIC run (written to native_diagnostic_subdir,
    # never the official output path, so it can't be mistaken for it).
    regrid_to_quarter_degree: bool = True
    target_resolution_deg: float = 0.25
    regrid_method: str = "linear"  # requires scipy (now a hard dependency of the official path)
    # Cell-center convention, both endpoints included:
    # n_points = round((max-min)/spacing) + 1. For the configured
    # 5-30N / 45-105E bounds at 0.25 deg this gives exactly 101 x 241
    # points -- see netcdf_utils.build_target_grid().
    # No extrapolation: target cells outside the source data's actual
    # coverage are left NaN (bounds_error=False, fill_value=NaN passed
    # to the interpolator), never filled with an edge/extrapolated value.
    final_grid_tolerance_deg: float = 1e-6  # numerical tolerance for post-regrid spacing/range checks
    native_diagnostic_subdir: str = "native_diagnostic"

    # ------------------------------------------------------------------
    # 6. MISSING VALUES
    # ------------------------------------------------------------------
    # Off by default. This is now a verified Step 5 COMPATIBILITY
    # requirement, not just a convenience default: oceanembed_step5/
    # preprocessing.py computes its own valid-data mask via
    # np.isfinite() directly on the temperature array -- it does not
    # read a separate mask variable. If NaN were filled with a real
    # number here, Step 5 would silently treat those filled cells as
    # valid observations. Leave this False unless Step 5's masking
    # logic is changed to match.
    fill_enabled: bool = False
    fill_value: float = 0.0

    # ------------------------------------------------------------------
    # 6b. PHYSICAL-VALUE SAFETY CHECK
    # ------------------------------------------------------------------
    # Finite values outside roughly [-5, 45] degC almost always mean an
    # unmasked _FillValue/missing_value sentinel, not real ocean
    # physics. A few such values get reported as a warning; if they
    # make up more than this fraction of all finite values, something
    # is more seriously wrong (units, masking, wrong variable) and the
    # run fails rather than silently shipping data that looks corrupt.
    # Checked BEFORE regridding on purpose: an unmasked sentinel value
    # (e.g. 1e20) would otherwise get smeared into real neighbouring
    # ocean cells by linear interpolation, corrupting them.
    implausible_low_c: float = -5.0
    implausible_high_c: float = 45.0
    max_implausible_fraction: float = 0.05

    # ------------------------------------------------------------------
    # 7. OUTPUT
    # ------------------------------------------------------------------
    output_dir: str = "data/processed/temperature"
    output_nc_name: str = "temperature_target.nc"
    output_npy_name: str = "temperature_target.npy"
    output_mask_npy_name: str = "temperature_valid_mask.npy"
    output_metadata_name: str = "temperature_metadata.json"
    dryrun_report_name: str = "temperature_dryrun_report.json"
    write_npy_outputs: bool = True
    # zlib compression on the two data variables in the NetCDF output.
    # A full 2025 x 15-depth x 101x241 float32 array is roughly half a
    # gigabyte uncompressed; compression trades a little CPU for
    # meaningfully less disk, and xarray/netCDF4 support it natively --
    # no new dependency.
    enable_netcdf_compression: bool = True
    netcdf_complevel: int = 4
    # Chunked/lazy loading via dask (chunks="auto") when opening input
    # file(s), mirroring oceanembed_step5/data_loader.py's own use of
    # chunked xarray objects, so the whole year isn't necessarily
    # pulled into memory just to open and validate structure. Uses
    # "auto" (not a fixed per-dimension chunk) because the real
    # dimension names aren't known until after detect_all_coords() --
    # this is decided at open time, before detection runs.
    use_dask_chunking: bool = True
    # Writes are staged to a temp file in the same directory and
    # atomically replaced into the final path only after a fully
    # successful write, so a crash mid-write never leaves a corrupt or
    # half-written file at the final path. See netcdf_utils.atomic_write.
    atomic_writes: bool = True

    # ------------------------------------------------------------------
    # 8. DIAGNOSTIC PLOT
    # ------------------------------------------------------------------
    save_diagnostic_plot: bool = True
    diagnostic_plot_name: str = "temperature_diagnostic.png"

    # ------------------------------------------------------------------
    # 9. DRY RUN
    # ------------------------------------------------------------------
    # When True: open + detect + validate + subset + select depths and
    # report the would-be output shape/structure, but never write the
    # final processed outputs (temperature_target.nc/.npy/mask). Writes
    # only a dryrun report (dryrun_report_name), clearly labeled as
    # such, so it's never confused with a completed run's metadata.
    dry_run: bool = False

    # ------------------------------------------------------------------
    # 10. LOGGING
    # ------------------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.depth_selection_method not in ("nearest", "exact"):
            raise ValueError(
                f"depth_selection_method must be 'nearest' or 'exact', "
                f"got {self.depth_selection_method!r}."
            )
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max}).")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max}).")
        if not self.requested_depths_m:
            raise ValueError("requested_depths_m must not be empty.")
        if self.regrid_method not in ("linear", "nearest", "cubic"):
            raise ValueError(f"regrid_method {self.regrid_method!r} not recognized.")
        if not (0.0 <= self.max_implausible_fraction <= 1.0):
            raise ValueError(
                f"max_implausible_fraction must be within [0, 1], got {self.max_implausible_fraction}."
            )
        if self.implausible_low_c >= self.implausible_high_c:
            raise ValueError("implausible_low_c must be < implausible_high_c.")
        if self.final_grid_tolerance_deg <= 0:
            raise ValueError("final_grid_tolerance_deg must be > 0.")
        if not (0 <= self.netcdf_complevel <= 9):
            raise ValueError(f"netcdf_complevel must be within [0, 9], got {self.netcdf_complevel}.")

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config-relative path against project_root."""
        return (Path(self.project_root) / relative_path).resolve()


# Ready-to-use default instance. Import this directly, or construct your
# own TemperatureConfig(...) (e.g. from argparse) for a different run.
CONFIG = TemperatureConfig()
