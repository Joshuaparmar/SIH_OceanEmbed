"""
config_sst.py
=============
Single source of truth for the OceanEmbed SST preprocessing stage
(SIH26066). preprocess_sst.py, netcdf_utils.py, and validation.py never
hard-code a file path, coordinate name, region bound, or variable name
-- everything dataset-specific lives here.

IMPORTANT -- this config does NOT assume any coordinate/variable name,
filename, or unit is correct. Every *_candidates / *_glob field is a
discovery mechanism to try against the real file(s) at runtime, not a
verified fact. No real SST file was opened in the environment this
config was authored in to confirm any of it against -- see README.md
"What has and hasn't actually been run".

This module is independently importable/executable and does not import
anything from preprocessing/temp_preprocess_trial3/. Some structural
patterns (dataclass config, *_candidates discovery, fatal-on-regrid-
failure, atomic writes) are deliberately reused from that pipeline for
project-wide consistency; SST-specific concerns (no depth dimension,
Kelvin<->Celsius unit handling, land-mask semantics, normalization/
statistics for model input) are new here, not copied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Bump when this config's shape or preprocess_sst.py's behavior changes
# in a way that could affect output -- recorded in every run's metadata
# JSON so an output file is traceable to the code that made it.
PIPELINE_VERSION = "0.2.0"  # 0.2.0: memory-safe lazy diagnostics, batched regridding, end-inclusive
                            # date handling, strict official spatial coverage, stricter unit
                            # validation, input/output collision check, missingness before/after
                            # subsetting/regridding. 0.1.0 = initial version.


@dataclass
class SSTConfig:
    # ------------------------------------------------------------------
    # 1. INPUT FILE(S)
    # ------------------------------------------------------------------
    # PRIMARY discovery mechanism: a glob, not a claimed filename.
    # Matches the OSTIA/CMEMS naming convention actually observed for
    # this project (see README "Real filename observed"), applied
    # broadly under data/raw/sst/. NOT narrowed to one exact filename
    # on purpose -- a year of daily L4 SST is sometimes one file,
    # sometimes split. All matches are opened together
    # (xr.open_mfdataset, combine="by_coords") if more than one.
    sst_input_glob: str = "data/raw/sst/**/*.nc"

    # Exact filenames to try first, if known. Left empty by default --
    # guessing a specific filename risks exactly the "fictional
    # filename" problem the project brief warns against. Use --input
    # to point at a known single file, or --input-glob for a known
    # multi-file pattern.
    sst_input_candidates: List[str] = field(default_factory=list)

    # Explicit override (--input / --input-glob on the CLI): used as-is
    # with no discovery/fallback.
    explicit_input_path: Optional[str] = None
    explicit_input_glob: Optional[str] = None

    # If set, treat explicit_input_path (or the first discovered path)
    # as a directory and glob *.nc inside it, rather than a single file.
    # Populated automatically by find_input_files() when a discovered
    # path turns out to be a directory -- not something callers usually
    # set directly.
    input_is_directory: bool = False

    # Project root that relative paths above are resolved against.
    project_root: str = "."

    # ------------------------------------------------------------------
    # 2. VARIABLE / COORDINATE NAME CANDIDATES (source file detection)
    # ------------------------------------------------------------------
    # "analysed_sst" first: the confirmed CMEMS/OSTIA variable name
    # (METOFFICE-GLO-SST-L4-REP-OBS-SST product). Others are fallback
    # candidates for a differently-packaged SST source, never assumed
    # correct without checking ds.data_vars first.
    sst_var_candidates: List[str] = field(
        default_factory=lambda: [
            "analysed_sst", "sst", "sea_surface_temperature", "SST", "temperature",
        ]
    )
    time_name_candidates: List[str] = field(
        default_factory=lambda: ["time", "TIME", "Time"]
    )
    latitude_name_candidates: List[str] = field(
        default_factory=lambda: ["latitude", "lat", "nav_lat", "Latitude"]
    )
    longitude_name_candidates: List[str] = field(
        default_factory=lambda: ["longitude", "lon", "nav_lon", "Longitude"]
    )
    # OSTIA-style categorical land/sea/ice mask and/or per-pixel
    # analysis-error variable, if present. Optional: detected and
    # reported (and used to distinguish land-masked cells from missing
    # observations, see G/H in the brief) but the pipeline does not
    # fail if none of these are found -- not every SST source ships one.
    mask_var_candidates: List[str] = field(
        default_factory=lambda: ["mask", "land_sea_mask", "lsm"]
    )
    error_var_candidates: List[str] = field(
        default_factory=lambda: ["analysis_error", "analysed_sst_error"]
    )
    sea_ice_var_candidates: List[str] = field(
        default_factory=lambda: ["sea_ice_fraction", "sea_ice_frac", "ice_conc"]
    )

    # ------------------------------------------------------------------
    # 2b. OUTPUT COORDINATE / VARIABLE NAMES -- common-grid compatibility
    # ------------------------------------------------------------------
    # No depth dimension for SST (unlike the subsurface temperature
    # target) -- output is (time, latitude, longitude) only. These
    # names match what the temperature pipeline standardizes to
    # (config_temperature.py's output_* fields), so downstream
    # tensor-generation code can use one shared coordinate convention
    # across variables. NOT re-verified here against a Step-5-equivalent
    # consumer for SST specifically -- record this as an assumption if
    # that consumer's exact expected names differ.
    output_time_name: str = "time"
    output_latitude_name: str = "latitude"
    output_longitude_name: str = "longitude"
    output_variable_name: str = "sst"

    # ------------------------------------------------------------------
    # 3. REGION / TIME
    # ------------------------------------------------------------------
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0

    # Internal-round target is stated as "2025" without exact start/end
    # dates -- this defaults to the full calendar year (standard for a
    # train/val/test split), same assumption made by the temperature
    # pipeline. Set date_range_confirmed=True once the exact intended
    # range is confirmed by the project.
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    date_range_confirmed: bool = False

    # Spatial overlap: lenient by default at the low-level subsetting
    # step -- partial coverage right at the region's edges is common
    # and rarely a real problem there; actual overlap achieved is
    # always reported either way. THIS DOES NOT govern the official
    # output requirement below -- see require_full_official_coverage.
    allow_partial_spatial_overlap: bool = True

    # STRICT by default, and independent of allow_partial_spatial_overlap
    # above: for an OFFICIAL (regridded) run, the source file must fully
    # cover the requested 5-30N/45-105E region. A source that only
    # partially covers the region would otherwise be silently regridded
    # into a model-ready file with an undocumented coverage gap -- that
    # is exactly the kind of "silently accept a partial source" behavior
    # the project brief prohibits for official output. Only relevant
    # when regrid_to_quarter_degree=True; a --no-regrid diagnostic run
    # is never "official" regardless of coverage, so this is not checked
    # there (allow_partial_spatial_overlap alone governs that path).
    require_full_official_coverage: bool = True

    # Temporal coverage: STRICT by default for an official run -- an
    # incomplete date range must not be silently processed and labeled
    # complete. Set False only for a deliberate partial-range run (and
    # set date_range_confirmed=True with a narrower start_date/end_date
    # so that's explicit too).
    require_complete_temporal_coverage: bool = True
    # None = auto-computed as (end_date - start_date).days + 1 assuming
    # daily data. Set explicitly if a different count is intentional.
    expected_time_step_count: Optional[int] = None

    # ------------------------------------------------------------------
    # 4. SPATIAL RESOLUTION / REGRIDDING
    # ------------------------------------------------------------------
    # ON by default: the OceanEmbed common grid is 0.25 x 0.25 deg and
    # the official SST output must be on it, not the ~0.05 deg native
    # OSTIA grid. Regridding failure for ANY reason is FATAL (see
    # netcdf_utils.regrid_to_target_grid) -- no try/except-and-continue
    # around it anywhere. Pass --no-regrid for an explicit, clearly-
    # marked non-official native-grid DIAGNOSTIC run (written to
    # native_diagnostic_subdir, never the official output path).
    regrid_to_quarter_degree: bool = True
    target_resolution_deg: float = 0.25
    regrid_method: str = "linear"  # "linear" or "nearest" (scipy-backed via xarray.interp)
    # Cell-center convention, both endpoints included:
    # n_points = round((max-min)/spacing) + 1. For the configured
    # 5-30N / 45-105E bounds at 0.25 deg this gives 101 x 241 points --
    # see netcdf_utils.build_target_grid(). Calculated, not assumed;
    # verify_final_grid() re-checks this against the actual output.
    # No extrapolation: target cells outside the source data's actual
    # coverage are left NaN (bounds_error=False, fill_value=NaN passed
    # to the interpolator), never filled with an edge/extrapolated value.
    final_grid_tolerance_deg: float = 1e-6
    native_diagnostic_subdir: str = "native_diagnostic"

    # Regridding is performed in TIME BATCHES to bound peak memory:
    # rather than interpolating the entire subsetted (potentially
    # native-~0.05-deg-resolution, full-year) array in one call, the
    # time dimension is sliced into batches of this many steps, each
    # batch is interpolated and materialized independently, and the
    # results are concatenated. None = no batching (interpolate the
    # whole array in one call -- fine for a short test range, riskier
    # for a full year at native resolution). A full 2025 daily run
    # SHOULD set this (e.g. 30-60) rather than rely on the default.
    regrid_time_batch_size: Optional[int] = None

    # ------------------------------------------------------------------
    # 5. UNITS
    # ------------------------------------------------------------------
    # OSTIA/CMEMS analysed_sst is conventionally stored in Kelvin, but
    # this must be VERIFIED from the actual file's `units` attribute,
    # never assumed. "auto" inspects units and converts Kelvin -> Celsius
    # only if the units attribute says so (recorded explicitly in the
    # metadata and NetCDF attrs either way); if units are missing or
    # unrecognized, "auto" FAILS the run by default (see
    # fail_on_unrecognized_units) rather than guessing -- silently
    # treating an unknown unit as Celsius would risk shipping physically
    # ambiguous or wrong data. "never" disables conversion entirely
    # (output stays exactly as stored, no failure either -- an explicit,
    # permissive choice for a diagnostic/exploratory run, not the
    # official default). "force_k_to_c" always subtracts 273.15
    # regardless of the units attribute -- an EXPERT OVERRIDE: use only
    # if you have independently verified (from product documentation,
    # not guessed) that the data is Kelvin despite a missing/wrong
    # attribute; always recorded explicitly in the output metadata.
    unit_conversion_policy: str = "auto"  # "auto", "never", "force_k_to_c"
    # Only consulted when unit_conversion_policy == "auto". True (the
    # official default): missing/unrecognized units attribute is FATAL.
    # Set False to downgrade this to a warning-only skip (the previous,
    # more permissive behavior) -- not recommended for an official run,
    # since it means SST values could be written under a unit label
    # that was never actually confirmed correct.
    fail_on_unrecognized_units: bool = True

    # ------------------------------------------------------------------
    # 6. PHYSICAL-VALUE SAFETY CHECK
    # ------------------------------------------------------------------
    # Finite values outside roughly this Celsius range almost always
    # mean an unmasked _FillValue/missing_value sentinel or a units bug,
    # not real sea-surface physics. Sea water freezes around -1.8 degC,
    # so the low bound has headroom below that; the high bound has
    # headroom above the hottest observed open-ocean SST (~35-36 degC).
    # A few such values are reported as a warning; if they exceed
    # max_implausible_fraction of all finite values, the run fails
    # rather than silently shipping data that looks corrupt. Checked
    # BEFORE regridding on purpose -- see netcdf_utils docstring.
    implausible_low_c: float = -3.0
    implausible_high_c: float = 45.0
    max_implausible_fraction: float = 0.05

    # ------------------------------------------------------------------
    # 7. MISSING-VALUE POLICY
    # ------------------------------------------------------------------
    # "preserve": NaN left as-is everywhere (land + any missing
    #   observation); the recommended default. Because OSTIA is an L4
    #   (gap-filled) analysis product, ocean NaNs are not generally
    #   expected pre-regrid -- but this is verified per-file, not
    #   assumed, and land pixels are still NaN/masked.
    # "reject_excessive": same as "preserve", plus the run FAILS if the
    #   fraction of NaN cells over the subsetted region exceeds
    #   max_missing_fraction (guards against silently shipping a mostly-
    #   empty file, e.g. wrong variable or wrong region).
    # Filling is deliberately NOT offered as an automatic default policy
    # -- see fill_enabled below; SST has a real land/ocean boundary and
    # this pipeline never fabricates SST values over land.
    missing_value_policy: str = "preserve"  # "preserve" or "reject_excessive"
    max_missing_fraction: float = 0.9

    # Off by default, and only usable with an explicit, scientifically
    # named method -- never "fill everything with 0" or similar. Kept
    # for a clearly labeled, optional post-processing step; the default
    # official pipeline output always has fill_enabled=False (NaN
    # preserved, valid_mask records exactly which cells are valid).
    fill_enabled: bool = False
    fill_method: str = "nearest_ocean"  # only meaningful if fill_enabled=True

    # ------------------------------------------------------------------
    # 8. NORMALIZATION / STATISTICS
    # ------------------------------------------------------------------
    # "none": raw (unit-converted, regridded) SST values are written,
    #   no z-score/min-max applied.
    # "zscore": (x - mean) / std using statistics computed as configured
    #   below.
    # "minmax": (x - min) / (max - min) using the same statistics scope.
    normalization_method: str = "none"  # "none", "zscore", "minmax"
    # Statistics are ALWAYS computed and saved to a separate JSON
    # (stats_output_name) regardless of normalization_method, so they
    # are available for later model-development use even on an
    # un-normalized run. This also documents exactly what data they
    # were computed from -- see stats_source_description in the output.
    #
    # For the current internal-round setup (single 2025 file, no
    # separate held-out test period has been defined yet), statistics
    # are computed from the SAME subsetted/regridded data being written
    # -- this is recorded explicitly in the stats file and in README as
    # a documented limitation, not hidden. If/when a train/test time
    # split is defined for OceanEmbed, statistics MUST be recomputed
    # from the training portion only (see stats_time_range below) to
    # avoid leakage; this field exists for exactly that purpose.
    stats_time_range: Optional[List[str]] = None  # e.g. ["2025-01-01", "2025-09-30"]; None = use the whole processed range
    stats_output_name: str = "sst_statistics.json"

    # ------------------------------------------------------------------
    # 9. OUTPUT
    # ------------------------------------------------------------------
    output_dir: str = "data/processed/sst"
    output_nc_name: str = "sst_target.nc"
    output_npy_name: str = "sst_target.npy"
    output_mask_npy_name: str = "sst_valid_mask.npy"
    output_metadata_name: str = "sst_metadata.json"
    dryrun_report_name: str = "sst_dryrun_report.json"
    write_npy_outputs: bool = True
    enable_netcdf_compression: bool = True
    netcdf_complevel: int = 4
    # Chunked/lazy loading via dask (chunks="auto") when opening input
    # file(s) so a full year of daily 0.05 deg global-crop SST isn't
    # necessarily pulled fully into memory just to open and validate
    # structure.
    use_dask_chunking: bool = True
    # Writes are staged to a temp file in the same directory and
    # atomically replaced into the final path only after a fully
    # successful write, so a crash mid-write never leaves a corrupt or
    # half-written file at the final path.
    atomic_writes: bool = True

    # ------------------------------------------------------------------
    # 10. DIAGNOSTIC PLOT
    # ------------------------------------------------------------------
    save_diagnostic_plot: bool = True
    diagnostic_plot_name: str = "sst_diagnostic.png"

    # ------------------------------------------------------------------
    # 11. DRY RUN
    # ------------------------------------------------------------------
    # When True: open + detect + validate + subset + regrid and report
    # the would-be output shape/structure, but never write the final
    # processed outputs (sst_target.nc/.npy/mask/stats). Writes only a
    # dry-run report (dryrun_report_name), clearly labeled as such.
    dry_run: bool = False

    # ------------------------------------------------------------------
    # 12. LOGGING
    # ------------------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max}).")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max}).")
        if self.regrid_method not in ("linear", "nearest"):
            raise ValueError(f"regrid_method {self.regrid_method!r} not recognized (use 'linear' or 'nearest').")
        if self.unit_conversion_policy not in ("auto", "never", "force_k_to_c"):
            raise ValueError(f"unit_conversion_policy {self.unit_conversion_policy!r} not recognized.")
        if self.missing_value_policy not in ("preserve", "reject_excessive"):
            raise ValueError(f"missing_value_policy {self.missing_value_policy!r} not recognized.")
        if self.normalization_method not in ("none", "zscore", "minmax"):
            raise ValueError(f"normalization_method {self.normalization_method!r} not recognized.")
        if not (0.0 <= self.max_implausible_fraction <= 1.0):
            raise ValueError(
                f"max_implausible_fraction must be within [0, 1], got {self.max_implausible_fraction}."
            )
        if not (0.0 <= self.max_missing_fraction <= 1.0):
            raise ValueError(
                f"max_missing_fraction must be within [0, 1], got {self.max_missing_fraction}."
            )
        if self.implausible_low_c >= self.implausible_high_c:
            raise ValueError("implausible_low_c must be < implausible_high_c.")
        if self.final_grid_tolerance_deg <= 0:
            raise ValueError("final_grid_tolerance_deg must be > 0.")
        if not (0 <= self.netcdf_complevel <= 9):
            raise ValueError(f"netcdf_complevel must be within [0, 9], got {self.netcdf_complevel}.")
        if self.regrid_time_batch_size is not None and self.regrid_time_batch_size <= 0:
            raise ValueError(f"regrid_time_batch_size must be a positive integer or None, got {self.regrid_time_batch_size}.")

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config-relative path against project_root."""
        return (Path(self.project_root) / relative_path).resolve()


# Ready-to-use default instance. Import this directly, or construct your
# own SSTConfig(...) (e.g. from argparse) for a different run.
CONFIG = SSTConfig()
