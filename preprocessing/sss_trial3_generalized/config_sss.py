"""
config_sss.py
=============
Single source of truth for the SIH26066 / OceanEmbed Sea Surface
Salinity (SSS) preprocessing stage.

CHANGELOG
---------
0.2.1 (this version): Reject mixed-calendar-year input by default.
  - detect_year() now raises ValueError if a file's time coordinate
    spans more than one calendar year, instead of silently assigning
    it to the dominant year. New config.allow_multi_year (default
    False) is an explicit, advanced opt-out (--allow-multi-year) that
    still assigns the whole file to its dominant year (does not split
    it) and records a loud warning. Nothing else changed -- regridding
    algorithm, min_valid_weight, unit handling, target grid, region,
    missing-value/mask behavior, extrapolation policy, and output
    variable name are all untouched (verified by an identical-input
    regression diff against 0.2.0's output).
0.2.0: Generalized to be YEAR-AGNOSTIC.
  - Removed the 2025-specific default input path, start_date/end_date,
    and expected_time_step_count=365 that 0.1.0 hard-coded.
  - Year is now DETECTED from the input file's time coordinate (with
    the filename used only as a cross-check, never as the source of
    truth) -- see netcdf_utils.detect_year().
  - require_complete_temporal_coverage now defaults to False: a
    partial-year file is processed and reported, not rejected, unless
    explicitly required to be complete.
  - Output paths are now templates ({year} substituted at runtime),
    not fixed 2025 filenames.
  - Added explicit unit_interpretation_status / unit_documentation_reference
    / unit_interpretation_notes fields (Section 7 of the generalization
    brief) -- assume_psu_equivalent continues to default to False and,
    even when enabled, never changes a numerical value, only metadata.
  - Added output_variable_name as an explicit, documented config option
    (no existing "tensor pipeline" convention was available to inspect
    in this environment -- see README "Output variable naming").
0.1.0: initial SSS pipeline (2025-only, weighted-linear regrid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

PIPELINE_VERSION = "0.2.1"


@dataclass
class SSSConfig:
    # ------------------------------------------------------------------
    # 1. INPUT FILE -- year-agnostic: no default year-specific path.
    # ------------------------------------------------------------------
    explicit_input_path: Optional[str] = None
    project_root: str = "."

    # Optional override: force the year used for output naming/reporting
    # instead of deriving it from the file's time coordinate. Normally
    # left None (derive from data). If both this and the filename imply
    # different years than the data, that mismatch is reported as a
    # warning, never silently resolved one way or the other.
    expected_year: Optional[int] = None

    # SAFETY DEFAULT (OFF): if a single input file's time coordinate
    # spans MORE THAN ONE calendar year, detect_year() raises rather
    # than silently assigning the whole file to its dominant year. This
    # pipeline is built around "one calendar year's data per input
    # file, per output file" -- a mixed-year file being silently
    # folded into a single dominant-year output would misrepresent
    # what that output actually contains. Setting this True is an
    # ADVANCED / NON-STANDARD override (--allow-multi-year) -- it is
    # NOT used by normal, one-file-per-year OceanEmbed preprocessing,
    # and when used it still assigns the whole file to the dominant
    # year (does NOT split the file), with a loud warning recorded in
    # metadata.
    allow_multi_year: bool = False

    # ------------------------------------------------------------------
    # 2. VARIABLE / COORDINATE NAME CANDIDATES (source file detection)
    # ------------------------------------------------------------------
    sss_var_candidates: List[str] = field(
        default_factory=lambda: ["sos", "sss", "salinity", "so", "sea_surface_salinity"]
    )
    cf_standard_name: str = "sea_surface_salinity"
    cf_long_name_substring: str = "sea surface salinity"

    time_name_candidates: List[str] = field(default_factory=lambda: ["time", "TIME", "Time"])
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
    # 2b. OUTPUT VARIABLE / COORDINATE NAMES
    # ------------------------------------------------------------------
    # No existing "tensor pipeline" was available to inspect in this
    # environment to confirm a shared downstream naming convention (see
    # README "Output variable naming"). 'sss' is the documented,
    # configurable default; source_variable / output_variable are both
    # recorded separately in metadata regardless.
    output_variable_name: str = "sss"
    output_time_name: str = "time"
    output_latitude_name: str = "latitude"
    output_longitude_name: str = "longitude"

    # ------------------------------------------------------------------
    # 3. REGION -- authoritative OceanEmbed project grid, unchanged
    #    across years.
    # ------------------------------------------------------------------
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0

    normalize_longitude_convention: bool = True
    normalize_descending_coordinates: bool = True
    allow_partial_spatial_overlap: bool = True

    # ------------------------------------------------------------------
    # 3b. TIME -- year-agnostic. No fixed start_date/end_date/expected
    # count: the actual dates present in the source file are preserved
    # and reported, never fabricated or padded to a full year.
    # ------------------------------------------------------------------
    require_complete_temporal_coverage: bool = False

    # ------------------------------------------------------------------
    # 4. DEPTH -- SSS has a singleton surface depth dimension.
    # ------------------------------------------------------------------
    expected_surface_depth_m: float = 0.0
    surface_depth_tolerance_m: float = 1e-3

    # ------------------------------------------------------------------
    # 5. SPATIAL RESOLUTION / REGRIDDING -- unchanged scientific
    #    configuration, applies identically to every year.
    # ------------------------------------------------------------------
    regrid_to_quarter_degree: bool = True
    target_resolution_deg: float = 0.25
    regrid_method: str = "weighted_linear"  # only supported value -- see __post_init__
    final_grid_tolerance_deg: float = 1e-6

    # Minimum fraction of a target cell's bilinear interpolation weight
    # that must come from FINITE source cells for that cell to be
    # valid. This is a genuine interpolation-SUPPORT FRACTION (how much
    # of the 4-corner weighted average was built from real data), NOT a
    # simple count of valid neighboring pixels -- e.g. a cell built 90%
    # from one very-close finite neighbor and 10% from a far, also-
    # finite neighbor has weight 1.0 even though only 2 of 4 corners
    # were finite; a cell built 50/50 from one finite and one NaN
    # (masked to 0) neighbor has weight 0.5 regardless of distance.
    # UNRESOLVED / FLAGGED FOR APPROVAL: this default has not been
    # confirmed as a project decision -- override via --min-valid-weight.
    min_valid_weight: float = 0.5

    allow_edge_nearest_fill: bool = False  # not implemented -- see README

    regrid_time_batch_size: int = 60

    # ------------------------------------------------------------------
    # 6. MISSING VALUES -- never fabricate.
    # ------------------------------------------------------------------
    fill_enabled: bool = False
    fill_value: float = 0.0

    # ------------------------------------------------------------------
    # 6b. PHYSICAL-VALUE SAFETY CHECK
    # ------------------------------------------------------------------
    implausible_low: float = 0.0
    implausible_high: float = 45.0
    max_implausible_fraction: float = 0.05

    # ------------------------------------------------------------------
    # 6c. UNIT HANDLING -- do not silently relabel or convert.
    # ------------------------------------------------------------------
    # Source units attribute is preserved verbatim. assume_psu_equivalent
    # MUST default to False. Even when a user explicitly sets it True,
    # it only changes DOCUMENTED METADATA (adds an "interpreted_units"
    # note and flips unit_interpretation_status to "user_confirmed") --
    # it never multiplies/divides the numeric SSS values. No conversion
    # is ever applied automatically based on the unusual '.001' string.
    assume_psu_equivalent: bool = False
    # Free-text pointer to the official source-product documentation
    # that would resolve what '.001' means. Left as None/null until a
    # human actually locates and confirms it -- never auto-filled with
    # a guessed or invented URL.
    unit_documentation_reference: Optional[str] = None
    unit_interpretation_notes: str = (
        "Source units '.001' preserved; interpretation requires confirmation "
        "from product documentation."
    )

    # ------------------------------------------------------------------
    # 7. OUTPUT -- year-agnostic templates. '{year}' is substituted with
    #    the detected (or --year-overridden) 4-digit year at runtime.
    # ------------------------------------------------------------------
    output_dir_template: str = "data/processed/sss/{year}"
    output_nc_name_template: str = "sss_{year}_0p25deg.nc"
    output_npy_name_template: str = "sss_{year}_0p25deg.npy"
    output_mask_npy_name_template: str = "sss_{year}_0p25deg_valid_mask.npy"
    output_metadata_name_template: str = "sss_{year}_0p25deg_metadata.json"
    dryrun_report_name_template: str = "sss_{year}_dryrun_report.json"
    diagnostic_plot_name_template: str = "sss_{year}_diagnostic.png"

    # Explicit --output-dir (single-file mode) overrides output_dir_template
    # entirely for that run; left None to use the template.
    explicit_output_dir: Optional[str] = None

    write_npy_outputs: bool = True
    enable_netcdf_compression: bool = True
    netcdf_complevel: int = 4
    use_dask_chunking: bool = True
    atomic_writes: bool = True
    overwrite: bool = False

    # ------------------------------------------------------------------
    # 8. DIAGNOSTIC PLOT
    # ------------------------------------------------------------------
    save_diagnostic_plot: bool = True

    # ------------------------------------------------------------------
    # 9. DRY RUN
    # ------------------------------------------------------------------
    dry_run: bool = False

    # ------------------------------------------------------------------
    # 10. BATCH MODE
    # ------------------------------------------------------------------
    input_dir: Optional[str] = None
    output_root: Optional[str] = None
    input_glob_pattern: str = "**/*.nc"

    # ------------------------------------------------------------------
    # 11. LOGGING
    # ------------------------------------------------------------------
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.lat_min >= self.lat_max:
            raise ValueError(f"lat_min ({self.lat_min}) must be < lat_max ({self.lat_max}).")
        if self.lon_min >= self.lon_max:
            raise ValueError(f"lon_min ({self.lon_min}) must be < lon_max ({self.lon_max}).")
        if self.regrid_method != "weighted_linear":
            raise ValueError(
                f"regrid_method {self.regrid_method!r} is not supported. Only "
                f"'weighted_linear' is implemented for the official SSS path."
            )
        if not (0.0 <= self.min_valid_weight <= 1.0):
            raise ValueError(f"min_valid_weight must be within [0, 1], got {self.min_valid_weight}.")
        if not (0.0 <= self.max_implausible_fraction <= 1.0):
            raise ValueError(
                f"max_implausible_fraction must be within [0, 1], got {self.max_implausible_fraction}."
            )
        if self.implausible_low >= self.implausible_high:
            raise ValueError("implausible_low must be < implausible_high.")
        if self.final_grid_tolerance_deg <= 0:
            raise ValueError("final_grid_tolerance_deg must be > 0.")
        if not (0 <= self.netcdf_complevel <= 9):
            raise ValueError(f"netcdf_complevel must be within [0, 9], got {self.netcdf_complevel}.")
        if self.surface_depth_tolerance_m < 0:
            raise ValueError("surface_depth_tolerance_m must be >= 0.")
        if self.regrid_time_batch_size <= 0:
            raise ValueError(f"regrid_time_batch_size must be > 0, got {self.regrid_time_batch_size}.")

    def resolve(self, relative_path: str) -> Path:
        return (Path(self.project_root) / relative_path).resolve()


CONFIG = SSSConfig()
