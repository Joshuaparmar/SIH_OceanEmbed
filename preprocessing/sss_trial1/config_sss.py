"""
config_sss.py
=============
Single source of truth for the SIH26066 / OceanEmbed Sea Surface
Salinity (SSS) preprocessing stage. preprocess_sss.py and
netcdf_utils.py never hard-code a file path, coordinate name, region
bound, or interpolation threshold -- everything dataset-specific lives
here.

RELATIONSHIP TO THE TEMPERATURE PIPELINE (config_temperature.py):
This config follows the SAME dataclass / candidate-list / fail-fast
conventions as the existing GLORYS temperature pipeline, but it is a
SEPARATE, independent config -- SSS preprocessing must never import
from or write into temperature's config/output paths. Two design
choices deliberately DIVERGE from the temperature pipeline; both are
flagged inline below because they affect scientific correctness:

1. regrid_method is NOT the temperature pipeline's naive
   xarray.interp(method="linear") (which lets a single NaN/land
   neighbour in the interpolation stencil silently NaN-out or corrupt
   a target cell with no accounting of how much real support the
   result actually had). SSS is ~50.6% missing (satellite coverage
   gaps), not just land -- see missing-value inspection below -- so
   naive interpolation is scientifically inappropriate here. The only
   supported method is "weighted_linear": a NaN-aware, normalized
   weighted bilinear interpolation with an explicit min_valid_weight
   support threshold (see netcdf_utils.regrid_weighted_linear).
2. date_range_confirmed defaults to True here (unlike temperature's
   False): the brief gave an exact, file-inspection-confirmed 2025-01-01
   .. 2025-12-31 / 365-daily-step range for THIS specific raw file,
   not an assumed calendar-year default guessed in the absence of a
   real file.

STATUS: this is the ACTIVE internal-round SSS pipeline for 2025 data
only. The old training-plan structure (2021-2023 train / 2024
cross-validation / 2025 test) mentioned in earlier project history is
NOT used here and is not searched for by this config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Bump when this config's shape or preprocess_sss.py's behavior changes
# in a way that could affect output -- recorded in every run's metadata
# JSON so an output file is traceable to the code that made it.
PIPELINE_VERSION = "0.1.0"  # 0.1.0 = initial SSS pipeline (2025-only, weighted-linear regrid)


@dataclass
class SSSConfig:
    # ------------------------------------------------------------------
    # 1. INPUT FILE -- ACTIVE: 2025 internal-round SSS dataset
    # ------------------------------------------------------------------
    # Unlike temperature's config (no real file was ever available to
    # confirm a name against), the exact raw SSS path for this internal
    # round WAS given and file-inspected in the brief. It is listed here
    # as a candidate (tried if --input is not given), not silently
    # assumed to always exist -- find_input_file() still checks it on
    # disk and reports clearly if it's missing.
    default_input_candidate: str = "data/raw/sss/test/2025/sss_2025.nc"

    # Explicit override (--input on the CLI): exact path, used as-is,
    # no discovery/fallback.
    explicit_input_path: Optional[str] = None

    # Project root that relative paths above are resolved against.
    project_root: str = "."

    # ------------------------------------------------------------------
    # 2. VARIABLE / COORDINATE NAME CANDIDATES (source file detection)
    # ------------------------------------------------------------------
    # Confirmed variable name for this file is 'sos' (CF standard_name
    # sea_surface_salinity). Other candidates kept only as a fallback
    # discovery mechanism if a different SSS product is ever pointed at
    # via --input -- never assumed to be 'sos' without checking.
    sss_var_candidates: List[str] = field(
        default_factory=lambda: ["sos", "sss", "salinity", "so"]
    )
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
    # 2b. OUTPUT COORDINATE NAMES
    # ------------------------------------------------------------------
    # Standardized regardless of source names, mirroring the temperature
    # pipeline's Step-5-compatibility convention. NOTE (flagged, not
    # assumed): unlike temperature's config, this has NOT been verified
    # against a downstream SSS consumer/loader -- there is no known
    # "oceanembed_step5 SSS config" to check against yet. These are the
    # obvious/conventional names; confirm before treating as final.
    output_time_name: str = "time"
    output_latitude_name: str = "latitude"
    output_longitude_name: str = "longitude"

    # ------------------------------------------------------------------
    # 3. REGION / TIME -- CURRENT ROUND: 2025 ONLY
    # ------------------------------------------------------------------
    # Authoritative OceanEmbed project grid (all physical variables must
    # eventually share this). Do NOT derive from the source file's own
    # native extent -- see build_target_grid().
    lat_min: float = 5.0
    lat_max: float = 30.0
    lon_min: float = 45.0
    lon_max: float = 105.0

    # Confirmed by direct inspection of the real 2025 raw file (unlike
    # temperature's assumed default) -- first/last timestamp and daily
    # cadence were reported directly in the brief.
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    date_range_confirmed: bool = True
    expected_time_step_count: Optional[int] = 365

    # Spatial overlap: the native grid (5.0625-29.9375N, 45.0625-
    # 104.9375E) sits INSIDE the requested region and does not reach
    # its exact edges (see boundary note in Step 5/6 of build_target_grid
    # and regrid_weighted_linear). This is expected and handled
    # explicitly (edge target cells stay NaN via no-extrapolation), not
    # an error -- so partial spatial overlap must be allowed.
    allow_partial_spatial_overlap: bool = True

    # Temporal coverage: STRICT by default -- an incomplete 2025 date
    # range must not be silently processed and labeled complete.
    require_complete_temporal_coverage: bool = True

    # ------------------------------------------------------------------
    # 4. DEPTH -- SSS has a SINGLETON surface depth dimension
    # ------------------------------------------------------------------
    # Confirmed by inspection: depth = [0.0] m, positive=down, one value
    # only. The pipeline must FAIL (not guess) if more than one depth
    # level is ever found in a future input file.
    expected_surface_depth_m: float = 0.0
    surface_depth_tolerance_m: float = 1e-3

    # ------------------------------------------------------------------
    # 5. SPATIAL RESOLUTION / REGRIDDING
    # ------------------------------------------------------------------
    regrid_to_quarter_degree: bool = True
    target_resolution_deg: float = 0.25
    # ONLY "weighted_linear" is implemented/supported for the official
    # path -- see module docstring point 1 above for why the
    # temperature pipeline's naive linear .interp() is NOT reused here.
    regrid_method: str = "weighted_linear"
    # Cell-center convention, both endpoints included:
    # n_points = round((max-min)/spacing) + 1. For 5-30N / 45-105E at
    # 0.25 deg this gives exactly 101 x 241 points -- see
    # netcdf_utils.build_target_grid().
    final_grid_tolerance_deg: float = 1e-6

    # UNRESOLVED / FLAGGED FOR APPROVAL (see README "Unresolved
    # decisions"): the minimum fraction of a target cell's bilinear
    # interpolation weight that must come from FINITE source cells for
    # that target cell to be considered valid (not just "an
    # interpolation call returned a number"). 0.5 = majority of the
    # interpolation support must be real, finite ocean observations.
    # This directly trades off spatial coverage vs. conservativeness
    # and has NOT been confirmed as a project decision -- override via
    # --min-valid-weight.
    min_valid_weight: float = 0.5

    # Native grid boundary vs. target grid boundary (documented, not
    # silently absorbed): native lat 5.0625..29.9375, native lon
    # 45.0625..104.9375; target lat 5.00..30.00, target lon
    # 45.00..105.00. The outermost target row (lat=5.00), outermost
    # target row (lat=30.00), outermost target column (lon=45.00), and
    # outermost target column (lon=105.00) fall OUTSIDE the native
    # grid's convex hull by construction (half the native 0.125 deg
    # spacing). With extrapolation disabled (see regrid_weighted_linear),
    # these are guaranteed all-NaN/invalid after regridding -- not a
    # bug, and not silently patched by nearest-neighbor extension
    # unless allow_edge_nearest_fill is explicitly turned on.
    allow_edge_nearest_fill: bool = False

    # Interpolating the full (time, 200, 480) cube in one call is fine
    # in memory (~280 MB float64) for a single 2025 file, but batching
    # along time is kept as a documented, working control (following
    # the temperature pipeline's precedent) in case of a larger/future
    # multi-year SSS file.
    regrid_time_batch_size: int = 60

    # ------------------------------------------------------------------
    # 6. MISSING VALUES -- never fabricate
    # ------------------------------------------------------------------
    # OFF by default and NOT RECOMMENDED, mirroring temperature's
    # convention: filling NaN with a real number would let downstream
    # code that masks via np.isfinite() silently treat invalid/missing
    # SSS cells as real observations.
    fill_enabled: bool = False
    fill_value: float = 0.0

    # ------------------------------------------------------------------
    # 6b. PHYSICAL-VALUE SAFETY CHECK
    # ------------------------------------------------------------------
    # Soft, generous bounds (source units are UNVERIFIED as PSU -- see
    # unit handling below) meant only to catch an unmasked fill-value
    # sentinel (e.g. -32767) slipping through undetected, not to assert
    # a scientifically tight salinity range. Real open-ocean/regional
    # salinity in this domain (Arabian Sea to Bay of Bengal) is roughly
    # 20-40 in whatever the source's ".001" unit denotes; bounds here
    # are deliberately wider than that.
    implausible_low: float = 0.0
    implausible_high: float = 45.0
    max_implausible_fraction: float = 0.05

    # ------------------------------------------------------------------
    # 6c. UNIT HANDLING -- do not silently relabel
    # ------------------------------------------------------------------
    # Source units attribute is reported as ".001". This is preserved
    # verbatim in output provenance. assume_psu_equivalent=False means
    # the pipeline does NOT rename/relabel the output units attribute to
    # "PSU" or "1e-3" or any other value -- it copies the source string
    # through unchanged and adds a documentation note only. Set True
    # only after the project has confirmed (from source documentation,
    # not guessed) that ".001" is intended as practical-salinity-unit
    # equivalent; even then this flag only adds an explanatory note, it
    # does not change any numeric value.
    assume_psu_equivalent: bool = False

    # ------------------------------------------------------------------
    # 7. OUTPUT
    # ------------------------------------------------------------------
    output_dir: str = "data/processed/sss/2025"
    output_nc_name: str = "sss_2025_0p25deg.nc"
    output_npy_name: str = "sss_2025_0p25deg.npy"
    output_mask_npy_name: str = "sss_2025_0p25deg_valid_mask.npy"
    output_metadata_name: str = "sss_2025_0p25deg_metadata.json"
    dryrun_report_name: str = "sss_2025_dryrun_report.json"
    # Flagged as an open question in the README: no confirmed downstream
    # SSS loader is known yet, so it's not certain a separate .npy pair
    # is actually needed (the NetCDF is self-describing and sufficient
    # on its own). Kept on by default only for parity with the
    # temperature pipeline's output set.
    write_npy_outputs: bool = True
    enable_netcdf_compression: bool = True
    netcdf_complevel: int = 4
    use_dask_chunking: bool = True
    atomic_writes: bool = True
    # Refuse to overwrite an existing official output unless explicitly
    # confirmed -- NOT present in the temperature pipeline (which always
    # overwrites with a warning). Safer default for a scientific output
    # that must never be silently clobbered.
    overwrite: bool = False

    # ------------------------------------------------------------------
    # 8. DIAGNOSTIC PLOT
    # ------------------------------------------------------------------
    save_diagnostic_plot: bool = True
    diagnostic_plot_name: str = "sss_diagnostic.png"

    # ------------------------------------------------------------------
    # 9. DRY RUN
    # ------------------------------------------------------------------
    dry_run: bool = False

    # ------------------------------------------------------------------
    # 10. LOGGING
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
                f"'weighted_linear' is implemented for the official SSS path -- see "
                f"config_sss.py module docstring for why naive linear interpolation "
                f"is scientifically inappropriate for this dataset's ~50% missing rate."
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
        """Resolve a config-relative path against project_root."""
        return (Path(self.project_root) / relative_path).resolve()


# Ready-to-use default instance. Import this directly, or construct your
# own SSSConfig(...) (e.g. from argparse) for a different run.
CONFIG = SSSConfig()
