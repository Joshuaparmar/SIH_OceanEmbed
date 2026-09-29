import xarray as xr
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "data/raw/temperature/thetao_2025_test_1day.nc"
)

OUTPUT_FILE = Path(
    "data/raw/temperature/temperature_2025_test_1day_processed.nc"
)

# ============================================================
# IMPORTANT:
# These are the 15 STANDARD DEPTH LEVELS specified by the SIH
# problem statement.
#
# 0 m  -> surface 0-5 m proxy
# 5 m  -> native GLORYS level closest to 5 m
# 10 m onward -> vertical interpolation
# ============================================================

TARGET_DEPTHS = np.array(
    [0, 5, 10, 20, 30, 50, 75, 100, 125, 150,
     200, 300, 500, 700, 1000],
    dtype=np.float32
)

# Required spatial domain
LAT_MIN = 5.0
LAT_MAX = 30.0

LON_MIN = 45.0
LON_MAX = 105.0

RESOLUTION = 0.25


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING RAW TEMPERATURE DATA")
print("=" * 60)

ds = xr.open_dataset(INPUT_FILE)

print(ds)

thetao = ds["thetao"]

print("\nNative depths:")
print(ds.depth.values)

print("\nNative latitude:")
print(float(ds.latitude.min()), "to", float(ds.latitude.max()))

print("\nNative longitude:")
print(float(ds.longitude.min()), "to", float(ds.longitude.max()))


# ============================================================
# SPATIAL SUBSETTING
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL SUBSETTING")
print("=" * 60)

thetao = thetao.sel(
    latitude=slice(LAT_MIN, LAT_MAX),
    longitude=slice(LON_MIN, LON_MAX)
)

print("Subset shape:", thetao.shape)


# ============================================================
# CREATE REQUIRED 0.25 DEGREE GRID
# ============================================================

print("\n" + "=" * 60)
print("CREATING 0.25 DEGREE GRID")
print("=" * 60)

target_lat = np.arange(
    LAT_MIN,
    LAT_MAX + RESOLUTION / 2,
    RESOLUTION,
    dtype=np.float32
)

target_lon = np.arange(
    LON_MIN,
    LON_MAX + RESOLUTION / 2,
    RESOLUTION,
    dtype=np.float32
)

print("Target latitude points:", len(target_lat))
print("Target longitude points:", len(target_lon))

print("Latitude:", target_lat[0], "to", target_lat[-1])
print("Longitude:", target_lon[0], "to", target_lon[-1])


# ============================================================
# SPATIAL INTERPOLATION
# ============================================================

print("\n" + "=" * 60)
print("SPATIAL INTERPOLATION")
print("=" * 60)

thetao = thetao.interp(
    latitude=target_lat,
    longitude=target_lon,
    method="linear"
)

print("After spatial interpolation:", thetao.shape)


# ============================================================
# SURFACE 0–5 m PROXY
# ============================================================

print("\n" + "=" * 60)
print("CREATING SURFACE 0–5 m PROXY")
print("=" * 60)

# Native GLORYS level closest to 5 m
surface_native_depth = float(
    ds.depth.sel(depth=5.0, method="nearest").values
)

print(
    "Native depth selected for surface proxy:",
    surface_native_depth,
    "m"
)

surface_temp = thetao.sel(
    depth=surface_native_depth
)

surface_temp = surface_temp.rename(
    "surface_temp_0_5m"
)

print("Surface proxy shape:", surface_temp.shape)


# ============================================================
# VERTICAL INTERPOLATION
# ============================================================

print("\n" + "=" * 60)
print("VERTICAL INTERPOLATION")
print("=" * 60)

# We interpolate only the depths BELOW 5 m.
#
# 0 m and 5 m are handled separately below.

interpolation_depths = np.array(
    [10, 20, 30, 50, 75, 100, 125, 150,
     200, 300, 500, 700, 1000],
    dtype=np.float32
)

print("Interpolating target depths:")
print(interpolation_depths)

temperature_target_deep = thetao.interp(
    depth=interpolation_depths,
    method="linear"
)

temperature_target_deep = temperature_target_deep.rename(
    "temperature_target_deep"
)

print(
    "Interpolated target shape:",
    temperature_target_deep.shape
)


# ============================================================
# CONSTRUCT THE 15-LEVEL TARGET
# ============================================================

print("\n" + "=" * 60)
print("CONSTRUCTING 15-LEVEL TARGET")
print("=" * 60)

# ------------------------------------------------------------
# 0 m
# ------------------------------------------------------------
#
# The raw GLORYS 0.494 m layer has severe missing-value issues
# in our dataset. Therefore the 0 m standard level is represented
# using the same surface 0–5 m proxy used by the input.
# ------------------------------------------------------------

target_0m = surface_temp.expand_dims(
    depth=[0.0]
).rename("temperature_target")


target_5m = surface_temp.expand_dims(
    depth=[5.0]
).rename("temperature_target")


# ------------------------------------------------------------
# 10–1000 m
# ------------------------------------------------------------

target_deep = temperature_target_deep.rename(
    "temperature_target"
)


# ============================================================
# COMBINE ALL 15 DEPTH LEVELS
# ============================================================

temperature_target = xr.concat(
    [
        target_0m,
        target_5m,
        target_deep
    ],
    dim="depth"
)

temperature_target = temperature_target.assign_coords(
    depth=TARGET_DEPTHS
)

temperature_target = temperature_target.astype(
    "float32"
)

print("\nFINAL TARGET SHAPE:")
print(temperature_target.shape)

print("\nFINAL TARGET DEPTHS:")
print(temperature_target.depth.values)


# ============================================================
# BUILD FINAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("BUILDING FINAL TARGET DATASET")
print("=" * 60)

output_ds = xr.Dataset(
    {
        "surface_temp_0_5m":
            surface_temp.astype("float32"),

        "temperature_target":
            temperature_target,
    }
)


# ============================================================
# METADATA
# ============================================================

output_ds["surface_temp_0_5m"].attrs = {
    "description":
        "Surface 0-5 m temperature proxy",
    "source_depth":
        f"{surface_native_depth:.6f} m",
    "source":
        "GLORYS12V1",
}


output_ds["temperature_target"].attrs = {
    "description":
        "Subsurface ocean temperature at 15 SIH standard depth levels",
    "source":
        "GLORYS12V1",
    "standard_depths_m":
        str(TARGET_DEPTHS.tolist()),
    "zero_meter_method":
        "Surface 0-5 m proxy",
    "five_meter_method":
        f"Native GLORYS level at approximately {surface_native_depth:.6f} m",
}


output_ds.attrs = {
    "project":
        "OceanEmbed",

    "purpose":
        "Target temperature data for deep learning",

    "spatial_resolution":
        "0.25 degree",

    "domain":
        "5N-30N, 45E-105E",

    "temporal_resolution":
        "daily",

    "target_depth_count":
        "15",

    "target_depths_m":
        str(TARGET_DEPTHS.tolist()),
}


# ============================================================
# SAVE
# ============================================================

output_ds.to_netcdf(
    OUTPUT_FILE,
    format="NETCDF4"
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\nCreated:")
print(OUTPUT_FILE)

print("\nFINAL DATASET:")
print(output_ds)

print("\nFINAL DEPTHS:")
print(
    output_ds.temperature_target.depth.values
)

print("\nFINAL LATITUDE:")
print(
    output_ds.latitude.values[[0, -1]]
)

print("\nFINAL LONGITUDE:")
print(
    output_ds.longitude.values[[0, -1]]
)

print("\nFINAL TARGET SHAPE:")
print(
    output_ds.temperature_target.shape
)

print("\nDONE.")