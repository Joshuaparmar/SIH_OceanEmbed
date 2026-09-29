from pathlib import Path
import xarray as xr
import numpy as np

DATA_DIR = Path("data/raw/currents/train/2021")

files = sorted(DATA_DIR.glob("*.nc"))

print("=" * 65)
print("OSCAR 2021 — FULL SPATIAL & TEMPORAL VERIFICATION")
print("=" * 65)

print(f"\nFiles found: {len(files)}")

# ---------------------------------------------------------
# 1. BASIC FILE COUNT
# ---------------------------------------------------------
if len(files) != 365:
    print("❌ ERROR: Expected 365 daily files.")
else:
    print("✅ File count = 365")

# ---------------------------------------------------------
# 2. INSPECT FIRST FILE FOR GRID / VARIABLES
# ---------------------------------------------------------
with xr.open_dataset(files[0]) as ds:

    print("\n--- GRID INFORMATION ---")

    print("Dimensions:")
    print(ds.sizes)

    print("\nCoordinates:")
    print(list(ds.coords))

    print("\nVariables:")
    print(list(ds.data_vars))

    # Find latitude / longitude names
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    lat = ds[lat_name].values
    lon = ds[lon_name].values

    print(f"\nLatitude range : {lat.min()} to {lat.max()}")
    print(f"Longitude range: {lon.min()} to {lon.max()}")

    if len(lat) > 1:
        lat_res = abs(float(lat[1] - lat[0]))
        print(f"Latitude resolution : {lat_res}°")

    if len(lon) > 1:
        lon_res = abs(float(lon[1] - lon[0]))
        print(f"Longitude resolution: {lon_res}°")

    # -----------------------------------------------------
    # 3. U/V CHECK
    # -----------------------------------------------------
    print("\n--- U/V INFORMATION ---")

    u = ds["u"]
    v = ds["v"]

    print("u dimensions:", u.dims)
    print("v dimensions:", v.dims)

    print("u units:", u.attrs.get("units", "Not specified"))
    print("v units:", v.attrs.get("units", "Not specified"))

    print("u min:", float(u.min(skipna=True)))
    print("u max:", float(u.max(skipna=True)))

    print("v min:", float(v.min(skipna=True)))
    print("v max:", float(v.max(skipna=True)))

# ---------------------------------------------------------
# 4. CHECK ALL DATES
# ---------------------------------------------------------
print("\n--- DATE VERIFICATION ---")

dates = []

for file in files:
    with xr.open_dataset(file) as ds:
        if "time" not in ds.coords:
            print(f"❌ Missing time: {file.name}")
            continue

        dates.append(np.datetime64(ds.time.values[0], "D"))

dates = sorted(dates)

expected_dates = np.arange(
    np.datetime64("2021-01-01"),
    np.datetime64("2022-01-01"),
    np.timedelta64(1, "D")
)

print("First date:", dates[0])
print("Last date :", dates[-1])
print("Number of dates:", len(dates))

if len(dates) == 365 and np.array_equal(np.array(dates), expected_dates):
    print("✅ Dates are continuous: 2021-01-01 → 2021-12-31")
else:
    print("❌ Date continuity problem detected!")

# ---------------------------------------------------------
# 5. CHECK ALL FILES FOR CORRUPTION / EMPTY DATA
# ---------------------------------------------------------
print("\n--- DATA QUALITY CHECK ---")

bad_files = []
all_nan_u = []
all_nan_v = []

for i, file in enumerate(files, 1):

    try:
        with xr.open_dataset(file) as ds:

            u = ds["u"]
            v = ds["v"]

            if u.size == 0 or v.size == 0:
                bad_files.append(file.name)

            if bool(u.isnull().all()):
                all_nan_u.append(file.name)

            if bool(v.isnull().all()):
                all_nan_v.append(file.name)

        if i % 50 == 0:
            print(f"Checked {i}/{len(files)} files...")

    except Exception as e:
        bad_files.append(f"{file.name} -> {e}")

print("\nBad/corrupt files:", len(bad_files))
print("All-NaN U files  :", len(all_nan_u))
print("All-NaN V files  :", len(all_nan_v))

# ---------------------------------------------------------
# 6. FINAL RESULT
# ---------------------------------------------------------
print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)

success = True

if len(files) != 365:
    success = False

if len(dates) != 365 or not np.array_equal(np.array(dates), expected_dates):
    success = False

if bad_files or all_nan_u or all_nan_v:
    success = False

if success:
    print("\n🎉 OSCAR 2021 FULL VERIFICATION PASSED!")
    print("✅ 365 daily files")
    print("✅ Continuous dates")
    print("✅ 0.25° grid information checked")
    print("✅ U variable present")
    print("✅ V variable present")
    print("✅ U/V data are not completely empty")
    print("✅ All files readable")
else:
    print("\n❌ VERIFICATION FAILED — PROBLEMS FOUND")

print("=" * 65)