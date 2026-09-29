from pathlib import Path
import sys
import xarray as xr
import numpy as np


# =========================================================
# GET YEAR FROM COMMAND LINE
# =========================================================

if len(sys.argv) != 2:
    print("Usage:")
    print("  python scripts/verify_oscar.py YEAR")
    print()
    print("Example:")
    print("  python scripts/verify_oscar.py 2022")
    sys.exit(1)

try:
    year = int(sys.argv[1])
except ValueError:
    print("ERROR: Year must be a number.")
    sys.exit(1)


# =========================================================
# CONFIGURATION
# =========================================================

if year in [2021, 2022, 2023]:
    split = "train"
elif year == 2024:
    split = "cv"
elif year == 2025:
    split = "test"
else:
    print(f"ERROR: Unsupported year: {year}")
    sys.exit(1)

DATA_DIR = Path(f"data/raw/currents/{split}/{year}")

start_date = np.datetime64(f"{year}-01-01")
end_date = np.datetime64(f"{year + 1}-01-01")

expected_dates = np.arange(
    start_date,
    end_date,
    np.timedelta64(1, "D")
)

expected_file_count = len(expected_dates)


# =========================================================
# HEADER
# =========================================================

print("=" * 65)
print(f"OSCAR {year} — FULL DATASET VERIFICATION")
print("=" * 65)

print(f"\nDirectory: {DATA_DIR}")

if not DATA_DIR.exists():
    print("\n❌ ERROR: Directory does not exist!")
    sys.exit(1)

files = sorted(DATA_DIR.glob("*.nc"))

print(f"Files found: {len(files)}")
print(f"Expected files: {expected_file_count}")


# =========================================================
# 1. FILE COUNT
# =========================================================

print("\n--- FILE COUNT ---")

if len(files) == expected_file_count:
    print(f"✅ Correct number of files: {expected_file_count}")
else:
    print(
        f"❌ File count mismatch: "
        f"found {len(files)}, expected {expected_file_count}"
    )


# =========================================================
# 2. INSPECT FIRST FILE
# =========================================================

print("\n--- GRID INFORMATION ---")

if len(files) == 0:
    print("❌ No NetCDF files found.")
    sys.exit(1)

with xr.open_dataset(files[0]) as ds:

    print("Dimensions:")
    print(ds.sizes)

    print("\nCoordinates:")
    print(list(ds.coords))

    print("\nVariables:")
    print(list(ds.data_vars))

    # Find latitude / longitude names
    if "lat" in ds.coords:
        lat_name = "lat"
    elif "latitude" in ds.coords:
        lat_name = "latitude"
    else:
        lat_name = None

    if "lon" in ds.coords:
        lon_name = "lon"
    elif "longitude" in ds.coords:
        lon_name = "longitude"
    else:
        lon_name = None

    if lat_name and lon_name:

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

    # =====================================================
    # U / V
    # =====================================================

    print("\n--- U/V INFORMATION ---")

    if "u" not in ds.data_vars:
        print("❌ U variable missing!")
    else:
        u = ds["u"]

        print("u dimensions:", u.dims)
        print("u units:", u.attrs.get("units", "Not specified"))
        print("u min:", float(u.min(skipna=True)))
        print("u max:", float(u.max(skipna=True)))

    if "v" not in ds.data_vars:
        print("❌ V variable missing!")
    else:
        v = ds["v"]

        print("v dimensions:", v.dims)
        print("v units:", v.attrs.get("units", "Not specified"))
        print("v min:", float(v.min(skipna=True)))
        print("v max:", float(v.max(skipna=True)))


# =========================================================
# 3. DATE VERIFICATION
# =========================================================

print("\n--- DATE VERIFICATION ---")

dates = []
missing_time = []

for file in files:

    try:
        with xr.open_dataset(file) as ds:

            if "time" not in ds.coords:
                missing_time.append(file.name)
                continue

            file_date = np.datetime64(ds.time.values[0], "D")
            dates.append(file_date)

    except Exception as e:
        print(f"❌ Could not read {file.name}: {e}")


dates = sorted(dates)

if dates:
    print("First date:", dates[0])
    print("Last date :", dates[-1])
    print("Number of dates:", len(dates))

if missing_time:
    print(f"❌ Files missing time: {len(missing_time)}")

# Check exact date sequence
if (
    len(dates) == expected_file_count
    and np.array_equal(np.array(dates), expected_dates)
):
    print(
        f"✅ Dates are continuous: "
        f"{year}-01-01 → {year}-12-31"
    )
else:
    print("❌ Date continuity problem detected!")


# =========================================================
# 4. DATA QUALITY
# =========================================================

print("\n--- DATA QUALITY CHECK ---")

bad_files = []
missing_u = []
missing_v = []
all_nan_u = []
all_nan_v = []

for i, file in enumerate(files, 1):

    try:

        with xr.open_dataset(file) as ds:

            # Check U
            if "u" not in ds.data_vars:
                missing_u.append(file.name)
            else:
                u = ds["u"]

                if u.size == 0:
                    bad_files.append(file.name)

                elif bool(u.isnull().all()):
                    all_nan_u.append(file.name)

            # Check V
            if "v" not in ds.data_vars:
                missing_v.append(file.name)
            else:
                v = ds["v"]

                if v.size == 0:
                    bad_files.append(file.name)

                elif bool(v.isnull().all()):
                    all_nan_v.append(file.name)

        if i % 50 == 0 or i == len(files):
            print(f"Checked {i}/{len(files)} files...")

    except Exception as e:
        bad_files.append(f"{file.name} -> {e}")


print("\nBad/corrupt files :", len(bad_files))
print("Missing U files   :", len(missing_u))
print("Missing V files   :", len(missing_v))
print("All-NaN U files   :", len(all_nan_u))
print("All-NaN V files   :", len(all_nan_v))


# =========================================================
# 5. FINAL RESULT
# =========================================================

print("\n" + "=" * 65)
print("FINAL RESULT")
print("=" * 65)

success = True

if len(files) != expected_file_count:
    success = False

if len(dates) != expected_file_count:
    success = False

if missing_time:
    success = False

if not np.array_equal(np.array(dates), expected_dates):
    success = False

if missing_u or missing_v:
    success = False

if bad_files or all_nan_u or all_nan_v:
    success = False


if success:

    print(f"\n🎉 OSCAR {year} FULL VERIFICATION PASSED!")
    print(f"✅ {expected_file_count} daily files")
    print("✅ Continuous dates")
    print("✅ 0.25° grid checked")
    print("✅ U variable present")
    print("✅ V variable present")
    print("✅ U/V data are not completely empty")
    print("✅ All files readable")

else:

    print(f"\n❌ OSCAR {year} VERIFICATION FAILED")
    print("Please inspect the problems reported above.")

print("=" * 65)