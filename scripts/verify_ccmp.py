import sys
from pathlib import Path
import xarray as xr
import numpy as np
import pandas as pd


BASE_DIR = Path("data/raw/winds")


def verify_year(year):
    year = str(year)

    # Automatically determine train/cv/test
    if year in ["2021", "2022", "2023"]:
        split = "train"
    elif year == "2024":
        split = "cv"
    elif year == "2025":
        split = "test"
    else:
        print(f"❌ Unsupported year: {year}")
        return False

    directory = BASE_DIR / split / year

    print("=" * 65)
    print(f"CCMP {year} — FULL DATASET VERIFICATION")
    print("=" * 65)
    print()
    print(f"Directory: {directory}")

    if not directory.exists():
        print("❌ ERROR: Directory does not exist!")
        return False

    files = sorted(directory.glob("*.nc"))

    # ---------------------------------------------------------
    # FILE COUNT
    # ---------------------------------------------------------
    expected = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365

    print(f"Files found: {len(files)}")
    print(f"Expected files: {expected}")
    print()

    if len(files) == expected:
        print(f"✅ Correct number of files: {expected}")
    else:
        print(f"❌ Expected {expected}, found {len(files)}")

    if not files:
        return False

    # ---------------------------------------------------------
    # INSPECT FIRST FILE
    # ---------------------------------------------------------
    try:
        ds = xr.open_dataset(files[0])

        print()
        print("--- GRID INFORMATION ---")

        print("Dimensions:")
        print(ds.dims)

        print()
        print("Coordinates:")
        print(list(ds.coords))

        print()
        print("Variables:")
        print(list(ds.data_vars))

        lat = ds["latitude"]
        lon = ds["longitude"]

        print()
        print(
            f"Latitude range : "
            f"{float(lat.min()):.3f} to {float(lat.max()):.3f}"
        )

        print(
            f"Longitude range: "
            f"{float(lon.min()):.3f} to {float(lon.max()):.3f}"
        )

        if len(lat) > 1:
            lat_res = float(abs(lat[1] - lat[0]))
            print(f"Latitude resolution : {lat_res:.2f}°")

        if len(lon) > 1:
            lon_res = float(abs(lon[1] - lon[0]))
            print(f"Longitude resolution: {lon_res:.2f}°")

        # -----------------------------------------------------
        # U/V INFORMATION
        # -----------------------------------------------------
        print()
        print("--- WIND INFORMATION ---")

        if "uwnd" not in ds:
            print("❌ Missing uwnd variable")
        else:
            u = ds["uwnd"]

            print("uwnd dimensions:", u.dims)
            print("uwnd units:", u.attrs.get("units", "N/A"))
            print("uwnd min:", float(u.min(skipna=True)))
            print("uwnd max:", float(u.max(skipna=True)))

        if "vwnd" not in ds:
            print("❌ Missing vwnd variable")
        else:
            v = ds["vwnd"]

            print("vwnd dimensions:", v.dims)
            print("vwnd units:", v.attrs.get("units", "N/A"))
            print("vwnd min:", float(v.min(skipna=True)))
            print("vwnd max:", float(v.max(skipna=True)))

        ds.close()

    except Exception as e:
        print(f"❌ Could not open first file: {e}")
        return False

    # ---------------------------------------------------------
    # DATE VERIFICATION
    # ---------------------------------------------------------
    print()
    print("--- DATE VERIFICATION ---")

    dates = []

    for f in files:
        try:
            with xr.open_dataset(f) as ds:
                if "time" not in ds:
                    print(f"❌ Missing time: {f.name}")
                    continue

                date = pd.Timestamp(ds.time.values[0]).normalize()
                dates.append(date)

        except Exception as e:
            print(f"❌ Error reading {f.name}: {e}")

    dates = sorted(dates)

    if dates:
        print("First date:", dates[0].date())
        print("Last date :", dates[-1].date())
        print("Number of dates:", len(dates))

        expected_dates = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31",
            freq="D"
        )

        if dates == list(expected_dates):
            print(
                f"✅ Dates are continuous: "
                f"{year}-01-01 → {year}-12-31"
            )
        else:
            print("❌ Date sequence has missing or duplicate dates")

    # ---------------------------------------------------------
    # FULL DATA QUALITY CHECK
    # ---------------------------------------------------------
    print()
    print("--- DATA QUALITY CHECK ---")

    bad_files = []
    missing_u = []
    missing_v = []
    all_nan_u = []
    all_nan_v = []

    for i, f in enumerate(files, start=1):

        try:
            with xr.open_dataset(f) as ds:

                if "uwnd" not in ds:
                    missing_u.append(f.name)

                else:
                    u = ds["uwnd"]

                    if bool(u.isnull().all()):
                        all_nan_u.append(f.name)

                if "vwnd" not in ds:
                    missing_v.append(f.name)

                else:
                    v = ds["vwnd"]

                    if bool(v.isnull().all()):
                        all_nan_v.append(f.name)

        except Exception:
            bad_files.append(f.name)

        if i % 50 == 0 or i == len(files):
            print(f"Checked {i}/{len(files)} files...")

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------
    print()
    print("--- RESULTS ---")

    print("Bad/corrupt files :", len(bad_files))
    print("Missing U files   :", len(missing_u))
    print("Missing V files   :", len(missing_v))
    print("All-NaN U files   :", len(all_nan_u))
    print("All-NaN V files   :", len(all_nan_v))

    success = (
        len(files) == expected
        and len(bad_files) == 0
        and len(missing_u) == 0
        and len(missing_v) == 0
        and len(all_nan_u) == 0
        and len(all_nan_v) == 0
        and len(dates) == expected
        and dates == list(
            pd.date_range(
                start=f"{year}-01-01",
                end=f"{year}-12-31",
                freq="D"
            )
        )
    )

    print()
    print("=" * 65)

    if success:
        print(f"🎉 CCMP {year} FULL VERIFICATION PASSED!")
        print(f"✅ {expected} daily files")
        print("✅ Continuous dates")
        print("✅ 0.25° grid")
        print("✅ uwnd variable present")
        print("✅ vwnd variable present")
        print("✅ Wind data are not completely empty")
        print("✅ All files readable")
    else:
        print(f"❌ CCMP {year} VERIFICATION FAILED")

    print("=" * 65)

    return success


# =============================================================
# MAIN
# =============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print()
        print("Usage:")
        print("  python scripts/verify_ccmp.py YEAR")
        print()
        print("Examples:")
        print("  python scripts/verify_ccmp.py 2021")
        print("  python scripts/verify_ccmp.py 2022")
        print("  python scripts/verify_ccmp.py 2023")
        print("  python scripts/verify_ccmp.py 2024")
        print("  python scripts/verify_ccmp.py 2025")
        sys.exit(1)

    year = sys.argv[1]

    success = verify_year(year)

    sys.exit(0 if success else 1)