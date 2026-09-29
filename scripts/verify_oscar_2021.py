from pathlib import Path
import xarray as xr

DATA_DIR = Path("data/raw/currents/train/2021")

files = sorted(DATA_DIR.glob("*.nc"))

print("=" * 60)
print("OSCAR 2021 DATASET VERIFICATION")
print("=" * 60)

print(f"\nFiles found: {len(files)}")

if len(files) != 365:
    print("WARNING: Expected 365 files!")

bad_files = []
wrong_variables = []
wrong_dates = []

for i, file in enumerate(files, 1):
    try:
        with xr.open_dataset(file) as ds:

            # Check variables
            if "u" not in ds.data_vars or "v" not in ds.data_vars:
                wrong_variables.append(file.name)

            # Check time
            if "time" not in ds.coords:
                wrong_dates.append(file.name)

            # Check that data actually exists
            u = ds["u"]
            v = ds["v"]

            if u.size == 0 or v.size == 0:
                bad_files.append(file.name)

        if i % 50 == 0:
            print(f"Checked {i}/{len(files)} files...")

    except Exception as e:
        bad_files.append(f"{file.name} -> {e}")

print("\n" + "=" * 60)
print("VERIFICATION RESULT")
print("=" * 60)

print(f"\nTotal files found      : {len(files)}")
print(f"Expected files        : 365")
print(f"Files with errors     : {len(bad_files)}")
print(f"Missing u/v variables : {len(wrong_variables)}")
print(f"Missing time          : {len(wrong_dates)}")

if bad_files:
    print("\nFILES WITH ERRORS:")
    for f in bad_files:
        print("  ", f)

if wrong_variables:
    print("\nFILES WITH MISSING u/v:")
    for f in wrong_variables:
        print("  ", f)

if wrong_dates:
    print("\nFILES WITH MISSING TIME:")
    for f in wrong_dates:
        print("  ", f)

if (
    len(files) == 365
    and not bad_files
    and not wrong_variables
    and not wrong_dates
):
    print("\n✅ ALL 365 OSCAR FILES PASSED BASIC VERIFICATION!")
    print("✅ u and v variables are present.")
    print("✅ Time coordinate is present.")
    print("✅ Files can be opened successfully.")
else:
    print("\n❌ VERIFICATION FOUND PROBLEMS.")