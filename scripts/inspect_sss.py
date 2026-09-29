import sys
import xarray as xr

if len(sys.argv) != 2:
    print("Usage: python inspect_sss.py <path_to_nc_file>")
    sys.exit(1)

file_path = sys.argv[1]

print("\nOpening:", file_path)

ds = xr.open_dataset(file_path)

print("\n========== DATASET ==========")
print(ds)

print("\n========== VARIABLES ==========")
print(list(ds.data_vars))

print("\n========== COORDINATES ==========")
print(list(ds.coords))

print("\n========== DIMENSIONS ==========")
print(ds.sizes)

print("\n========== TIME ==========")

if "time" in ds:
    print("First:", ds.time.values[0])
    print("Last :", ds.time.values[-1])
    print("Number of timesteps:", len(ds.time))

print("\n========== SSS ==========")

if "sos" in ds:
    sss = ds["sos"]

    print(sss)

    print("\nSSS attributes:")
    print(sss.attrs)

    print("\n========== SSS STATISTICS ==========")
    print("Minimum :", float(sss.min(skipna=True)))
    print("Maximum :", float(sss.max(skipna=True)))
    print("Mean    :", float(sss.mean(skipna=True)))

    print("\n========== NaN COUNT ==========")
    print("NaN values:", int(sss.isnull().sum()))

else:
    print("ERROR: 'sos' variable was not found!")

print("\n========== DATASET ATTRIBUTES ==========")
print(ds.attrs)

ds.close()

print("\nInspection complete.")