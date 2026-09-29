from pathlib import Path
import sys

import numpy as np
import xarray as xr


def inspect_coordinate(name, values):
    values = np.asarray(values)

    print(f"\n========== {name.upper()} ==========")
    print(f"Number of points: {len(values)}")
    print(f"First 5 values: {values[:5]}")
    print(f"Last 5 values : {values[-5:]}")
    print(f"Minimum: {values.min()}")
    print(f"Maximum: {values.max()}")

    if len(values) > 1:
        differences = np.diff(values)

        print(f"First 5 spacings: {differences[:5]}")
        print(f"Minimum spacing: {differences.min()}")
        print(f"Maximum spacing: {differences.max()}")
        print(f"Mean spacing: {differences.mean()}")

        is_increasing = np.all(differences > 0)
        is_decreasing = np.all(differences < 0)
        is_regular = np.allclose(
            differences,
            differences[0],
            rtol=1e-5,
            atol=1e-6,
        )

        print(f"Increasing: {is_increasing}")
        print(f"Decreasing: {is_decreasing}")
        print(f"Regular grid: {is_regular}")


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python inspect_sss_grid.py "
            "\"path\\to\\sss_2025.nc\""
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"ERROR: File does not exist:\n{input_path}")
        sys.exit(1)

    print(f"Opening: {input_path}")

    with xr.open_dataset(input_path, decode_cf=True) as ds:
        print("\n========== DATASET SUMMARY ==========")
        print(ds)

        print("\n========== VARIABLES ==========")
        for name, variable in ds.data_vars.items():
            print(
                f"{name}: "
                f"dims={variable.dims}, "
                f"shape={variable.shape}, "
                f"dtype={variable.dtype}"
            )
            print(f"Attributes: {dict(variable.attrs)}")

        print("\n========== COORDINATES ==========")
        for name in ds.coords:
            print(f"{name}: dims={ds[name].dims}, shape={ds[name].shape}")

        inspect_coordinate("latitude", ds["latitude"].values)
        inspect_coordinate("longitude", ds["longitude"].values)

        print("\n========== DEPTH ==========")
        if "depth" in ds.coords:
            depth = ds["depth"].values
            print(f"Depth values: {depth}")
            print(f"Depth attributes: {dict(ds['depth'].attrs)}")
        else:
            print("No depth coordinate found.")

        print("\n========== TIME COVERAGE ==========")
        if "time" in ds.coords:
            time_values = ds["time"].values
            print(f"First timestamp: {time_values[0]}")
            print(f"Last timestamp : {time_values[-1]}")
            print(f"Number of timestamps: {len(time_values)}")

            if len(time_values) > 1:
                time_differences = np.diff(time_values)
                print(f"Unique time differences: {np.unique(time_differences)}")

        print("\n========== REQUIRED REGION COVERAGE ==========")
        lat = ds["latitude"].values
        lon = ds["longitude"].values

        required_lat_min = 5.0
        required_lat_max = 30.0
        required_lon_min = 45.0
        required_lon_max = 105.0

        print(f"Required latitude : {required_lat_min} to {required_lat_max}")
        print(f"Required longitude: {required_lon_min} to {required_lon_max}")

        print(
            "Latitude covers required minimum:",
            lat.min() <= required_lat_min,
        )
        print(
            "Latitude covers required maximum:",
            lat.max() >= required_lat_max,
        )
        print(
            "Longitude covers required minimum:",
            lon.min() <= required_lon_min,
        )
        print(
            "Longitude covers required maximum:",
            lon.max() >= required_lon_max,
        )

        print("\n========== QUALITY-CONTROL / MASK VARIABLES ==========")
        possible_names = [
            "quality_flag",
            "quality_flags",
            "qc_flag",
            "qc_flags",
            "mask",
            "land_mask",
            "sea_ice_fraction",
            "error",
            "uncertainty",
        ]

        found = []

        for name in ds.variables:
            name_lower = name.lower()

            if any(
                candidate in name_lower
                for candidate in possible_names
            ):
                found.append(name)

        if found:
            print("Potential QC/mask-related variables:")
            for name in found:
                print(f"- {name}")
        else:
            print("No obvious QC/mask-related variables found.")

        print("\n========== MISSING-VALUE DISTRIBUTION ==========")

        if "sos" not in ds.data_vars:
            print("ERROR: Variable 'sos' was not found.")
            sys.exit(1)

        sss = ds["sos"]

        finite_mask = np.isfinite(sss.values)

        if sss.ndim == 4:
            # Expected dimensions: time, depth, latitude, longitude
            valid_per_timestep = finite_mask.sum(axis=(1, 2, 3))
            total_per_timestep = finite_mask.shape[1] * finite_mask.shape[2] * finite_mask.shape[3]
        elif sss.ndim == 3:
            valid_per_timestep = finite_mask.sum(axis=(1, 2))
            total_per_timestep = finite_mask.shape[1] * finite_mask.shape[2]
        else:
            valid_per_timestep = None
            total_per_timestep = None

        total_values = finite_mask.size
        valid_values = finite_mask.sum()
        missing_values = total_values - valid_values

        print(f"Total values: {total_values:,}")
        print(f"Finite values: {valid_values:,}")
        print(f"Missing/invalid values: {missing_values:,}")
        print(f"Finite percentage: {100 * valid_values / total_values:.4f}%")
        print(f"Missing percentage: {100 * missing_values / total_values:.4f}%")

        if valid_per_timestep is not None:
            valid_percentages = (
                100.0 * valid_per_timestep / total_per_timestep
            )

            print("\nValid percentage per timestep:")
            print(f"Minimum: {valid_percentages.min():.4f}%")
            print(f"Maximum: {valid_percentages.max():.4f}%")
            print(f"Mean   : {valid_percentages.mean():.4f}%")

            print("\nFirst 10 timestep valid percentages:")
            print(valid_percentages[:10])

            print("\nLast 10 timestep valid percentages:")
            print(valid_percentages[-10:])

        print("\n========== COMPLETE ==========")


if __name__ == "__main__":
    main()