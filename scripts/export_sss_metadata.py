from pathlib import Path
import json
import sys

import numpy as np
import xarray as xr


def convert_value(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (np.datetime64,)):
        return str(value)

    if isinstance(value, dict):
        return {str(k): convert_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [convert_value(v) for v in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def main():
    if len(sys.argv) != 3:
        print(
            "Usage:\n"
            "python export_sss_metadata.py "
            "\"input.nc\" \"output.json\""
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    with xr.open_dataset(input_path, decode_cf=True) as ds:
        report = {
            "dataset": {
                "dimensions": {
                    name: int(size)
                    for name, size in ds.sizes.items()
                },
                "coordinates": list(ds.coords),
                "data_variables": list(ds.data_vars),
                "global_attributes": {
                    str(k): convert_value(v)
                    for k, v in ds.attrs.items()
                },
            },
            "coordinates": {},
            "variables": {},
        }

        for name in ds.coords:
            values = ds[name].values

            coordinate_info = {
                "dimensions": list(ds[name].dims),
                "shape": list(ds[name].shape),
                "dtype": str(ds[name].dtype),
                "attributes": {
                    str(k): convert_value(v)
                    for k, v in ds[name].attrs.items()
                },
            }

            if values.size <= 1000:
                coordinate_info["values"] = [
                    convert_value(v) for v in values.tolist()
                ]
            else:
                coordinate_info["first_values"] = [
                    convert_value(v) for v in values[:5].tolist()
                ]
                coordinate_info["last_values"] = [
                    convert_value(v) for v in values[-5:].tolist()
                ]

                if values.size > 1:
                    differences = np.diff(values)
                    coordinate_info["minimum_spacing"] = convert_value(
                        differences.min()
                    )
                    coordinate_info["maximum_spacing"] = convert_value(
                        differences.max()
                    )

            report["coordinates"][name] = coordinate_info

        for name in ds.data_vars:
            variable = ds[name]

            variable_info = {
                "dimensions": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "attributes": {
                    str(k): convert_value(v)
                    for k, v in variable.attrs.items()
                },
            }

            if name == "sos":
                values = variable.values
                finite = np.isfinite(values)

                variable_info["statistics"] = {
                    "total_values": int(values.size),
                    "finite_values": int(finite.sum()),
                    "missing_values": int((~finite).sum()),
                    "finite_percentage": float(
                        100 * finite.sum() / values.size
                    ),
                    "missing_percentage": float(
                        100 * (~finite).sum() / values.size
                    ),
                    "minimum_finite": float(np.nanmin(values)),
                    "maximum_finite": float(np.nanmax(values)),
                    "mean_finite": float(np.nanmean(values)),
                }

            report["variables"][name] = variable_info

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Metadata report written to: {output_path}")


if __name__ == "__main__":
    main()