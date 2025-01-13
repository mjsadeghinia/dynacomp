# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
from collections import defaultdict

import json
import argparse


logger = get_logger()


# %%# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
def initialize_results_dict(group_list, time_list, diameter_list):
    results_dict = dict()
    for g in group_list:
        results_dict.setdefault(g, {})
        for t in time_list:
            if g == "SHAM":
                results_dict[g].setdefault(t, [])
            else:
                results_dict[g].setdefault(t, {})
                for d in diameter_list:
                    results_dict[g][t].setdefault(d, [])
    return results_dict


def parse_sample_data(settings_fname, output_folder):
    with open(settings_fname, "r") as file:
        settings = json.load(file)

    sample_name = settings_fname.stem
    data_dir = Path(settings["path"]) / output_folder / "00_Modeling"
    result_path = data_dir / "results_data.csv"

    if not result_path.exists():
        logger.warning(f"Results do not exist for {sample_name}")
        return None, None, None

    sample_data = np.loadtxt(result_path, delimiter=",", skiprows=1)
    return sample_name, settings, sample_data


# %%
def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="t3",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args(args)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12, 20]
    diameter_list = [107, 130, 150]

    ids = initialize_results_dict(group_list, time_list, diameter_list)
    times = initialize_results_dict(group_list, time_list, diameter_list)
    volumes = initialize_results_dict(group_list, time_list, diameter_list)
    pressures = initialize_results_dict(group_list, time_list, diameter_list)

    setting_dir = Path(args.settings_dir)
    output_folder = args.output_folder
    for settings_fname in sorted(setting_dir.iterdir()):
        if not settings_fname.suffix == ".json":
            continue

        sample_name, settings, sample_data = parse_sample_data(
            settings_fname, output_folder
        )
        if sample_data is None:
            continue

        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)

        if diameter is None:
            ids[group][time].append(sample_name)
            times[group][time].append(sample_data[:, 0])
            pressures[group][time].append(sample_data[:, 1])
            volumes[group][time].append(sample_data[:, 2])
        else:
            ids[group][time][diameter].append(sample_name)
            times[group][time][diameter].append(sample_data[:, 0])
            pressures[group][time][diameter].append(sample_data[:, 1])
            volumes[group][time][diameter].append(sample_data[:, 2])


if __name__ == "__main__":
    main()
# %%
