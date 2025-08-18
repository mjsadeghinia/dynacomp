import argparse
import json
from pathlib import Path
import utils_post
import numpy as np
import csv

from structlog import get_logger

logger = get_logger()


# %%
def parse_arguments(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        nargs="+",
        type=int,
        help="The sample number(s), will process all the sample if not indicated",
    )

    parser.add_argument(
        "-i",
        "--ID",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )

    parser.add_argument(
        "-x",
        "--exclusion",
        action="store_true",
        help="The flag for excluding samples with high error",
    )

    parser.add_argument(
        "--error_threshold",
        default=0.5,
        type=float,
        help="The value used for excluding samples with high error",
    )

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/home/shared/02_post_processing/02_EDPVR_Results_v2",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    return parser.parse_args(args)


def load_settings(setting_dir, sample_num):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    settings_fname = sorted_files[sample_num - 1]
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")


def read_edpvr_data(edpvr_folder):
    fname = edpvr_folder / "inflation_results.txt"
    if not fname.exists():
        raise FileNotFoundError(f"EDPVR data file {fname} does not exist.")
    edpvr_data = np.loadtxt(fname, delimiter=',', skiprows=1)
    return edpvr_data

def prepare_results_dict(data_dict, ordered_keys=None):
    data_dict = utils_post.flatten_data_dict(data_dict)
    data_dict = {k: np.round(v,3) for k, v in data_dict.items() if v}
    # Determine ordering of keys
    if ordered_keys is not None:
        ordered = [k for k in ordered_keys if k in data_dict]
    else:
        ordered = list(data_dict.keys())
    # Reorder data_dict according to ordered list
    data_dict = {k: data_dict[k] for k in ordered}
    return data_dict


# %%
def main(args=None) -> int:
    if args is None:
        args = parse_arguments()
    else:
        # updating arguments if called by function
        default_args = parse_arguments()
        default_args = vars(default_args)
        for key, value in vars(args).items():
            if value is not None:
                default_args[key] = value
        args = argparse.Namespace(**default_args)
    # Getting the arguments
    sample_num = args.number
    sample_ID = args.ID
    exclusion_flag = args.exclusion
    error_threshold = args.error_threshold
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12, 20]
    diameter_list = [107, 130, 150]

    ids = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    a_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    af_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    a_af_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    err = utils_post.initialize_results_dict(group_list, time_list, diameter_list)


    # Load settings
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])

    if sample_ID is not None:
        sample_num = get_num_from_id(sample_ID, settings_dir)
        sample_nums = [sample_num]
    elif sample_num is None:
        sample_nums = range(1, len(sorted_files) + 1)

    for n in sample_nums:
        settings = load_settings(settings_dir, n)
        sample_name = settings["id"]
        edpvr_folder = Path(results_dir) / sample_name / "TPM" / "02_EDPVR_Modeling"
        if not edpvr_folder.exists():
            continue
        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)
        edpvr_data = read_edpvr_data(edpvr_folder)
        edpvr_data_sorted = edpvr_data[edpvr_data[:, -1].argsort()]
        a = edpvr_data_sorted[0][0]
        af = edpvr_data_sorted[0][1]
        error = edpvr_data_sorted[0][-1]
        if exclusion_flag and error>error_threshold :
            logger.warning(f"Sample {sample_name} with the error of {error} is ignored")
            continue
        if diameter is None:
            ids[group][time].append(sample_name)
            a_matparam[group][time].append(a)
            af_matparam[group][time].append(af)
            a_af_matparam[group][time].append((np.round(a/af, 3)))
            err[group][time].append(error)
        else:
            ids[group][time][diameter].append(sample_name)
            a_matparam[group][time][diameter].append(a)
            af_matparam[group][time][diameter].append(af)
            a_af_matparam[group][time][diameter].append((np.round(a/af, 3)))
            err[group][time][diameter].append(error)

    # Save the results
    ordered_keys = ["SHAM_6", "SHAM_12", "SHAM_20", "AS_6_150", "AS_12_150", "AS_6_130", "AS_12_130", "AS_12_107"]
    fname = output_dir / "ECM_Stiffness.png"
    utils_post.plot_bar_with_data(a_matparam, fname, ylabel="ECM Stiffness [kPa]", ordered_keys=ordered_keys)
    fname = output_dir / "Myocyte_Stiffness.png"
    utils_post.plot_bar_with_data(af_matparam, fname, ylabel="Myocyte Stiffness [kPa]", ordered_keys=ordered_keys)
    fname = output_dir / "ECM_Myocyte_Stiffness_Ratio.png"
    utils_post.plot_bar_with_data(a_af_matparam, fname, ylabel="ECM/Myocyte Stiffness Ratio", ordered_keys=ordered_keys)
    fname = output_dir / "EDPVR_Error.png"
    utils_post.plot_bar_with_data(err, fname, ylabel="EDPVR Error", ordered_keys=ordered_keys, ylim=(0, 1))

    a_matparam = prepare_results_dict(a_matparam, ordered_keys=ordered_keys)
    af_matparam = prepare_results_dict(af_matparam, ordered_keys=ordered_keys)
    a_af_matparam = prepare_results_dict(a_af_matparam, ordered_keys=ordered_keys)
    err = prepare_results_dict(err, ordered_keys=ordered_keys)
    # Save the results to a csv file

    fname = output_dir / "EDPVR_Results.csv"
    with open(fname, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow([
            "Group",
            "a_matparam [kPa]",
            "af_matparam [kPa]",
            "a_af_matparam",
            "err [kPa]"
        ])

        # rows in the exact ordered_keys sequence
        for key in a_matparam.keys():
            # each dict returns something like a numpy array or list
            row = [ key ]
            for d in (a_matparam, af_matparam, a_af_matparam, err):
                vals = d.get(key, [])
                # join into a string; fallback to empty if missing
                row.append(";".join(str(x) for x in vals))
            writer.writerow(row)
             
# %%
if __name__ == "__main__":
    main()