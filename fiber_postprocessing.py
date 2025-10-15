import argparse
import json
from pathlib import Path
import utils_post
import numpy as np
import csv
import scipy.stats
import pulse
import matplotlib.pyplot as plt

import utils

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
        default=1,
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
        default="/home/shared/02_post_processing/02_Fiber_Results",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-f",
        "--folder",
        default="02_Fiber_Modeling",
        type=str,
        help="The folder containing the Fiber results."
    )

    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Version number (e.g., 3)",
    )

    return parser.parse_args(args)

def prepare_results_dict(data_dict, ordered_keys=None, round_flag=True):
    data_dict = utils_post.flatten_data_dict(data_dict)
    if round_flag:
        data_dict = {k: np.round(v,3) for k, v in data_dict.items() if v}
    # Determine ordering of keys
    if ordered_keys is not None:
        ordered = [k for k in ordered_keys if k in data_dict]
    else:
        ordered = list(data_dict.keys())
    # Reorder data_dict according to ordered list
    data_dict = {k: data_dict[k] for k in ordered}
    return data_dict

def plot_angles(ids, epi_angles, endo_angles, output_dir):
    keys = list(ids.keys())
    # Compute means and SEMs (avoid ddof warning for n=1)
    epi_means = []
    endo_means = []
    for k in keys:
        epi_means.append(np.mean(np.array(epi_angles[k])))
        endo_means.append(np.mean(np.array(endo_angles[k])))

    colors_dict, _ = utils_post.get_colors_styles(ids.keys())

    fig, ax = plt.subplots()
    for key in keys:
        ax.plot([0,1], [np.mean(np.array(epi_angles[key])), np.mean(np.array(endo_angles[key]))], label=key, color=colors_dict[key])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Angles (degrees)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Epi', 'Endo'])
    ax.legend()
    fig.tight_layout()
    fname = output_dir / "Fiber_Angles_averages.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    for key in keys:
        for i in range(len(ids[key])):
            ax.plot([0,1], [epi_angles[key][i], endo_angles[key][i]], label=key, color=colors_dict[key])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Angles (degrees)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Epi', 'Endo'])
    fig.tight_layout()
    fname = output_dir / "Fiber_Angles.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    
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
    v = f"_{args.version}" if args.version else ""
    results_dir = f"{args.results_dir}{v}"
    fiber_folder = f"{args.folder}{v}"
    output_dir = args.output_dir
    output_dir = output_dir.parent / (output_dir.name + v)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12]
    diameter_list = [107, 130, 150]

    ids = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    epi_angles = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    endo_angles = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    delta_angles = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    errors = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    errors_std = utils_post.initialize_results_dict(group_list, time_list, diameter_list)


    # Load settings
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])

    if sample_ID is not None:
        sample_num = utils.get_num_from_id(sample_ID, settings_dir)
        sample_nums = [sample_num]
    elif sample_num is None:
        sample_nums = range(1, len(sorted_files) + 1)

    for n in sample_nums:
        settings = utils.load_settings(settings_dir, n)
        sample_name = settings["id"]
        fiber_dir = Path(results_dir) / sample_name / "TPM" / fiber_folder
        if not fiber_dir.exists():
            continue
        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)
        fiber_data = utils.read_fiber_data(fiber_dir)
        if fiber_data is None:
            continue
        fiber_data_sorted = fiber_data[fiber_data[:, -2].argsort()]
        epi_angle = fiber_data_sorted[0][2]
        endo_angle = fiber_data_sorted[0][3]
        error = fiber_data_sorted[0][-2]
        error_std = fiber_data_sorted[0][-1]

        if exclusion_flag and error>error_threshold :
            logger.warning(f"Sample {sample_name} with the error of {error} is ignored")
            continue
        if diameter is None:
            ids[group][time].append(sample_name)
            epi_angles[group][time].append(epi_angle)
            endo_angles[group][time].append(endo_angle)
            delta_angles[group][time].append(endo_angle - epi_angle)
            errors[group][time].append(error)
            errors_std[group][time].append(error_std)
        else:
            ids[group][time][diameter].append(sample_name)
            epi_angles[group][time][diameter].append(epi_angle)
            endo_angles[group][time][diameter].append(endo_angle)
            delta_angles[group][time][diameter].append(endo_angle - epi_angle)
            errors[group][time][diameter].append(error)
            errors_std[group][time][diameter].append(error_std)

    # Save the results
    ordered_keys = ["SHAM_6", "SHAM_12", "SHAM_20", "AS_6_150", "AS_12_150", "AS_6_130", "AS_12_130", "AS_12_107"]
    fname = output_dir / "Epi_Angle.png"
    utils_post.plot_bar_with_data(epi_angles, fname, ylabel="Epi Angle [degrees]", ordered_keys=ordered_keys)
    fname = output_dir / "Endo_Angle.png"
    utils_post.plot_bar_with_data(endo_angles, fname, ylabel="Endo Angle [degrees]", ordered_keys=ordered_keys)
    fname = output_dir / "Delta_Angle.png"
    utils_post.plot_bar_with_data(delta_angles, fname, ylabel="Delta Angle [degrees]", ordered_keys=ordered_keys)
    fname = output_dir / "Angle_Error.png"
    utils_post.plot_bar_with_data(errors, fname, ylabel="Error [mm]", ordered_keys=ordered_keys, ylim=(0, 1))
    fname = output_dir / "Angle_Error_Std.png"
    utils_post.plot_bar_with_data(errors_std, fname, ylabel="Error Std [mm]", ordered_keys=ordered_keys, ylim=(0, 1))

    ids = prepare_results_dict(ids, ordered_keys=ordered_keys, round_flag=False)
    epi_angles = prepare_results_dict(epi_angles, ordered_keys=ordered_keys)
    endo_angles = prepare_results_dict(endo_angles, ordered_keys=ordered_keys)
    errors = prepare_results_dict(errors, ordered_keys=ordered_keys)
    errors_std = prepare_results_dict(errors_std, ordered_keys=ordered_keys)
    plot_angles(ids, epi_angles, endo_angles, output_dir)

    fname = output_dir / "EDPVR_Results.csv"
    with open(fname, 'w', newline='') as csvfile:
        # Define header
        header = ["Group", "ID", "epi_angle [degrees]", "endo_angle [degrees]", "error [-]", "error_std [-]"]
        writer = csv.writer(csvfile)
        writer.writerow(header)
        # Loop over groups
        for group, id_list in ids.items():
            for i, sample_id in enumerate(id_list):
                row = [
                    group,
                    sample_id,
                    epi_angles.get(group, [None])[i] if group in epi_angles and len(epi_angles[group]) > i else None,
                    endo_angles.get(group, [None])[i] if group in endo_angles and len(endo_angles[group]) > i else None,
                    errors.get(group, [None])[i] if group in errors and len(errors[group]) > i else None,
                    errors_std.get(group, [None])[i] if group in errors_std and len(errors_std[group]) > i else None,
                ]
                writer.writerow(row)
             
# %%
if __name__ == "__main__":
    main()