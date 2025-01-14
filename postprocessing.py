# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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


def get_time_data(sample_dir, pv_folder="PV Data"):
    data_path = sample_dir / pv_folder / pv_folder
    csv_files = list(data_path.glob("*.csv"))
    if len(csv_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")
        return
    pv_data = np.loadtxt(csv_files[0], delimiter=",", skiprows=1)
    return pv_data[:, 0]


def parse_sample_data(settings_fname, results_folder):
    with open(settings_fname, "r") as file:
        settings = json.load(file)

    sample_name = settings_fname.stem
    sample_dir = Path(settings["path"])
    data_dir = sample_dir / results_folder / "00_Modeling"
    result_path = data_dir / "results_data.csv"

    if not result_path.exists():
        logger.warning(f"Results do not exist for {sample_name}")
        return None, None, None

    sample_data = np.loadtxt(result_path, delimiter=",", skiprows=1)
    time_data = get_time_data(sample_dir)
    # append two additional timing for unloading and loading to ED
    time_data = np.append([0, 0], time_data)
    sample_data[:, 0] = time_data

    return sample_name, settings, sample_data


def normalize_and_interpolate(times_dict, data_dict, N=200):
    """
    Normalize time to range [0, 1] and interpolate data for N points.

    Parameters:
    - times_dict (dict): Dictionary containing times corresponding to the data.
    - data_dict (dict): Dictionary containing volumes or pressures.
    - N (int): Number of equally spaced points for interpolation. Default is 200.

    Returns:
    - interpolated_dict (dict): New dictionary with interpolated values.
    - normalized_times_dict (dict): New dictionary with normalized time values.
    """
    interpolated_dict = {}
    normalized_times_dict = {}

    for group, times_group in times_dict.items():
        interpolated_dict[group] = {}
        normalized_times_dict[group] = {}
        for time_key, times_list in times_group.items():
            if isinstance(times_list, dict):  # Check if there are diameters
                interpolated_dict[group][time_key] = {}
                normalized_times_dict[group][time_key] = {}
                for diameter, times in times_list.items():
                    interpolated_dict[group][time_key][diameter] = []
                    normalized_times_dict[group][time_key][diameter] = []
                    for time_series, data_series in zip(
                        times, data_dict[group][time_key][diameter]
                    ):
                        # Normalize time to [0, 1]
                        normalized_time = (time_series - time_series.min()) / (
                            time_series.max() - time_series.min()
                        )

                        # Interpolate data
                        interpolator = interp1d(
                            normalized_time, data_series, kind="linear"
                        )
                        new_time = np.linspace(0, 1, N)
                        new_values = interpolator(new_time)

                        interpolated_dict[group][time_key][diameter].append(new_values)
                        normalized_times_dict[group][time_key][diameter].append(
                            new_time
                        )
            else:  # Handle case without diameters
                interpolated_dict[group][time_key] = []
                normalized_times_dict[group][time_key] = []
                for time_series, data_series in zip(
                    times_list, data_dict[group][time_key]
                ):
                    # Normalize time to [0, 1]
                    normalized_time = (time_series - time_series.min()) / (
                        time_series.max() - time_series.min()
                    )

                    # Interpolate data
                    interpolator = interp1d(normalized_time, data_series, kind="linear")
                    new_time = np.linspace(0, 1, N)
                    new_values = interpolator(new_time)

                    interpolated_dict[group][time_key].append(new_values)
                    normalized_times_dict[group][time_key].append(new_time)

    return interpolated_dict, normalized_times_dict


def calculate_data_average_and_std(data_dict):
    """
    Calculate the average and standard deviation of all data along the list for each group, time_key, and diameter (if exists).

    Parameters:
    - data_dict (dict): Dictionary containing data to be averaged.

    Returns:
    - averaged_dict (dict): Dictionary with keys as "group_time_key_diameter" and averaged data as values.
    - std_dict (dict): Dictionary with keys as "group_time_key_diameter" and standard deviations as values.
    """
    averaged_dict = {}
    std_dict = {}

    for group, times_group in data_dict.items():
        for time_key, times_list in times_group.items():
            if isinstance(times_list, dict):  # Check if there are diameters
                for diameter, data_list in times_list.items():
                    key = f"{group}_{time_key}_{diameter}"
                    if data_list:
                        averaged_dict[key] = np.mean(data_list, axis=0)
                        std_dict[key] = np.std(data_list, axis=0)
                    else:
                        averaged_dict[key] = None
                        std_dict[key] = None
            else:  # Handle case without diameters
                key = f"{group}_{time_key}"
                if times_list:
                    averaged_dict[key] = np.mean(times_list, axis=0)
                    std_dict[key] = np.std(times_list, axis=0)
                else:
                    averaged_dict[key] = None
                    std_dict[key] = None

    return averaged_dict, std_dict

def plot_data_with_std(averaged_values, normalized_time, std_values=None,  figure=None, color="gray", style='-', label=None):
    """
    Plot data with average and shaded area for ±1 standard deviation.

    Parameters:
    - averaged_values (list): list with averaged data.
    - normalized_times_dict (list): list with normalized time values.
    - std_values (list): list with standard deviations or None if not needed.
    - figure (matplotlib.figure.Figure or None): Existing figure to plot on. If None, a new figure is created.

    Returns:
    - matplotlib.figure.Figure: The figure object for further modification.
    """
    # Create or use the existing figure
    if figure is None:
        figure = plt.figure()
    else:
        plt.figure(figure.number)

    ax = figure.gca()  # Get the current axes
    
    # Plot average line and fill standard deviation area
    ax.plot(normalized_time, averaged_values, label=label, color=color, linestyle=style)
    if std_values is not None:
        ax.fill_between(
            normalized_time,
            averaged_values - std_values,
            averaged_values + std_values,
            color=color,
            alpha=0.3,
            label='STD between samples',
        )

    # Return the figure for further modification
    return figure


def get_colors_styles(dict_keys):
    styles_dict = {}
    colors_dict = {}
    for key in dict_keys:
        if 'SHAM' in key:
            color = '#1f77b4'
        else:
            if '150' in key:
                color = '#ff7f0e'
            elif '130' in key:
                color = '#d62728'
            elif '107' in key:
                color = '#800000'
            else:
                color = 'black'
                logger.warning(f'You need to specify colors for {key}')
        
        if '6' in key:
            style = '-'
        elif '12' in key:
            style = '--'
        elif '20' in key:
            style = '-.'
        else:
            logger.warning(f'You need to specify style for {key}')

        styles_dict.update({key:style})
        colors_dict.update({key:color})
    return colors_dict, styles_dict
        
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
        "-r",
        "--results_folder",
        default="t3",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="/home/shared/00_results_average",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args(args)

    setting_dir = Path(args.settings_dir)
    results_folder = args.results_folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12, 20]
    diameter_list = [107, 130, 150]

    ids = initialize_results_dict(group_list, time_list, diameter_list)
    times = initialize_results_dict(group_list, time_list, diameter_list)
    activations = initialize_results_dict(group_list, time_list, diameter_list)
    volumes = initialize_results_dict(group_list, time_list, diameter_list)
    pressures = initialize_results_dict(group_list, time_list, diameter_list)

    for settings_fname in sorted(setting_dir.iterdir()):
        if not settings_fname.suffix == ".json":
            continue

        sample_name, settings, sample_data = parse_sample_data(
            settings_fname, results_folder
        )
        if sample_data is None:
            continue

        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)

        if diameter is None:
            ids[group][time].append(sample_name)
            times[group][time].append(sample_data[:, 0])
            activations[group][time].append(sample_data[:, 1])
            volumes[group][time].append(sample_data[:, 2])
            pressures[group][time].append(sample_data[:, 4])
        else:
            ids[group][time][diameter].append(sample_name)
            times[group][time][diameter].append(sample_data[:, 0])
            activations[group][time][diameter].append(sample_data[:, 1])
            volumes[group][time][diameter].append(sample_data[:, 2])
            pressures[group][time][diameter].append(sample_data[:, 4])

    interpolated_volumes, normalized_times = normalize_and_interpolate(times, volumes)
    interpolated_pressures, _ = normalize_and_interpolate(times, pressures)
    interpolated_actvations, _ = normalize_and_interpolate(times, activations)

    averaged_actvations, std_actvations = calculate_data_average_and_std(interpolated_actvations)
    averaged_volumes, std_volumes = calculate_data_average_and_std(interpolated_volumes)
    averaged_pressures, std_pressures = calculate_data_average_and_std(interpolated_pressures)
    normalized_times, _ = calculate_data_average_and_std(normalized_times)
    
    fig_all = plt.figure()
    colors_dict, styles_dict = get_colors_styles(averaged_actvations.keys())
    
    for key in averaged_actvations.keys():
        
        averaged_values = averaged_actvations[key]
        std_values = std_actvations[key]
        normalized_time = normalized_times[key]
        
        if averaged_values is None:
            continue
        fig_all = plot_data_with_std(averaged_values, normalized_time, std_values=None, figure=fig_all, color=colors_dict[key], style=styles_dict[key], label=key)
        
        fig = plot_data_with_std(averaged_values, normalized_time, std_values=std_values, color=colors_dict[key], style=styles_dict[key], label='Averaged between Samples')
        ax = fig.gca()
        ax.set_title(key)
        ax.set_xlim(0, 1)
        ax.set_ylim(-10,120)
        ax.set_xlabel("Normalized Time [-]")
        ax.set_ylabel("Cardiac Muscle Tension Generation (Activation) [kPa]")
        fname = output_folder / key
        fig.savefig(fname.as_posix(), dpi = 300)
        
    
    ax = fig_all.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-10,120)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Cardiac Muscle Tension Generation (Activation) [kPa]")
    plt.legend()
    fname = output_folder / "all"
    fig_all.savefig(fname.as_posix(), dpi = 300)

if __name__ == "__main__":
    main()
# %%
