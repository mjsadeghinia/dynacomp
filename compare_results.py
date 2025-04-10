import numpy as np
from pathlib import Path
from structlog import get_logger
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import dolfin
import pulse
import utils_post
import json

import argparse


logger = get_logger()


# %%# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


def get_sample_name(sample_num, setting_dir):
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    sample_name = sorted_files[sample_num - 1].with_suffix("").name
    return sample_name


def get_time_data(data_dir, pv_folder="PV Data"):
    data_path = data_dir.parent / pv_folder / pv_folder
    csv_files = list(data_path.glob("*.csv"))
    csv_file = csv_files[0]
    if len(csv_files) != 1:
        n = [n for n, address in enumerate(csv_files) if "RAW" not in address.name]
        csv_file = csv_files[n[0]]
        logger.warning(f"There are {len(csv_files)} .csv files, we are using {csv_file.name}.")

    pv_data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
    time_data = pv_data[:, 0]
    # append two additional timing for unloading and loading to ED
    time_data = np.append([0, 0], time_data)
    return time_data


def load_data(dir):
    data_dir = Path(dir)
    result_path = data_dir / "00_Modeling" / "results_data.csv"
    if not result_path.exists():
        logger.warning(f"Results do not exist for {dir}")
        return None
    sample_data = np.loadtxt(result_path, delimiter=",", skiprows=1)
    time_data = get_time_data(data_dir)
    sample_data[:, 0] = time_data
    return sample_data


def normalize_data(data, N=100):
    normalize_data = np.zeros((N, data.shape[1]))
    # Normalize time to [0, 1]
    time_series = data[:, 0]
    normalized_time = (time_series - time_series.min()) / (time_series.max() - time_series.min())
    # Interpolate data
    for k in range(data.shape[1] - 1):
        interpolator = interp1d(normalized_time, data[:, k + 1], kind="slinear")
        new_time = np.linspace(0, 1, N)
        new_values = interpolator(new_time)
        normalize_data[:, k + 1] = new_values
    normalize_data[:, 0] = np.linspace(0, 1, N)

    return normalize_data


def load_strain(dir):
    F_fname = dir / "00_Modeling/Deformation_Gradient.xdmf"

    geo_dir = dir / "Geometry"
    unloaded_geometry_fname = geo_dir / "unloaded_geometry_with_fibers.h5"
    geo = pulse.HeartGeometry.from_file(unloaded_geometry_fname.as_posix())

    Eff_value = utils_post.compute_fiber_strain_values_from_file(F_fname, geo.mesh, geo.f0)
    Eff_ave = utils_post.compute_spatial_average(Eff_value)
    Eff_std = utils_post.compute_spatial_std(Eff_value)

    time_data = get_time_data(dir)
    return np.vstack((time_data, Eff_ave, Eff_std)).T


def calulate_rmse(data):
    reference = data[0, :]
    rmse_per_expe = np.sqrt(np.mean((data - reference) ** 2, axis=1))
    return rmse_per_expe


def plot_activations(results_dict):
    fig, ax = plt.subplots()
    all_data = []
    exp_keys = list(results_dict.keys())
    for exp in exp_keys:
        data_normalized = results_dict[exp]["data_normalized"]
        sample_name = results_dict[exp]["directory"].parent.stem
        ax.plot(data_normalized[:, 0], data_normalized[:, 1], label=sample_name + "_" + exp)
        all_data.append(data_normalized[:, 1])
    all_data = np.array(all_data)
    rmse_values = calulate_rmse(all_data)
    # Format RMSE values for title
    rmse_text = ", ".join([f"{exp_keys[i+1]}: {rmse:.2f}kPa" for i, rmse in enumerate(rmse_values[1:])])
    ax.set_title(f"RMSE : {rmse_text}")

    ax.set_xlim(0, 1)
    ax.set_ylim(-10, 110)
    ax.set_ylabel("Activation [kPa]")
    ax.set_xlabel("Normalized Time [-]")
    ax.grid()
    ax.legend()
    return fig


def plot_strains(results_dict):
    # Plot average line and fill standard deviation area
    fig, ax = plt.subplots()
    all_data = []
    exp_keys = list(results_dict.keys())
    for exp in exp_keys:
        data_normalized = results_dict[exp]["data_normalized"]
        strain_normalized = results_dict[exp]["strain_normalized"]
        sample_name = results_dict[exp]["directory"].parent.stem
        ax.plot(strain_normalized[:, 0], strain_normalized[:, 1], label=sample_name + "_" + exp)
        ax.fill_between(
            strain_normalized[:, 0],
            strain_normalized[:, 1] - strain_normalized[:, 2],
            strain_normalized[:, 1] + strain_normalized[:, 2],
            alpha=0.3,
        )
        all_data.append(strain_normalized[:, 1])
    all_data = np.array(all_data)
    rmse_values = calulate_rmse(all_data)
    # Format RMSE values for title
    rmse_text = ", ".join([f"{exp_keys[i+1]}: {rmse:.2f}" for i, rmse in enumerate(rmse_values[1:])])
    ax.set_title(f"RMSE per Experiment (w.r.t. First): {rmse_text}")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.1)
    ax.set_ylabel("Strain [-]")
    ax.set_xlabel("Normalized Time [-]")
    ax.grid()
    ax.legend()

    return fig


def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiments",
        default="coarse_mesh_v2",
        nargs="+",
        type=str,
        help="The result folder to be compared with each other, it should be more than two.",
    )

    parser.add_argument(
        "-n",
        "--sample_numbers",
        default=[1],
        nargs="+",
        type=int,
        help="The sample numbers from which we compare the different results.",
    )

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="00_results_mesh_convergence",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args(args)
    experiments = args.experiments
    sample_numbers = args.sample_numbers
    settings_dir = args.settings_dir
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    for n in sample_numbers:
        sample_name = get_sample_name(n, settings_dir)
        settings = load_settings(settings_dir, sample_name)
        data_dir = Path(settings["path"])

        results_dict = {}
        for exp in experiments:
            results_dict.setdefault(exp, {})
            dir = data_dir / exp
            data = load_data(dir)
            data_normalized = normalize_data(data)
            strain = load_strain(dir)
            strain_normalized = normalize_data(strain)

            results_dict[exp].update({"directory": dir})
            results_dict[exp].update({"data": data})
            results_dict[exp].update({"data_normalized": data_normalized})
            results_dict[exp].update({"strain": strain})
            results_dict[exp].update({"strain_normalized": strain_normalized})

        fig = plot_activations(results_dict)
        fname = output_folder / f"Activation_comparison_{sample_name}"
        fig.savefig(fname.as_posix(), dpi=300)
        plt.close(fig)

        fig = plot_strains(results_dict)
        fname = output_folder / f"Strain_comparison_{sample_name}"
        fig.savefig(fname.as_posix(), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
