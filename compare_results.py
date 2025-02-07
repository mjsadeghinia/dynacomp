import numpy as np
from pathlib import Path
from structlog import get_logger
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import dolfin
import pulse
import utils_post

import logging
import argparse


logger = get_logger()


# %%# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
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


def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--first",
        default="/home/shared/dynacomp/00_data/CineData/AS/6weeks/130/OP126_1/coarse_mesh_v2",
        type=Path,
        help="The address to the sample 1.",
    )

    parser.add_argument(
        "--second",
        default="/home/shared/dynacomp/00_data/CineData/AS/6weeks/130/OP126_3/coarse_mesh_v2",
        type=Path,
        help="The address to the sample 2.",
    )

    args = parser.parse_args(args)
    dir_1 = args.first
    dir_2 = args.second

    data_1 = load_data(dir_1)
    data_2 = load_data(dir_2)

    normalize_data_1 = normalize_data(data_1)
    normalize_data_2 = normalize_data(data_2)

    strain_1 = load_strain(dir_1)
    strain_2 = load_strain(dir_2)

    strain_1_norm = normalize_data(strain_1)
    strain_2_norm = normalize_data(strain_2)

    fig, ax = plt.subplots()
    ax.plot(normalize_data_1[:, 0], normalize_data_1[:, 1], color="r", label=dir_1.parent.stem)
    ax.plot(normalize_data_2[:, 0], normalize_data_2[:, 1], color="b", label=dir_2.parent.stem)
    ax.set_xlim(0, 1)
    ax.set_ylim(-10, 100)
    ax.set_ylabel("Activation [kPa]")
    ax.set_xlabel("Normalized Time [-]")
    ax.grid()
    ax.legend()

    fname = f"Activation_comparison"
    fig.savefig(fname, dpi=300)
    plt.close(fig)

    # Plot average line and fill standard deviation area
    fig, ax = plt.subplots()
    ax.plot(strain_1_norm[:, 0], strain_1_norm[:, 1], label=dir_1.parent.stem, color="r")
    ax.fill_between(
        strain_1_norm[:, 0],
        strain_1_norm[:, 1] - strain_1_norm[:, 2],
        strain_1_norm[:, 1] + strain_1_norm[:, 2],
        color="r",
        alpha=0.3,
    )

    ax.plot(strain_2_norm[:, 0], strain_2_norm[:, 1], label=dir_2.parent.stem, color="b")
    ax.fill_between(
        strain_2_norm[:, 0],
        strain_2_norm[:, 1] - strain_2_norm[:, 2],
        strain_2_norm[:, 1] + strain_2_norm[:, 2],
        color="b",
        alpha=0.3,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.1)
    ax.set_ylabel("Strain [-]")
    ax.set_xlabel("Normalized Time [-]")
    ax.grid()
    ax.legend()

    fname = f"Strain_comparison"
    fig.savefig(fname, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
