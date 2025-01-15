# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import dolfin
import pulse

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
                    for time_series, data_series in zip(times, data_dict[group][time_key][diameter]):
                        # Normalize time to [0, 1]
                        normalized_time = (time_series - time_series.min()) / (time_series.max() - time_series.min())

                        # Interpolate data
                        interpolator = interp1d(normalized_time, data_series, kind="linear")
                        new_time = np.linspace(0, 1, N)
                        new_values = interpolator(new_time)

                        interpolated_dict[group][time_key][diameter].append(new_values)
                        normalized_times_dict[group][time_key][diameter].append(new_time)
            else:  # Handle case without diameters
                interpolated_dict[group][time_key] = []
                normalized_times_dict[group][time_key] = []
                for time_series, data_series in zip(times_list, data_dict[group][time_key]):
                    # Normalize time to [0, 1]
                    normalized_time = (time_series - time_series.min()) / (time_series.max() - time_series.min())

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


def plot_data_with_std(
    averaged_values, normalized_time, std_values=None, figure=None, color="gray", style="-", label=None
):
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
            label="STD between samples",
        )

    # Return the figure for further modification
    return figure


def plot_and_save(
    key,
    averaged_values,
    normalized_time,
    std_values,
    colors_dict,
    styles_dict,
    output_folder,
    ylim=None,
    ylabel="Y axis",
    fname_prefix=None,
):
    fig = plot_data_with_std(
        averaged_values,
        normalized_time,
        std_values=std_values,
        color=colors_dict[key],
        style=styles_dict[key],
        label="Averaged between Samples",
    )
    ax = fig.gca()
    # ax.set_title(key)
    ax.set_xlim(0, 1)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel(ylabel)
    if fname_prefix is None:
        fname = output_folder / key
    else:
        fname = output_folder / f"{fname_prefix}_{key}"
    fig.savefig(fname.as_posix(), dpi=300)


def get_colors_styles(dict_keys):
    styles_dict = {}
    colors_dict = {}
    for key in dict_keys:
        if "SHAM" in key:
            color = "#1f77b4"
        else:
            if "150" in key:
                color = "#ff7f0e"
            elif "130" in key:
                color = "#d62728"
            elif "107" in key:
                color = "#800000"
            else:
                color = "black"
                logger.warning(f"You need to specify colors for {key}")

        if "6" in key:
            style = "-"
        elif "12" in key:
            style = "--"
        elif "20" in key:
            style = "-."
        else:
            logger.warning(f"You need to specify style for {key}")

        styles_dict.update({key: style})
        colors_dict.update({key: color})
    return colors_dict, styles_dict


# %%
def load_mesh_from_file(mesh_fname: Path):
    # Read the mesh
    mesh_fname = Path(mesh_fname)
    with dolfin.XDMFFile(mesh_fname.as_posix()) as xdmf:
        mesh = dolfin.Mesh()
        xdmf.read(mesh)
    return mesh


def load_F_function_from_file(F_fname: Path, t: float, mesh: dolfin.mesh):
    F_fname = Path(F_fname)
    tensor_element = dolfin.TensorElement("DG", mesh.ufl_cell(), 0)
    function_space = dolfin.FunctionSpace(mesh, tensor_element)
    F = dolfin.Function(function_space)
    with dolfin.XDMFFile(F_fname.as_posix()) as xdmf:
        xdmf.read_checkpoint(F, "Deformation Gradiant", t)
    return F


def compute_fiber_strain(E: dolfin.Function, fib0: dolfin.Function, mesh: dolfin.mesh):
    V = dolfin.FunctionSpace(mesh, "DG", 0)
    Eff = dolfin.project(dolfin.inner(E * fib0, fib0), V)
    return Eff


def compute_fiber_strain_values_from_file(F_fname: Path, mesh: dolfin.mesh, fib0, num_time_step: int = 1000):
    F_fname = Path(F_fname)
    Eff_value = []
    F0 = load_F_function_from_file(F_fname, 1, mesh)
    for t in range(num_time_step):
        try:
            F_function = load_F_function_from_file(F_fname, t, mesh)
            # Here we exclude the initial inflation part for calculation of strain values
            F_new = F_function * dolfin.inv(F0)
            E_function = pulse.kinematics.GreenLagrangeStrain(F_new)
            Eff_t = compute_fiber_strain(E_function, fib0, mesh)
            Eff_value.append(Eff_t.vector()[:])
        except:
            break
    return Eff_value


def compute_spatial_average(value):
    value_ave = []
    for value_t in value:
        value_ave.append(np.average(value_t))
    return np.array(value_ave)


def load_MW_function_from_file(MW_fname: Path, t: float, mesh: dolfin.mesh):
    MW_fname = Path(MW_fname)
    element = dolfin.FiniteElement("DG", mesh.ufl_cell(), 0)
    function_space = dolfin.FunctionSpace(mesh, element)
    MW = dolfin.Function(function_space)
    with dolfin.XDMFFile(MW_fname.as_posix()) as xdmf:
        xdmf.read_checkpoint(MW, "Myocardium Work", t)
    return MW


def compute_MW_values_from_file(MW_fname: Path, mesh: dolfin.mesh, num_time_step: int = 1000):
    MW_fname = Path(MW_fname)
    MW_value = []
    for t in range(num_time_step):
        try:
            MW_function = load_MW_function_from_file(MW_fname, t, mesh)
            # Here we exclude the initial inflation part for calculation of strain values
            MW_value.append(MW_function.vector()[:])
        except:
            break
    return MW_value

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
    fiber_strains = initialize_results_dict(group_list, time_list, diameter_list)
    MW = initialize_results_dict(group_list, time_list, diameter_list)

    for settings_fname in sorted(setting_dir.iterdir()):
        if not settings_fname.suffix == ".json":
            continue

        sample_name, settings, sample_data = parse_sample_data(settings_fname, results_folder)
        if sample_data is None:
            continue

        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)

        sample_dir = Path(settings["path"])
        geo_dir = sample_dir / results_folder / "Geometry"
        unloaded_geometry_fname = geo_dir / "unloaded_geometry_with_fibers.h5"
        geo = pulse.HeartGeometry.from_file(unloaded_geometry_fname.as_posix())
        F_fname = sample_dir / results_folder / "00_Modeling/Deformation_Gradient.xdmf"

        Eff_value = compute_fiber_strain_values_from_file(F_fname, geo.mesh, geo.f0)
        Eff_ave = compute_spatial_average(Eff_value)
        MW_fname = sample_dir / results_folder / "00_Modeling/Myocardial_Work.xdmf"
        MW_value = compute_MW_values_from_file(MW_fname, geo.mesh)
        MW_ave = compute_spatial_average(MW_value)
        
        # The strain is calculated based on ED not the unloaded geometry
        Eff_ave[0] = 0
        MW_ave[0] = 0
        MW_ave[1] = 0

        if diameter is None:
            ids[group][time].append(sample_name)
            times[group][time].append(sample_data[:, 0])
            activations[group][time].append(sample_data[:, 1])
            fiber_strains[group][time].append(Eff_ave)
            MW[group][time].append(MW_ave)
        else:
            ids[group][time][diameter].append(sample_name)
            times[group][time][diameter].append(sample_data[:, 0])
            activations[group][time][diameter].append(sample_data[:, 1])
            fiber_strains[group][time][diameter].append(Eff_ave)
            MW[group][time][diameter].append(MW_ave)
        

    interpolated_actvations, normalized_times = normalize_and_interpolate(times, activations)
    interpolated_fiber_strains, _ = normalize_and_interpolate(times, fiber_strains)
    interpolated_MW, _ = normalize_and_interpolate(times, MW)

    averaged_actvations, std_actvations = calculate_data_average_and_std(interpolated_actvations)
    averaged_fiber_strains, std_fiber_strains = calculate_data_average_and_std(interpolated_fiber_strains)
    averaged_MW, std_MW = calculate_data_average_and_std(interpolated_MW)
    normalized_times, _ = calculate_data_average_and_std(normalized_times)

    fig_activations = plt.figure()
    fig_fiber_strains = plt.figure()
    fig_MW = plt.figure()
    colors_dict, styles_dict = get_colors_styles(averaged_actvations.keys())

    for key, normalized_time in normalized_times.items():
        if averaged_actvations[key] is None:
            continue

        plot_and_save(
            key,
            averaged_actvations[key],
            normalized_time,
            std_actvations[key],
            colors_dict,
            styles_dict,
            output_folder,
            ylim=(-10, 110),
            ylabel="Cardiac Muscle Tension Generation (Activation) [kPa]",
            fname_prefix="activation",
        )
        fig_activations = plot_data_with_std(
            averaged_actvations[key],
            normalized_time,
            std_values=None,
            figure=fig_activations,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )

        plot_and_save(
            key,
            averaged_fiber_strains[key],
            normalized_time,
            std_fiber_strains[key],
            colors_dict,
            styles_dict,
            output_folder,
            ylim=(-0.1, 0),
            ylabel="Averaged Fiber Strains [-]",
            fname_prefix="strain",
        )
        fig_fiber_strains = plot_data_with_std(
            averaged_fiber_strains[key],
            normalized_time,
            std_values=None,
            figure=fig_fiber_strains,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )
        
        plot_and_save(
            key,
            averaged_MW[key],
            normalized_time,
            std_MW[key],
            colors_dict,
            styles_dict,
            output_folder,
            ylim=(-4, 4),
            ylabel="Averaged Myocaridal Work [mJ]",
            fname_prefix="work",
        )
        fig_MW = plot_data_with_std(
            averaged_MW[key],
            normalized_time,
            std_values=None,
            figure=fig_MW,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )

    ax = fig_activations.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-10, 120)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Cardiac Muscle Tension Generation (Activation) [kPa]")
    # plt.legend()
    fname = output_folder / "Activation"
    fig_activations.savefig(fname.as_posix(), dpi=300)

    ax = fig_fiber_strains.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Averaged Fiber Strains [-]")
    # plt.legend()
    fname = output_folder / "Fiber_Strain"
    fig_fiber_strains.savefig(fname.as_posix(), dpi=300)


    ax = fig_MW.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Averaged Myocaridal Work [mJ]")
    # plt.legend()
    fname = output_folder / "Myocardial_Work"
    fig_MW.savefig(fname.as_posix(), dpi=300)
    
    
    fig_activations_group_sham = plt.figure()
    fig_activations_group_107 = plt.figure()
    fig_activations_group_130 = plt.figure()
    fig_activations_group_150 = plt.figure()
    
    fig_strains_group_sham = plt.figure()
    fig_strains_group_107 = plt.figure()
    fig_strains_group_130 = plt.figure()
    fig_strains_group_150 = plt.figure()
    
    fig_mw_group_sham = plt.figure()
    fig_mw_group_107 = plt.figure()
    fig_mw_group_130 = plt.figure()
    fig_mw_group_150 = plt.figure()

    for key, normalized_time in normalized_times.items():
        if averaged_actvations[key] is None:
            continue
        if 'SHAM' in key:
            fig_activations_group_sham = plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_sham = plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            
            fig_mw_group_sham = plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
        if '107' in key:
            fig_activations_group_107 = plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_107 = plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_107 = plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
        if '130' in key:
            fig_activations_group_130 = plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_130 = plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_130 = plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
        if '150' in key:
            fig_activations_group_150 = plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_150 = plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_150 = plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
    
    figures = {
    'fig_activations_group_sham':fig_activations_group_sham,
    'fig_activations_group_107': fig_activations_group_107,
    'fig_activations_group_130': fig_activations_group_130,
    'fig_activations_group_150': fig_activations_group_150,
    'fig_strains_group_sham':fig_strains_group_sham,
    'fig_strains_group_107': fig_strains_group_107,
    'fig_strains_group_130': fig_strains_group_130,
    'fig_strains_group_150': fig_strains_group_150,
    'fig_mw_group_sham':fig_mw_group_sham,
    'fig_mw_group_107': fig_mw_group_107,
    'fig_mw_group_130': fig_mw_group_130,
    'fig_mw_group_150': fig_mw_group_150,
    }

    # Iterate over the figure dictionary
    for fig_name, fig in figures.items():
        ax = fig.gca()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Normalized Time [-]")
        if 'activations' in fig_name:
            ax.set_ylim(-10, 120)
            ax.set_ylabel("Cardiac Muscle Tension Generation (Activation) [kPa]")
        elif 'strain' in fig_name:
            ax.set_ylim(-0.1, 0)
            ax.set_ylabel("Averaged Fiber Strains [-]")
        elif 'mw' in fig_name:
            ax.set_ylim(-4, 4)
            ax.set_ylabel("Averaged Myocaridal Work [mJ]")
            
        # Save the figure
        fname = output_folder / f"{fig_name}.png"  # Save as PNG or desired format
        fig.savefig(fname.as_posix(), dpi=300)

if __name__ == "__main__":
    main()
# %%
