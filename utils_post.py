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
    csv_file = csv_files[0]
    if len(csv_files) != 1:
        n = [n for n, address in enumerate(csv_files) if "RAW" not in address.name]
        csv_file = csv_files[n[0]]
        logger.warning(f"There are {len(csv_files)} .csv files, we are using {csv_file.name}.")

    pv_data = np.loadtxt(csv_file, delimiter=",", skiprows=1)
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
    ax.grid()

    # Return the figure for further modification
    return figure


def plot_and_save(
    key,
    data,
    time,
    std,
    colors_dict,
    styles_dict,
    output_folder,
    ylim=None,
    ylabel="Y axis",
    fname_prefix=None,
):
    fig = plot_data_with_std(
        data,
        time,
        std_values=std,
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
    plt.close(fig)


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

def plot_bar_with_error(
    avg_dict,
    std_dict,
    output_path,
    ylim=None,
    ylabel="Tissue Volume [mm³]",
    ordered_keys=None
):
    
    if ordered_keys is not None:
        relevant_keys = [k for k in ordered_keys if k in avg_dict]
    else:
        # Default: use the keys from avg_dict
        relevant_keys = list(avg_dict.keys())
    
    data_keys = []
    data_means = []
    data_stds = []
    for key in relevant_keys:
        avg_val = avg_dict[key]
        std_val = std_dict[key]
        if avg_val is None:
            continue  # Skip entries with no data
        data_keys.append(key)
        data_means.append(avg_val)
        data_stds.append(std_val if std_val is not None else 0.0)

    colors_dict, styles_dict = get_colors_styles(data_keys)

    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(data_keys))

    for i, key in enumerate(data_keys):
        color = colors_dict.get(key, "gray")  # Fallback color
        style = styles_dict.get(key, "-")     # Fallback style (if used for hatching, etc.)
        ax.bar(
            x=i,
            height=data_means[i],
            yerr=data_stds[i],
            color=color,
            edgecolor=color,
            capsize=5,     # Error bar cap size
            alpha=0.9,
            label=key,     # Each bar labeled by its key
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(data_keys, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    fig.savefig(str(output_path), dpi=300)    

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

def export_results(output_folder, plot_vars, time):
    colors_dict, styles_dict = get_colors_styles(time.keys())
   # Create one figure per variable for the "all-in-one" plots
    figs_all = {}
    for var_name in plot_vars.keys():
        figs_all[var_name] = plt.figure()

    # Plot "all-in-one" figures (i.e., each variable with multiple curves)
    for key, normalized_time in time.items():
        for var_name, info in plot_vars.items():
            avg_data = info["avg"][key]
            std_data = info["std"][key]
            if avg_data is None:
                continue
            
            plot_and_save(
                key=key,
                data=avg_data,
                time=normalized_time,
                std=std_data,
                colors_dict=colors_dict,
                styles_dict=styles_dict,
                output_folder=output_folder,
                ylim=info["ylim"],
                ylabel=info["ylabel"],
                fname_prefix=info["fname_prefix"],
            )
            
            figs_all[var_name] = plot_data_with_std(
                avg_data,
                normalized_time,
                std_values=None,  # or std_data if you want the shaded region
                figure=figs_all[var_name],
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )

    for var_name, fig in figs_all.items():
        ax = fig.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(plot_vars[var_name]["ylim"])
        ax.set_xlabel("Normalized Time [-]")
        ax.set_ylabel(plot_vars[var_name]["ylabel"])
        ax.grid()
        
        fname = output_folder / (var_name + "_all_in_one.png")
        fig.savefig(fname.as_posix(), dpi=300)
        plt.close(fig)

def export_group_results(output_folder, plot_vars, group_names, time):
    colors_dict, styles_dict = get_colors_styles(time.keys())

    figs_by_group = {
        group: {var_name: plt.figure() for var_name in plot_vars.keys()}
        for group in group_names
    }
    for key, normalized_time in time.items():
        for var_name, info in plot_vars.items():
            avg_data = info["avg"][key]
            std_data = info["std"][key]
            if avg_data is None:
                continue
            
            # Figure out which group this key belongs to
            for group in group_names:
                if group in key: 
                    figs_by_group[group][var_name] = plot_data_with_std(
                        avg_data,
                        normalized_time,
                        std_values=std_data, 
                        figure=figs_by_group[group][var_name],
                        color=colors_dict[key],
                        style=styles_dict[key],
                        label=key,
                    )
                    break  

    # Format and save each group-specific figure
    for group, var_dict in figs_by_group.items():
        for var_name, fig in var_dict.items():
            ax = fig.gca()
            ax.set_xlim(0, 1)
            ax.set_ylim(plot_vars[var_name]["ylim"])
            ax.set_xlabel("Normalized Time [-]")
            ax.set_ylabel(plot_vars[var_name]["ylabel"])
            ax.grid()

            outname = f"{var_name}_group_{group}.png"
            fname = output_folder / outname
            fig.savefig(fname.as_posix(), dpi=300)
            plt.close(fig)


def get_maximums(results_dict):
    maximums = {}
    for group, times_group in results_dict.items():
        for time_key, times_list in times_group.items():
            if isinstance(times_list, dict):  # Check if there are diameters
                for diameter, data_list in times_list.items():
                    key = f"{group}_{time_key}_{diameter}"
                    if data_list:
                        maximums.update({key : [np.max(list) for list in results_dict[group][time_key][diameter]]})
            else:  # Handle case without diameters
                key = f"{group}_{time_key}"
                if times_list:
                    maximums.update({key : [np.max(list) for list in results_dict[group][time_key]]})
    return maximums            