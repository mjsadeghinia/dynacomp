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
        "--skip_samples",
        default="170_1",
        nargs="+",
        type=str,
        help="The list of samples to be skipped",
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
    skip_samples = args.skip_samples
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12, 20]
    diameter_list = [107, 130, 150]

    ids = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    times = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    activations = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    pressures = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    fiber_strains = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    MW = utils_post.initialize_results_dict(group_list, time_list, diameter_list)

    for settings_fname in sorted(setting_dir.iterdir()):
        if not settings_fname.suffix == ".json":
            continue

        sample_name, settings, sample_data = utils_post.parse_sample_data(settings_fname, results_folder)
        if sample_data is None:
            continue
        
        if sample_name in skip_samples:
            continue

        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)

        sample_dir = Path(settings["path"])
        geo_dir = sample_dir / results_folder / "Geometry"
        unloaded_geometry_fname = geo_dir / "unloaded_geometry_with_fibers.h5"
        geo = pulse.HeartGeometry.from_file(unloaded_geometry_fname.as_posix())
        F_fname = sample_dir / results_folder / "00_Modeling/Deformation_Gradient.xdmf"

        Eff_value = utils_post.compute_fiber_strain_values_from_file(F_fname, geo.mesh, geo.f0)
        Eff_ave = utils_post.compute_spatial_average(Eff_value)
        MW_fname = sample_dir / results_folder / "00_Modeling/Myocardial_Work.xdmf"
        MW_value = utils_post.compute_MW_values_from_file(MW_fname, geo.mesh)
        MW_ave = utils_post.compute_spatial_average(MW_value)
        
        # The strain is calculated based on ED not the unloaded geometry
        Eff_ave[0] = 0
        MW_ave[0] = 0
        MW_ave[1] = 0

        if diameter is None:
            ids[group][time].append(sample_name)
            times[group][time].append(sample_data[:, 0])
            activations[group][time].append(sample_data[:, 1])
            pressures[group][time].append(sample_data[:, 4])
            fiber_strains[group][time].append(Eff_ave)
            MW[group][time].append(MW_ave)
        else:
            ids[group][time][diameter].append(sample_name)
            times[group][time][diameter].append(sample_data[:, 0])
            activations[group][time][diameter].append(sample_data[:, 1])
            pressures[group][time][diameter].append(sample_data[:, 4])
            fiber_strains[group][time][diameter].append(Eff_ave)
            MW[group][time][diameter].append(MW_ave)
        

    interpolated_actvations, normalized_times = utils_post.normalize_and_interpolate(times, activations)
    interpolated_pressures, _ = utils_post.normalize_and_interpolate(times, pressures)
    interpolated_fiber_strains, _ = utils_post.normalize_and_interpolate(times, fiber_strains)
    interpolated_MW, _ = utils_post.normalize_and_interpolate(times, MW)

    averaged_actvations, std_actvations = utils_post.calculate_data_average_and_std(interpolated_actvations)
    averaged_pressures, std_pressures = utils_post.calculate_data_average_and_std(interpolated_pressures)
    averaged_fiber_strains, std_fiber_strains = utils_post.calculate_data_average_and_std(interpolated_fiber_strains)
    averaged_MW, std_MW = utils_post.calculate_data_average_and_std(interpolated_MW)
    normalized_times, _ = utils_post.calculate_data_average_and_std(normalized_times)

    
    # Define a dictionary for each variable you want to plot
    plot_vars = {
        "activation": {
            "avg": averaged_actvations,
            "std": std_actvations,
            "ylim": (-10, 110),
            "ylabel": "Cardiac Muscle Tension Generation (Activation) [kPa]",
            "fname_prefix": "activation",
        },
        "pressure": {
            "avg": averaged_pressures,
            "std": std_pressures,
            "ylim": (-2, 30),
            "ylabel": "LV Pressure [kPa]",
            "fname_prefix": "pressure",
        },
        "strain": {
            "avg": averaged_fiber_strains,
            "std": std_fiber_strains,
            "ylim": (-0.1, 0),
            "ylabel": "Averaged Fiber Strains [-]",
            "fname_prefix": "strain",
        },
        "work": {
            "avg": averaged_MW,
            "std": std_MW,
            "ylim": (-4, 4),
            "ylabel": "Averaged Myocardial Work [mJ]",
            "fname_prefix": "work",
        },
    }
    
    utils_post.export_results(output_folder, plot_vars, normalized_times)
    group_names = ["SHAM", "107", "130", "150"]  
    utils_post.export_group_results(output_folder, plot_vars, group_names, normalized_times)
if __name__ == "__main__":
    main()
# %%
