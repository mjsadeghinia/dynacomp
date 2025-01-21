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

    fig_activations = plt.figure()
    fig_pressures= plt.figure()
    fig_fiber_strains = plt.figure()
    fig_MW = plt.figure()
    colors_dict, styles_dict = utils_post.get_colors_styles(averaged_actvations.keys())

    for key, normalized_time in normalized_times.items():
        if averaged_actvations[key] is None:
            continue

        utils_post.plot_and_save(
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
        fig_activations = utils_post.plot_data_with_std(
            averaged_actvations[key],
            normalized_time,
            std_values=None,
            figure=fig_activations,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )

        utils_post.plot_and_save(
            key,
            averaged_pressures[key],
            normalized_time,
            std_pressures[key],
            colors_dict,
            styles_dict,
            output_folder,
            ylim=(-2, 30),
            ylabel="LV Pressure [kPa]",
            fname_prefix="pressure",
        )
        fig_pressures = utils_post.plot_data_with_std(
            averaged_pressures[key],
            normalized_time,
            std_values=None,
            figure=fig_pressures,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )

        utils_post.plot_and_save(
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
        fig_fiber_strains = utils_post.plot_data_with_std(
            averaged_fiber_strains[key],
            normalized_time,
            std_values=None,
            figure=fig_fiber_strains,
            color=colors_dict[key],
            style=styles_dict[key],
            label=key,
        )
        
        utils_post.plot_and_save(
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
        fig_MW = utils_post.plot_data_with_std(
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
    ax.grid()
    # plt.legend()
    fname = output_folder / "Activation"
    fig_activations.savefig(fname.as_posix(), dpi=300)

    ax = fig_fiber_strains.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 0)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Averaged Fiber Strains [-]")
    ax.grid()
    # plt.legend()
    fname = output_folder / "Fiber_Strain"
    fig_fiber_strains.savefig(fname.as_posix(), dpi=300)


    ax = fig_MW.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("Averaged Myocaridal Work [mJ]")
    ax.grid()
    # plt.legend()
    fname = output_folder / "Myocardial_Work"
    fig_MW.savefig(fname.as_posix(), dpi=300)
    
    
    ax = fig_pressures.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 30)
    ax.set_xlabel("Normalized Time [-]")
    ax.set_ylabel("LV Pressure [kPa]")
    ax.grid()
    # plt.legend()
    fname = output_folder / "LV_pressure"
    fig_pressures.savefig(fname.as_posix(), dpi=300)
    
    fig_activations_group_sham = plt.figure()
    fig_activations_group_107 = plt.figure()
    fig_activations_group_130 = plt.figure()
    fig_activations_group_150 = plt.figure()
    
    fig_pressure_group_sham = plt.figure()
    fig_pressure_group_107 = plt.figure()
    fig_pressure_group_130 = plt.figure()
    fig_pressure_group_150 = plt.figure()
    
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
            fig_activations_group_sham = utils_post.plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_pressure_group_sham = utils_post.plot_data_with_std(
                averaged_pressures[key],
                normalized_time,
                std_values=std_pressures[key],
                figure=fig_pressure_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_sham = utils_post.plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            
            fig_mw_group_sham = utils_post.plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_sham,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
        if '107' in key:
            fig_activations_group_107 = utils_post.plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_pressure_group_107 = utils_post.plot_data_with_std(
                averaged_pressures[key],
                normalized_time,
                std_values=std_pressures[key],
                figure=fig_pressure_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_107 = utils_post.plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_107 = utils_post.plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_107,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
        if '130' in key:
            fig_activations_group_130 = utils_post.plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_pressure_group_130 = utils_post.plot_data_with_std(
                averaged_pressures[key],
                normalized_time,
                std_values=std_pressures[key],
                figure=fig_pressure_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_130 = utils_post.plot_data_with_std(
                averaged_MW[key],
                normalized_time,
                std_values=std_MW[key],
                figure=fig_mw_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_130 = utils_post.plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_130,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
        if '150' in key:
            fig_activations_group_150 = utils_post.plot_data_with_std(
                averaged_actvations[key],
                normalized_time,
                std_values=std_actvations[key],
                figure=fig_activations_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_pressure_group_150 = utils_post.plot_data_with_std(
                averaged_pressures[key],
                normalized_time,
                std_values=std_pressures[key],
                figure=fig_pressure_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_strains_group_150 = utils_post.plot_data_with_std(
                averaged_fiber_strains[key],
                normalized_time,
                std_values=std_fiber_strains[key],
                figure=fig_strains_group_150,
                color=colors_dict[key],
                style=styles_dict[key],
                label=key,
            )
            
            fig_mw_group_150 = utils_post.plot_data_with_std(
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
    'fig_pressure_group_sham':fig_pressure_group_sham,
    'fig_pressure_group_107': fig_pressure_group_107,
    'fig_pressure_group_130': fig_pressure_group_130,
    'fig_pressure_group_150': fig_pressure_group_150,
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
        ax.grid()
        if 'activations' in fig_name:
            ax.set_ylim(-10, 120)
            ax.set_ylabel("Cardiac Muscle Tension Generation (Activation) [kPa]")
        elif 'strain' in fig_name:
            ax.set_ylim(-0.1, 0)
            ax.set_ylabel("Averaged Fiber Strains [-]")
        elif 'mw' in fig_name:
            ax.set_ylim(-4, 4)
            ax.set_ylabel("Averaged Myocaridal Work [mJ]")
        elif 'pressure' in fig_name:
            ax.set_ylim(-2, 30)
            ax.set_ylabel("LV Pressure [kPa]")
            
        # Save the figure
        fname = output_folder / f"{fig_name}.png"  # Save as PNG or desired format
        fig.savefig(fname.as_posix(), dpi=300)

if __name__ == "__main__":
    main()
# %%
