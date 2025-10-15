import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pymatreader
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, splprep, splev
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import linregress

import arg_parser

from structlog import get_logger

logger = get_logger()


# %%
def load_pv_data(pv_data_dir, recording_num=2):
    # Check if directory exist
    if not pv_data_dir.is_dir():
        logger.error("the folder does not exist")

    # Ensure there is exactly one .mat file
    mat_files = list(pv_data_dir.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")

    mat_file = mat_files[0]
    logger.info(f"{mat_file.name} is loading.")

    data = pymatreader.read_mat(mat_file)
    p_channel = get_pressure_channel(data["channel_meta"])
    v_channel = get_volume_channel(data["channel_meta"])

    pressures = data[f"data__chan_{p_channel+1}_rec_{recording_num}"]
    volumes = data[f"data__chan_{v_channel+1}_rec_{recording_num}"]
    dt = data["channel_meta"]["dt"][p_channel][recording_num]

    return {"pressures": pressures, "volumes": volumes, "dt": dt}


def get_pressure_channel(channel_meta):
    num_chan = len(channel_meta["units"])
    for i in range(num_chan):
        if all(element == "mmHg" for element in channel_meta["units"][i]):
            return i
    logger.error("Pressure channel has not found!")
    return -1


def get_volume_channel(channel_meta):
    num_chan = len(channel_meta["units"])
    for i in range(num_chan):
        if all(element == "RVU" for element in channel_meta["units"][i]):
            return i
        if all(element == "L" for element in channel_meta["units"][i]):
            logger.warning("Volume channel unit was not RVU but L!")
            return i
    logger.error("Volume channel has not found!")
    return -1


def load_caval_occlusion_data(pv_data_dir, occlusion_recording_num=None):
    # Check if directory exist
    if not pv_data_dir.is_dir():
        logger.error("the folder does not exist")

    # Ensure there is exactly one .mat file
    mat_files = list(pv_data_dir.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")

    mat_file = mat_files[0]
    data = pymatreader.read_mat(mat_file)
    if occlusion_recording_num is not None:
        recording_num = occlusion_recording_num
        logger.info(f"Channel no. {recording_num} is specified by the user")
    else:
        for i in range(1, len(data['comments']["str"])):
            comment = data['comments']['str'][-i].lower()
            # NB! the metadata is not fully right and consistent with typos as occulution or occulatio
            if 'caval' in comment or 'occ' in comment :                
                recording_num = int(data['comments']['record'][-i])
                logger.info(f"Channel no. {recording_num} is selected for Caval occlusion based on metadata")
                break
            else:
                logger.error("Metadata Caval occlusion is not in the dataset! Check the metadata")
                print(comment)
    p_channel = get_pressure_channel(data["channel_meta"])
    v_channel = get_volume_channel(data["channel_meta"])
    pressures = data[f"data__chan_{p_channel+1}_rec_{recording_num}"]
    volumes = data[f"data__chan_{v_channel+1}_rec_{recording_num}"]
    dt = data["channel_meta"]["dt"][p_channel][recording_num-1]
    return {"pressures": pressures, "volumes": volumes, "dt": dt}

def divide_pv_data(pres, vols):
    # Dividing the data into different curves
    pres_divided = []
    vols_divided = []
    peaks, _ = find_peaks(pres, distance=150)

    num_cycles = int(len(peaks))
    for i in range(num_cycles - 1):
        pres_divided.append(pres[peaks[i] : peaks[i + 1]])
        vols_divided.append(vols[peaks[i] : peaks[i + 1]])

    return pres_divided, vols_divided


def average_pv_data(pres_divided, vols_divided, dt, n_points=100):
    # average time
    pres_len = [len(array) for array in pres_divided]
    vols_len = [len(array) for array in vols_divided]
    average_len = np.average([pres_len, vols_len])
    time_average = np.linspace(0, average_len * dt, n_points)

    # average pressure and volume
    pres_average = average_array(pres_divided, n_points)
    vols_average = average_array(vols_divided, n_points)

    return pres_average, vols_average, time_average


def average_array(arrays, n_points):
    interpolated_arrays = []
    target_x = np.linspace(0, 1, n_points)
    for arr in arrays:
        original_len = len(arr)
        original_x = np.linspace(0, 1, original_len)
        interpolator = interp1d(original_x, arr, kind='linear', fill_value="extrapolate")
        interpolated_arr = interpolator(target_x)
        interpolated_arrays.append(interpolated_arr)

    interpolated_arrays = np.array(interpolated_arrays)

    average_y = np.mean(interpolated_arrays, axis=0)
    return average_y

def get_end_diastole_ind(
    pressures, volumes, pressure_threshold_percent=0.1, volume_threshold_percent=0.05
):
    # Calculate the thresholds for pressure and volume
    pressure_min = np.min(pressures)
    # Define the range for end-diastole
    pressure_threshold = pressure_min + pressure_threshold_percent * (
        np.max(pressures) - pressure_min
    )
    # Find indices where pressure is below the threshol
    valid_pressure_indices = np.where((pressures <= pressure_threshold))[0]
    new_volumes = volumes[valid_pressure_indices]
    new_volume_max = np.max(new_volumes)
    new_volume_min = np.min(new_volumes)
    volume_threshold = new_volume_max - volume_threshold_percent * (
        new_volume_max - new_volume_min
    )

    # Find indices where conditions are met
    valid_indices = np.where(
        (pressures <= pressure_threshold) & (volumes >= volume_threshold)
    )[0]
    # Find the index of the maximum volume in the valid region
    index = valid_indices[np.argmax(volumes[valid_indices])]

    return index

def get_edpvr_cycles(pres, max_pres_diff_edpvr_cycle = 10, min_pres_diff_edpvr_cycle = 0.75):
    max_pres = [np.max(p) for p in pres]
    # Create a new list for the filtered descending sequence
    descending_sequence = [0]  
    for i in range(1, len(max_pres)):
        if max_pres[i] < max_pres[descending_sequence[-1]] and max_pres[i]-max_pres[descending_sequence[-1]]<-min_pres_diff_edpvr_cycle and max_pres[i]-max_pres[descending_sequence[-1]]>-max_pres_diff_edpvr_cycle:
            descending_sequence.append(i)
    run = first_consecutive_run(descending_sequence)
    return descending_sequence[run:]

def first_consecutive_run(lst):
    # Build the consecutive run starting at the valid first element
    for i in range(len(lst)-1):
        if lst[i+1]-lst[i] <= 2:
            return i
    return 0

def delete_previous_EDPVR(output_dir):
    for file in output_dir.iterdir():
        if "EDPVR" in file.stem:
            file.unlink()

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
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    parser.add_argument(
        "-d",
       "--data_dir",
        default="/home/shared/00_data",
        type=Path,
        help="The settings directory where data files are stored.",
    )
    
    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        '-o',
        "--output_folder",
        default="PV Data",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    return parser.parse_args(args)


def load_settings(settings_dir, sample_num):
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])
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

    sample_nums = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    output_folder = args.output_folder
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in settings_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )

    if sample_ID is not None:
        sample_nums = [get_num_from_id(sample_ID, settings_dir)]

    if sample_nums is None:
        sample_nums = range(1,len(sorted_files)+1)
        
    for sample_num in sample_nums:
        settings = load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        pv_data_dir = data_dir / sample_name / "PV Data"
        output_dir = results_dir / sample_name / output_folder
        output_dir = arg_parser.prepare_outdir(output_dir)
        # output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Sample {sample_name} is being processed...")

        recording_num = settings["PV"]["recording_num"]
        data = load_pv_data(pv_data_dir, recording_num=recording_num)
        vols, pres = data["volumes"], data["pressures"]

        if "total_pressure_offset" in settings["PV"]:
            pres += settings["PV"]["total_pressure_offset"]
            logger.warning(f"Applied a total pressure offset of {settings['PV']['total_pressure_offset']} mmHg, check if it was necessary")

        pres_divided, vols_divided = divide_pv_data(pres, vols)
        pres_average, vols_average, time_average = average_pv_data(
            pres_divided, vols_divided, data["dt"]
        )
        # Removing redundant volume and pressure data
        if settings["PV"]["skip_redundant_data_flag"]:
            # Smoothing data
            smoothed_vols_average = savgol_filter(
                vols_average,
                window_length=settings["PV"]["volume_smooth_window_length"],
                polyorder=3,
            )
            smoothed_pres_average = savgol_filter(
                pres_average,
                window_length=settings["PV"]["pressure_smooth_window_length"],
                polyorder=3,
            )
            time = time_average
        else:
            v_0 = vols_average[0]
            ED_data_num = int(0.15 * len(vols_average))

            # Smoothing data
            smoothed_vols_average = savgol_filter(
                vols_average,
                window_length=settings["PV"]["volume_smooth_window_length"],
                polyorder=3,
            )
            smoothed_pres_average = savgol_filter(
                pres_average,
                window_length=settings["PV"]["pressure_smooth_window_length"],
                polyorder=3,
            )

            # Removing redundant volume and pressure data if any arised from smoothing
            v_0 = smoothed_vols_average[0]
            ED_data_num = int(0.1 * len(smoothed_vols_average))
            ind_repeated = np.where(smoothed_vols_average[-ED_data_num:] <= v_0)[0]
            if ind_repeated.shape[0] > 0:
                ind = ED_data_num - ind_repeated[0]
                smoothed_vols_average = smoothed_vols_average[:-ind]
                smoothed_pres_average = smoothed_pres_average[:-ind]
                time = time_average[:-ind]
            else:
                time = time_average

        # reodering the data based on end diastole
        ind = get_end_diastole_ind(smoothed_pres_average, smoothed_vols_average)
        # Reorder the data to start from the identified index
        pressures = np.roll(smoothed_pres_average, -ind)
        volumes = np.roll(smoothed_vols_average, -ind)
        vols_average = np.roll(vols_average, -ind)
        pres_average = np.roll(pres_average, -ind)

        # Plotting

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(time_average * 1000, vols_average, s=20, label="Average Data Points")
        ax.plot(time_average * 1000, vols_average, color="b", label="Average Data Points")
        ax.plot(time * 1000, volumes, color="k", label="Smoothed Data")
        plt.xlabel("time [ms]")
        plt.ylabel("Volume [RVU]")
        plt.legend()
        fname = output_dir / f"{sample_name}_volume_data_rec_{recording_num}_average.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(time_average * 1000, pres_average, s=20, label="Average Data Points")
        ax.plot(time_average * 1000, pres_average, "b", label="Average Data Points")
        ax.plot(time * 1000, pressures, "k", label="Smoothed Data")
        plt.xlabel("time [ms]")
        plt.ylabel("LV Pressure [mmHg]")
        plt.legend()
        fname = output_dir / f"{sample_name}_pressure_data_rec_{recording_num}_average.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(vols_divided)):
            ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.02)
        ax.scatter(volumes, pressures, s=15, c="k")
        ax.scatter(volumes[0], pressures[0], c="r", s=20)
        ax.plot(volumes, pressures, "k")
        fname = output_dir / f"{sample_name}_data_rec_{recording_num}_average.png"
        plt.xlabel("Volume [RVU]")
        plt.ylabel("LV Pressure [mmHg]")
        plt.savefig(fname, dpi=300)
        plt.close()
        
        fname = output_dir / f"{sample_name}_PV_data.csv"
        np.savetxt(fname, np.vstack((time, pressures, volumes)).T, delimiter=",")

        # Processing the caval occlusion data for EDPVR
        delete_previous_EDPVR(output_dir)

        if settings["PV"]["process_occlusion_flag"]:
            occlusion_data = load_caval_occlusion_data(pv_data_dir, settings["PV"]["Occlusion_recording_num"])
            pres_occlusion, vols_occlusion = occlusion_data["pressures"], occlusion_data["volumes"]
            if "total_pressure_offset" in settings["PV"]:
                pres_occlusion += settings["PV"]["total_pressure_offset"]
                logger.warning(f"Applied a total pressure offset of {settings['PV']['total_pressure_offset']} mmHg TO EDPVR, check if it was necessary")
                
            pres_occlusion_divided_all, vols_occlusion_divided_all = divide_pv_data(pres_occlusion, vols_occlusion)

            if settings["PV"]["Occlusion_data_index_i"] is None and settings["PV"]["Occlusion_data_index_f"] is None:
                if "max_pres_diff_edpvr_cycle" in settings["PV"]:
                    max_pres_diff_edpvr_cycle = settings["PV"]["max_pres_diff_edpvr_cycle"]
                else:
                    max_pres_diff_edpvr_cycle = 10
                if "min_pres_diff_edpvr_cycle" in settings["PV"]:
                    min_pres_diff_edpvr_cycle = settings["PV"]["min_pres_diff_edpvr_cycle"]
                else:
                    min_pres_diff_edpvr_cycle = 0.75
                selected_inds = get_edpvr_cycles(pres_occlusion_divided_all, min_pres_diff_edpvr_cycle=min_pres_diff_edpvr_cycle, max_pres_diff_edpvr_cycle=max_pres_diff_edpvr_cycle)
                pres_occlusion_divided = [pres_occlusion_divided_all[i] for i in selected_inds]
                vols_occlusion_divided = [vols_occlusion_divided_all[i] for i in selected_inds]
            else:
                first_cycle, last_cycle = settings["PV"]["Occlusion_data_index_i"], settings["PV"]["Occlusion_data_index_f"]
                selected_inds = np.arange(first_cycle, last_cycle + 1, 1)
                pres_occlusion_divided = pres_occlusion_divided_all[first_cycle:last_cycle]
                vols_occlusion_divided = vols_occlusion_divided_all[first_cycle:last_cycle]
                                                                
            # Plotting maximum pressure in occlusion acquisiton
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, p in enumerate(pres_occlusion_divided_all):
                ax.scatter(i, np.max(p), s=5, c="k")
                if i in selected_inds:
                    ax.scatter(i, np.max(p), s=5, c="r")
            plt.ylabel("Max LV Pressure during Caval Occlusion [mmHg]")
            plt.xlabel("Cycle no.")
            plt.grid()
            fname = output_dir / f"{sample_name}_EDPVR_max_Pressure.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            
            # Processing data to cacluate EDPVR
            edpvr_p = []
            edpvr_v = []
            fig, ax = plt.subplots(figsize=(8, 6))
            skip_cycle = settings["PV"]["Occlusion_data_skip_index"]
            for p, v in zip(pres_occlusion_divided[::skip_cycle],vols_occlusion_divided[::skip_cycle]):
                ind = get_end_diastole_ind(p,v, pressure_threshold_percent=0.05, volume_threshold_percent=0.05)
                edpvr_p.append(p[ind])
                edpvr_v.append(v[ind])
                ax.plot(v, p, "k", linewidth=0.1)
                ax.scatter(v[ind], p[ind], s=5, c="r")
            edpvr_p = np.array(edpvr_p)
            edpvr_v = np.array(edpvr_v)
            res = linregress(edpvr_v, edpvr_p)
            plt.plot(edpvr_v, res.intercept + res.slope*edpvr_v, 'b', label='EDVPR')
            # Create a text box with the regression parameters and confidence intervals
            from scipy.stats import t
            tinv = lambda p, df: abs(t.ppf(p/2, df))
            ts = tinv(0.05, len(edpvr_v)-2)
            # Calculate the x value at which y = 0 using the regression line equation (avoid division by zero)
            v_0 = -res.intercept / res.slope if res.slope != 0 else float('nan')
            textstr = (
                f"slope (95%): {res.slope:.3f} $\pm$ {ts*res.stderr:.3f}\n"
                f"intercept (95%): {res.intercept:.3f} $\pm$ {ts*res.intercept_stderr:.3f}\n"
                f"$v_0$ (P=0): {v_0:.5f}"
            )
            ax.text(
                0.05, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            plt.xlabel("Volume [RVU]")
            plt.ylabel("LV Pressure [mmHg]")
            ax.axhline(y=0, color='gray', linestyle='--')
            fname = output_dir / f"{sample_name}_EDPVR.png"
            plt.savefig(fname, dpi=300)
            plt.close()

            fname = output_dir / f"{sample_name}_EDPVR.csv"
            np.savetxt(fname, np.vstack((edpvr_p, edpvr_v)).T, delimiter=",")
            logger.info("--------------------------------")

            # write the EDPVR_pv_data data to a file
            fname = output_dir / f"{sample_name}_EDPVR_pressure_data.csv"
            pres_occlusion_divided_selected_lists = [pres_occlusion_divided_all[i].tolist() for i in selected_inds]
            with open(fname, 'w') as f:
                json.dump(pres_occlusion_divided_selected_lists, f)

            fname = output_dir / f"{sample_name}_EDPVR_volume_data.csv"
            vols_occlusion_divided_selected_lists = [vols_occlusion_divided_all[i].tolist() for i in selected_inds]
            with open(fname, 'w') as f:
                json.dump(vols_occlusion_divided_selected_lists, f)
if __name__ == "__main__":
    main()
