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
        logger.warning(f"Channel no. {recording_num} is specified by the user")
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
    # Determine the length of the longest array
    max_length = max(len(array) for array in arrays)

    # Define common x values
    average_x = np.linspace(0, max_length - 1, num=n_points)

    # Interpolate each array to the common x-axis
    interpolated_arrays = []
    for array in arrays:
        x = np.linspace(0, len(array) - 1, num=len(array))
        f = interp1d(x, array, kind="linear", fill_value="extrapolate")
        interpolated_y = f(average_x)
        interpolated_arrays.append(interpolated_y)

    # Convert list of arrays to a 2D NumPy array for averaging
    interpolated_arrays = np.array(interpolated_arrays)

    # Calculate the average along the common x-axis
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

def get_edpvr_cycles(pres):
    max_pres = [np.max(p) for p in pres]
    # Create a new list for the filtered descending sequence
    descending_sequence = [0]  
    for i in range(1, len(max_pres)):
        if max_pres[i] < max_pres[descending_sequence[-1]] and max_pres[i]-max_pres[descending_sequence[-1]]<-0.75 and max_pres[i]-max_pres[descending_sequence[-1]]>-10:
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
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="PV Data",
        type=str,
        help="The result folder name that would be created in the directory of the sample.",
    )

    parser.add_argument(
        "--output_edpvr",
        default="EDPVR_all_data",
        type=str,
        help="The result folder for all EDPVR data that would be created in the root directory.",
    )

    return parser.parse_args(args)


def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


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
    setting_dir = args.settings_dir
    output_folder = args.output_folder
    output_edpvr = Path('EDPVR_all_data')
    output_edpvr.mkdir(exist_ok=True)

    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in setting_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    
    if sample_nums is None:
        sample_nums = range(1,57)
        
    for sample_num in sample_nums:
        sample_name = sorted_files[sample_num - 1].with_suffix("").name

        logger.info(f"Sample {sample_name} is being processed...")

        settings = load_settings(setting_dir, sample_name)
        recording_num = settings["PV"]["recording_num"]
        data_dir = Path(settings["path"])
        pv_data_dir = data_dir / "PV Data"
        output_dir = pv_data_dir / output_folder
        output_dir.mkdir(exist_ok=True)

        data = load_pv_data(pv_data_dir, recording_num=recording_num)
        vols, pres = data["volumes"], data["pressures"]

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
            ind = ED_data_num - np.where(vols_average[-ED_data_num:] < v_0)[0][0]
            vols_average = vols_average[:-ind]
            pres_average = pres_average[:-ind]
            time_average = time_average[:-ind]

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
            pres_occlusion_divided_all, vols_occlusion_divided_all = divide_pv_data(pres_occlusion, vols_occlusion)

            if settings["PV"]["Occlusion_data_index_i"] is None and settings["PV"]["Occlusion_data_index_f"] is None:
                inds = get_edpvr_cycles(pres_occlusion_divided_all)
                pres_occlusion_divided = [pres_occlusion_divided_all[i] for i in inds]
                vols_occlusion_divided = [vols_occlusion_divided_all[i] for i in inds]
            else:
                first_cycle, last_cycle = settings["PV"]["Occlusion_data_index_i"], settings["PV"]["Occlusion_data_index_f"]
                inds = np.linspace(first_cycle,last_cycle,dtype=int)
                pres_occlusion_divided = pres_occlusion_divided_all[first_cycle:last_cycle]
                vols_occlusion_divided = vols_occlusion_divided_all[first_cycle:last_cycle]
                                                                
            # Plotting maximum pressure in occlusion acquisiton
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, p in enumerate(pres_occlusion_divided_all):
                ax.scatter(i, np.max(p), s=5, c="k")
                if i in inds:
                    ax.scatter(i, np.max(p), s=5, c="r")
            plt.ylabel("Max LV Pressure during Caval Occlusion [mmHg]")
            plt.xlabel("Cycle no.")
            plt.grid()
            fname = output_dir / f"{sample_name}_EDPVR_max_Pressure.png"
            plt.savefig(fname, dpi=300)
            fname = output_edpvr / f"{sample_name}_EDPVR_max_Pressure.png"
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
            fname = output_edpvr / f"{sample_name}_EDPVR.png"
            plt.savefig(fname, dpi=300)
            plt.close()

            fname = output_dir / f"{sample_name}_EDPVR.csv"
            np.savetxt(fname, np.vstack((edpvr_p, edpvr_v)).T, delimiter=",")
            logger.info("------------------")

if __name__ == "__main__":
    main()
