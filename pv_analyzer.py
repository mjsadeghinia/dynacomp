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
    logger.error("Volume channel has not found!")
    return -1


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


def fit_bspline(x, y, smooth_level=1, bspline_degree=3):
    x = np.array(x)
    y = np.array(y)

    # Ensure the curve is closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Fit a B-spline
    points = np.array([x, y])
    tck, u = splprep(points, s=smooth_level, per=True, k=bspline_degree)

    return tck


# %%
def parse_arguments(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--name",
        default="100_1",
        type=str,
        help="The sample file name to be p",
    )

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-m",
        "--mesh_quality",
        default="fine_mesh",
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )

    parser.add_argument(
        "-r",
        "--recording_num",
        default=10,
        type=int,
        help="The number of recording to be read from PV data",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default="refined_data",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
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

    sample_name = args.name
    setting_dir = args.settings_dir
    mesh_quality = args.mesh_quality
    output_folder = args.output_folder
    recording_num = args.recording_num

    settings = load_settings(setting_dir, sample_name)
    data_dir = Path(settings["path"])
    pv_data_dir = data_dir / "PV Data"
    output_dir = pv_data_dir / output_folder
    output_dir.mkdir(exist_ok=True)

    data = load_pv_data(pv_data_dir, recording_num=recording_num)
    vols, pres = data["volumes"], data["pressures"]
    plt.plot(vols, pres)
    fname = output_dir / f"raw_data_rec_{recording_num}.png"
    plt.xlabel("Volume (RVU)")
    plt.ylabel("LV Pressure (mmHg)")
    plt.savefig(fname, dpi=300)
    plt.close()

    pres_divided, vols_divided = divide_pv_data(pres, vols)
    pres_average, vols_average, time_average = average_pv_data(
        pres_divided, vols_divided, data["dt"]
    )
    # Smoothing data
    smoothed_vols_average = savgol_filter(vols_average, window_length=15, polyorder=3)
    smoothed_pres_average = savgol_filter(pres_average, window_length=15, polyorder=3)
    # Removing redundant volume and pressure data
    v_0 = smoothed_vols_average[0]
    ind = 10 - np.where(smoothed_vols_average[-10:] < v_0)[0][0]

    volumes = smoothed_vols_average[:-ind]
    pressures = smoothed_pres_average[:-ind]
    time = time_average[:-ind]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(time_average * 1000, vols_average, s=20)
    ax.plot(time_average * 1000, vols_average, "b")
    ax.plot(time * 1000, volumes, "k")
    plt.xlabel("time [s]")
    plt.ylabel("Volume [RVU]")
    fname = output_dir / f"volume_data_rec_{recording_num}_average.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(time_average * 1000, pres_average, s=20)
    ax.plot(time_average * 1000, pres_average, "b")
    ax.plot(time * 1000, pressures, "k")
    plt.xlabel("time [s]")
    plt.ylabel("LV Pressure [mmHg]")
    fname = output_dir / f"pressure_data_rec_{recording_num}_average.png"
    plt.savefig(fname, dpi=300)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(vols_divided)):
        ax.plot(vols_divided[i], pres_divided[i], "k", linewidth=0.02)
    ax.scatter(volumes, pressures, s=20)
    ax.plot(volumes, pressures, "k")
    fname = output_dir / f"raw_data_rec_{recording_num}_average.png"
    plt.xlabel('Volume (RVU)')
    plt.ylabel('LV Pressure (mmHg)') 
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
