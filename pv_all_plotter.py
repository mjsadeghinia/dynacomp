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
def plot_all_pv_recording(pv_data_dir, sample_name, output_dir):
    # Check if directory exist
    if not pv_data_dir.is_dir():
        logger.error("the folder does not exist")

    # Ensure there is exactly one .mat file
    mat_files = list(pv_data_dir.glob("*.mat"))
    if len(mat_files) != 1:
        logger.error("Data folder must contain exactly 1 .mat file.")

    mat_file = mat_files[0]

    data = pymatreader.read_mat(mat_file)
    p_channel = get_pressure_channel(data["channel_meta"])
    v_channel = get_volume_channel(data["channel_meta"])
    for recording_num in range(20):
        recording_num += 1
        if f"data__chan_{p_channel+1}_rec_{recording_num}" not in data.keys():
            continue
        pressures = data[f"data__chan_{p_channel+1}_rec_{recording_num}"]
        volumes = data[f"data__chan_{v_channel+1}_rec_{recording_num}"]
        plot_pv_recording(pressures, volumes, recording_num, output_dir, sample_name)



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

def plot_pv_recording(pres, vols, recording_num, output_dir, sample_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(vols, pres, linewidth=0.5)
    fname = output_dir / f"{sample_name}_data_rec_{recording_num}.png"
    plt.xlabel("Volume (RVU)")
    plt.ylabel("LV Pressure (mmHg)")
    plt.title(f"Sample {sample_name} recording no. {recording_num}")
    plt.savefig(fname, dpi=300)
    plt.close()

# %%
def parse_arguments(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        nargs='+',
        type=int,
        help="The sample number, will be used if sample is not passing",
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
        default="All_PV_data",
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

    sample_nums = args.number
    if sample_nums is None:
        sample_nums = range(1, 58)
    setting_dir = args.settings_dir
    output_folder = args.output_folder
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in setting_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    for sample_num in sample_nums:
        sample_name = sorted_files[sample_num - 1].with_suffix("").name

        logger.info(f"Sample {sample_name} is being processed...")

        settings = load_settings(setting_dir, sample_name)
        recording_num = settings["PV"]["recording_num"]
        data_dir = Path(settings["path"])
        pv_data_dir = data_dir / "PV Data"
        

        plot_all_pv_recording(pv_data_dir, sample_name, output_dir)
    
if __name__ == "__main__":
    main()
