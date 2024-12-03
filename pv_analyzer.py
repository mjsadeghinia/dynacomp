import argparse
import json
import matplotlib.pyplot as plt
import pymatreader
from pathlib import Path
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
    dt = data["channel_meta"]["dt"][p_channel + 1][recording_num]

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

    plt.plot(data["volumes"], data["pressures"])
    fname = output_dir / f"raw_data_rec_{recording_num}.png"
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    main()
