import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import utils

from structlog import get_logger

logger = get_logger()


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
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-rf",
        "--results_folder",
        default="PV Data",
        type=str,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-edpvr",
        action="store_true",
        help="If set, the script will process EDPVR modeling data.",
    )

    parser.add_argument(
        "-f",
        "--filename",
        default="EDPVR",
        type=str,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/home/shared/02_post_processing/02_EDPVR_Modeling",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    return parser.parse_args(args)


def load_settings(setting_dir, sample_num):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
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
    # Getting the arguments
    sample_num = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    results_folder = args.results_folder
    filename = args.filename
    output_dir = args.output_dir
    edpvr_flag = args.edpvr
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load settings
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])

    if sample_ID is not None:
        sample_num = get_num_from_id(sample_ID, settings_dir)
        sample_nums = [sample_num]
    elif sample_num is None:
        sample_nums = range(1, len(sorted_files) + 1)

    for n in sample_nums:
        settings = load_settings(settings_dir, n)
        sample_name = settings["id"]
        if edpvr_flag:
            edpvr_folder = Path(results_dir) / sample_name / "TPM" / results_folder

            if not edpvr_folder.exists():
                logger.warning(f"EDPVR folder {edpvr_folder} does not exist for sample {sample_name}")
                continue
            
            output_dir_sample = output_dir / sample_name
            output_dir_sample.mkdir(parents=True, exist_ok=True)
            edpvr_data = utils.read_edpvr_data(edpvr_folder)
            experimets_folders = utils.get_folders_from_edpvr_data(edpvr_data)
            for i, folder in enumerate(experimets_folders):
                    fname = edpvr_folder / folder / "inflation_results.png"
                    if not fname.exists():
                        logger.warning(f"{fname} does not exist")
                        continue
                    outname = output_dir_sample / f"{i}.png"
                    shutil.copy(fname, outname)
        else:
            fname = Path(results_dir) / sample_name / "TPM" / results_folder / f"{filename}.png"
            if not fname.exists():
                logger.warning(f"File {fname} does not exist")
                continue
            outname = output_dir / f"{sample_name}_{filename}.png"
            shutil.copy(fname, outname)


if __name__ == "__main__":
    main()
