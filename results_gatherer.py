import argparse
import json
from pathlib import Path
import shutil

from structlog import get_logger

logger = get_logger()

# %%
def parse_arguments(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
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
        "-f",
        "--filename",
        default="EDPVR",
        type=str,
        help="The results folder where the processed data should be saved.",
    )
    
    parser.add_argument(
        "-o",
        "--output_dir",
        default="/home/shared/02_post_processing/EDPVR",
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
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    results_folder = args.results_folder
    filename = args.filename
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load settings     
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in settings_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    
    sample_nums = range(1,len(sorted_files)+1)
    
    for n in sample_nums:
        settings = load_settings(settings_dir, n)
        sample_name = settings["id"]
        fname = Path(results_dir) / sample_name / results_folder / f"{sample_name}_{filename}.png"
        if not fname.exists():
            logger.warning(f"File {fname} does not exist")
            continue
        shutil.copy(fname, output_dir)
        
        
if __name__ == "__main__":
    main()
