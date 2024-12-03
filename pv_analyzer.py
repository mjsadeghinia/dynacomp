import argparse
import json
from pathlib import Path
from structlog import get_logger

logger = get_logger()


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

    return parser.parse_args(args)


def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


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

    settings = load_settings(setting_dir, sample_name)
    data_dir = Path(settings["path"])


if __name__ == "__main__":
    main()
