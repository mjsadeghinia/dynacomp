import argparse
from pathlib import Path
from structlog import get_logger

logger = get_logger()


def update_arguments(args, step='pre'):
    # If args is provided, merge with defaults
    if step == 'pre':
        default_args = parse_arguments_pre()
    elif step == 'unloading':
        default_args = parse_arguments_unloading()
    elif step == 'processing':
        default_args = parse_arguments_processing()
    else:
        logger.error('the update arguments step should be pre, unloading or processing')
    # Convert to namespace and update the defaults with provided args
    default_args = vars(default_args)
    for key, value in vars(args).items():
        if value is not None:
            default_args[key] = value
    args = argparse.Namespace(**default_args)
    return args


def parse_arguments_pre(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Geometry parameters
    
    valid_sample_names = ["156_1", "OP130_2", "138_1", "129_1"]
    parser.add_argument(
        "-n",
        "--name",
        default="156_1",
        choices=valid_sample_names,
        type=str,
        help="The sample file name to be p",
    )
    
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    valid_mesh_qualities = ['coarse', 'fine']
    parser.add_argument(
        "-m",
        "--mesh_quality",
        default='coarse',
        choices=valid_mesh_qualities,
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    
    
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "output",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    parser.add_argument(
        "--h5_overwrite",
        default=True,
        type=bool,
        help="The flag to overwrtie the current h5 file compiled from mat data.",
    )

    return parser.parse_args(args)


def parse_arguments_unloading(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    valid_sample_names = ["156_1", "OP130_2", "138_1", "129_1"]
    parser.add_argument(
        "-n",
        "--name",
        default="156_1",
        choices=valid_sample_names,
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
        "-o",
        "--output_folder",
        default= "output",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    return parser.parse_args(args)



def parse_arguments_processing(args=None):
    """
    Parse the command-line arguments.
    """
    return None



def prepare_outdir(data_dir, output_folder):
    """
    Prepare the output directory, ensuring it exists.
    """
    outdir = data_dir / output_folder
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir
