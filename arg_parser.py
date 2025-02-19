import argparse
from pathlib import Path
import shutil
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
    
    parser.add_argument(
        "-n",
        "--number",
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
        "-m",
        "--mesh_quality",
        default='coarse',
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    
    parser.add_argument(
        "-t",
        "--time_mesh",
        default=None,
        type=int,
        help="The time fram to create the mesh from, if specified would overwrite the settings json file",
    )
    
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "output",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    # Create a mutually exclusive group to allow only one of the two options
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--h5_overwrite",
        dest="h5_overwrite",
        action="store_true",
        help="Overwrite the h5 file (default behavior)."
    )
    group.add_argument(
        "--no-h5_overwrite",
        dest="h5_overwrite",
        action="store_false",
        help="Do not overwrite the h5 file."
    )
    parser.set_defaults(h5_overwrite=True)

    return parser.parse_args(args)


def parse_arguments_unloading(args=None):
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
    
    # Arguments for HeartModel boundary conditions
    parser.add_argument(
        "--pericardium_spring",
        default=0.0001,
        type=float,
        help="HeartModel BC: The stiffness of the spring on the pericardium.",
    )
    parser.add_argument(
        "--base_spring",
        default=1,
        type=float,
        help="HeartModel BC: The stiffness of the spring at the base.",
    )
    
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "fine_mesh",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    return parser.parse_args(args)



def parse_arguments_processing(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="The sample number(s), will process all the sample if not indicated",
    )
    
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    # Arguments for HeartModel boundary conditions
    parser.add_argument(
        "--pericardium_spring",
        default=0.0001,
        type=float,
        help="HeartModel BC: The stiffness of the spring on the pericardium.",
    )
    parser.add_argument(
        "--base_spring",
        default=1,
        type=float,
        help="HeartModel BC: The stiffness of the spring at the base.",
    )
    
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "fine_mesh",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    return parser.parse_args(args)

def create_bc_params(args):
    """
    Create a dictionary of B.C. parameters from the parsed arguments.
    """
    return {
        "pericardium_spring": args.pericardium_spring,
        "base_spring": args.base_spring,
    }



def prepare_oudir_processing(data_dir, output_folder, comm=None):
    outdir = data_dir / f"{output_folder}/00_Modeling"
    
    if comm is None:
        import dolfin
        comm = dolfin.MPI.comm_world
        
    if comm.rank == 0:
        # Create the directory if it doesn't exist
        if not outdir.exists():
            outdir.mkdir(parents=True)
        else:
            # Remove the directory contents but not the directory itself
            for item in outdir.iterdir():
                if item.is_file():
                    item.unlink()  # Remove file
                elif item.is_dir():
                    shutil.rmtree(item)  # Remove directory
    return outdir


def prepare_outdir(data_dir, output_folder):
    """
    Prepare the output directory by removing all files and folders if it exists,
    and ensuring it is created again.
    """
    outdir = data_dir / output_folder
    
    # If the directory exists, remove it and all its contents
    if outdir.exists():
        shutil.rmtree(outdir)
    
    # Create the directory again
    outdir.mkdir(exist_ok=True, parents=True)
    return outdir
