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
        help="The sample number(s), will process all the sample if not indicated and if the sample ID is not passed.",
    )

    parser.add_argument(
        "-i",
        "--ID",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )
    
    parser.add_argument(
        "-d",
       "--data_dir",
        default="/home/shared/00_data",
        type=Path,
        help="The settings directory where data files are stored.",
    )
    
    parser.add_argument(
        "-st",
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    
    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=str,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-m",
        "--mesh_quality",
        default='coarse',
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    
    parser.add_argument(
        "-s",
        "--scan_type",
        default='TPM',
        type=str,
        help="The scan type. Settings will be loaded accordingly from json file",
    )
    
    parser.add_argument(
        "-t",
        "--time_mesh",
        default=None,
        type=int,
        nargs='+',
        help="The time fram to create the mesh from, if specified would overwrite the settings json file",
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
        "-i",
        "--ID",
        nargs="+",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )
    
    parser.add_argument(
        "-st",
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
        "-m",
        "--mesh_quality",
        default='coarse',
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    
    parser.add_argument(
        "-s",
        "--scan_type",
        default='TPM',
        type=str,
        help="The scan type. Settings will be loaded accordingly from json file",
    )

    parser.add_argument(
        '--a_matparam',
        type=float,
        default=None,
        help='Material parameter a for the heart model.'
    )
    parser.add_argument(
        '--af_matparam',
        type=float,
        default=None,
        help='Material parameter a_f for the heart model.'
    )
    parser.add_argument(
        '--b_matparam',
        type=float,
        default=None,
        help='Material parameter a for the heart model.'
    )
    parser.add_argument(
        '--bf_matparam',
        type=float,
        default=None,
        help='Material parameter a_f for the heart model.'
    )

    parser.add_argument(
        '-o',
        "--output_folder",
        default="02_Unloading",
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



def prepare_oudir_processing(outdir, comm=None):
    # outdir = data_dir / f"{output_folder}/00_Modeling"
    
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


def prepare_outdir(outdir):
    """
    Prepare the output directory by removing all files and folders if it exists,
    and ensuring it is created again.
    """    
    # If the directory exists, remove it and all its contents
    if outdir.exists():
        try:
            shutil.rmtree(outdir)
        except OSError:
            import subprocess
            subprocess.run(['rm', '-rf', str(outdir)], check=False)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def prepare_matparams(args):
    matparams = {
        'a': args.a_matparam,
        'a_f': args.af_matparam,
        'b': args.b_matparam,
        'b_f': args.bf_matparam
    }
    # keep only those entries where the value is not None
    matparams = {k: v for k, v in matparams.items() if v is not None}
    return matparams