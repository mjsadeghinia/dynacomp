import os
import subprocess
from pathlib import Path
import argparse
import utils

""" 
Script to copy prepared data to ex3 server
THIS SHOULD BE RUN LOCALLY NOT ON DOCKER
"""

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        '--number',
        nargs='*',
        type=int,
        default=None,
        help='Sample number(s) to process. If omitted, all samples in settings_dir will be processed.'
    )
    parser.add_argument(
        "-i",
        "--ID",
        nargs="+",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )
    parser.add_argument(
        '--settings_dir',
        type=Path,
        default=Path('/Users/javad/Docker/dynacomp/dynacomp/settings'),
        help='Directory where JSON settings files are stored.'
        )
    
    parser.add_argument(
        '-s',
        '--scan_type',
        type=str,
        default='TPM',
        help='Scan type; subdirectories will be named accordingly.'
    )

    parser.add_argument(
        '--local_results_dir',
        type=Path,
        default=Path('/Users/javad/Docker/dynacomp/01_results_coarse_mesh'),
        help='Directory where results will be saved.'
    )

    parser.add_argument(
        '--remote_results_dir',
        type=str,
        default="/global/D1/homes/sadeghinia/01_results_coarse_mesh",
        help='Directory where results will be saved in the remote ex3 server.'
    )

    parser.add_argument(
        '--remote_user',
        type=str,
        default="sadeghinia",
        help='Username for the remote ex3 server.'
    )

    parser.add_argument(
        '--import_flag',
        action='store_true',
        help='If set, the data will be imported from the remote server.'
    )

    parser.add_argument(
        "-f",
        "--folders",
        nargs="+",
        type=str,
        help="The folders to be imported processed.",
    )

    args = parser.parse_args()

    number = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    scan_type = args.scan_type
    local_results_dir = Path(args.local_results_dir)
    remote_results_dir = args.remote_results_dir
    import_flag = args.import_flag
    remote_user = args.remote_user
    REMOTE = f"{remote_user}@ex3"


    if sample_ID is not None:
        sample_nums = []
        for id in sample_ID:
            id_num = utils.get_num_from_id(id, settings_dir)
            sample_nums.append(id_num)
    elif number:
        sample_nums = number
    else:
        settings_files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(settings_files) + 1))

    for sample_num in sample_nums:
        settings = utils.load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        sample_dir = local_results_dir / sample_name / scan_type

        if import_flag:
            if args.folders is None:
                print("Please specify folders to import using --folders")
                continue
            
            for folder in args.folders:
                remote_src = f"{REMOTE}:{remote_results_dir}/{sample_name}/{scan_type}/{folder}/"
                local_dest = sample_dir / folder
                
                # Ensure local destination exists
                local_dest.mkdir(parents=True, exist_ok=True)
                
                # Run rsync
                cmd = [
                    "rsync", "-avh", "--progress",
                    remote_src,
                    str(local_dest)
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

        else:
            # Construct relevant paths
            pvcalib_src = sample_dir / "01_PVCalibration"
            if not pvcalib_src.exists():
                continue
            
            # Remote destination
            dest_dir = f"{remote_results_dir}/{sample_name}/{scan_type}/01_PVCalibration/"
            remote_dest = f"{REMOTE}:{dest_dir}"
            
            # Ensure remote destination exists
            mkdir_cmd = ["ssh", REMOTE, f"mkdir -p {dest_dir}"]
            print("Ensuring remote dir:", " ".join(mkdir_cmd))
            subprocess.run(mkdir_cmd, check=True)
            
            # Files and dirs to copy
            items_to_copy = [
                pvcalib_src / "Geometries",
                pvcalib_src / f"{sample_name}_EDPVR_calibrated_shifted.csv",
                pvcalib_src / "ordered_calibrated_pv_data.csv",
            ]
            
            for item in items_to_copy:
                if not item.exists():
                    print(f"Skipping missing: {item}")
                    continue
                
                # Run rsync
                cmd = [
                    "rsync", "-avh", "--progress",
                    str(item),
                    remote_dest
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()