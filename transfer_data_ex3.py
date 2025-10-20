import os
import numpy as np
import subprocess
from pathlib import Path
import argparse
import utils
import shlex

""" 
Script to copy prepared data to ex3 server
THIS SHOULD BE RUN LOCALLY NOT ON DOCKER
"""

def remote_exists(remote_host: str, remote_path: str, kind: str = "any") -> bool:
    """
    kind: "file" -> test -f, "dir" -> test -d, "any" -> test -e
    """
    flag = {"file": "-f", "dir": "-d", "any": "-e"}[kind]
    cmd = [
        "ssh",
        remote_host,
        f"bash -lc 'test {flag} {shlex.quote(remote_path)}'"
    ]
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return res.returncode == 0

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
        '--export_data_flag',
        action='store_true',
        help='If set, the PV data will be exported to the remote server.'
    )

    parser.add_argument(
        "-f",
        "--folders",
        nargs="+",
        type=str,
        help="The folders to be imported processed.",
    )

    parser.add_argument(
        "--pv_folder",
        default="01_PVCalibration",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args()

    number = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    scan_type = args.scan_type
    local_results_dir = Path(args.local_results_dir)
    remote_results_dir = args.remote_results_dir
    pv_folder = args.pv_folder
    import_flag = args.import_flag
    export_data_flag = args.export_data_flag
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
                if folder == "02_Fiber_Modeling":
                    # Transfer only the Fiber_results.csv and contour file
                    files = ["Fiber_results.csv", "Fiber_contour.png"]
                    for file in files:
                        # --- CHANGED: check path-only on remote ---
                        remote_path_only = f"{remote_results_dir}/{sample_name}/{scan_type}/{folder}/{file}"
                        if not remote_exists(REMOTE, remote_path_only, kind="file"):
                            print(f"Skipping missing file on ex3: {remote_path_only}")
                            continue
                        remote_src = f"{REMOTE}:{remote_path_only}"

                        local_dest = sample_dir / folder
                        local_dest.mkdir(parents=True, exist_ok=True)
                        cmd = ["rsync", "-avh", "--progress", remote_src, str(local_dest)]
                        print("Running:", " ".join(cmd))
                        subprocess.run(cmd, check=True)

                    # Also transfer the best fit and -60 +60 files
                    csv_local = sample_dir / folder / "Fiber_results.csv"
                    if not csv_local.exists():
                        print(f"Missing local file: {csv_local}")
                        continue
                    fiber_results = np.loadtxt(csv_local, delimiter=',', skiprows=1)
                    ind_best = np.argmin(fiber_results[:,9])
                    best_fit_epi = fiber_results[ind_best,2]
                    best_fit_endo = fiber_results[ind_best,3]
                    best_fit_folder = f"epi_{best_fit_epi:.0f}_endo_{best_fit_endo:.0f}"
                    initial_fiber_folder = f"epi_-60_endo_60"
                    fiber_folders = [best_fit_folder, initial_fiber_folder]
                    for fiber_folder in fiber_folders:
                        # --- CHANGED: check remote dir existence ---
                        remote_folder_path = f"{remote_results_dir}/{sample_name}/{scan_type}/{folder}/{fiber_folder}"
                        if not remote_exists(REMOTE, remote_folder_path, kind="dir"):
                            print(f"Skipping missing remote folder on ex3: {remote_folder_path}")
                            continue
                        remote_src = f"{REMOTE}:{remote_folder_path}/"

                        local_dest = sample_dir / folder / fiber_folder
                        local_dest.mkdir(parents=True, exist_ok=True)
                        cmd = ["rsync", "-avh", "--progress", remote_src, str(local_dest)]
                        print("Running:", " ".join(cmd))
                        subprocess.run(cmd, check=True)

                else:
                    # --- CHANGED: check remote dir existence before rsync ---
                    remote_folder_path = f"{remote_results_dir}/{sample_name}/{scan_type}/{folder}"
                    if not remote_exists(REMOTE, remote_folder_path, kind="dir"):
                        print(f"Skipping missing remote folder on ex3: {remote_folder_path}")
                        continue
                    remote_src = f"{REMOTE}:{remote_folder_path}/"

                    local_dest = sample_dir / folder
                    local_dest.mkdir(parents=True, exist_ok=True)
                    cmd = ["rsync", "-avh", "--progress", remote_src, str(local_dest)]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)

        else:
            # Remote destination
            dest_dir = f"{remote_results_dir}/{sample_name}/{scan_type}"
            remote_dest = f"{REMOTE}:{dest_dir}"
            
            if export_data_flag:
                pvcalib_src = sample_dir / pv_folder
                if not pvcalib_src.exists():
                    continue    
                mkdir_cmd = ["ssh", REMOTE, f"mkdir -p {dest_dir}"]
                print("Ensuring remote dir:", " ".join(mkdir_cmd))
                subprocess.run(mkdir_cmd, check=True)
                
                items_to_copy = [
                    pvcalib_src / "Geometries",
                    pvcalib_src / f"{sample_name}_EDPVR_calibrated_shifted.csv",
                    pvcalib_src / "ordered_calibrated_pv_data.csv",
                ]
                
                for item in items_to_copy:
                    if not item.exists():
                        print(f"Skipping missing: {item}")
                        continue
                    cmd = ["rsync", "-avh", "--progress", str(item), f"{remote_dest}/{pv_folder}/"]
                    print("Running:", " ".join(cmd))
                    subprocess.run(cmd, check=True)
                
            if args.folders is None:
                print("Please specify folders to export, if any, using --folders")
                continue

            for folder in args.folders:
                folder_path = sample_dir / folder
                if not folder_path.exists():
                    print(f"Skipping missing folder: {folder_path}")
                    continue
                cmd = [
                    "rsync", "-avh", "--progress",
                    str(folder_path) + "/",  # Trailing slash to copy contents
                    f"{remote_dest}/{folder}/"
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
