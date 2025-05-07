import argparse
import numpy as np
from pathlib import Path
import json

from inflator import run_inflator, load_edpvr


# %%
def load_settings(settings_dir: Path, sample_num: int) -> dict:
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], "r") as f:
        return json.load(f)


def calculate_error(pressures, volumes, out_dirs, settings):
    edpvr_pres, edpvr_vols_unc = load_edpvr(out_dirs["pv"])
    a = settings["PV"]["calibration"]["a"]
    b = settings["PV"]["calibration"]["b"]
    edpvr_vols = a * edpvr_vols_unc + b
    idx = np.argsort(edpvr_vols)
    edpvr_pres = edpvr_pres[idx]
    edpvr_vols = edpvr_vols[idx]

    interpolated_edpvr_pres = np.interp(volumes, edpvr_vols, edpvr_pres)

    error = np.sqrt((pressures - interpolated_edpvr_pres) ** 2)
    error = np.mean(error)
    return error


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--number",
        nargs="*",
        type=int,
        default=None,
        help="Sample number(s) to process. If omitted, all samples in settings_dir will be processed.",
    )
    parser.add_argument(
        "--settings_dir",
        type=Path,
        default=Path("/home/shared/dynacomp/settings"),
        help="Directory where JSON settings files are stored.",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        default=Path("/home/shared/00_data"),
        help="Directory where data files are stored.",
    )
    parser.add_argument(
        "-s", "--scan_type", type=str, default="TPM", help="Scan type; subdirectories will be named accordingly."
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        type=Path,
        default=Path("/home/shared/01_results_coarse_mesh"),
        help="Directory where results will be saved.",
    )
    args = parser.parse_args()

    settings_dir = args.settings_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    scan_type = args.scan_type

    # Determine sample list
    if args.number:
        sample_list = args.number
    else:
        settings_files = sorted([f for f in args.settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(settings_files) + 1))

    # Run inflator for each sample with explicit params
    for sample in sample_list:
        # Load settings
        settings = load_settings(settings_dir, sample)
        sample_id = settings["id"]
        # Prepare directories
        out_dirs = {
            "pv": results_dir / sample_id / "PV Data",
            "calib": results_dir / sample_id / scan_type / "01_PVCalibration",
            "unload": results_dir / sample_id / scan_type / "02_Unloading",
            "model": results_dir / sample_id / scan_type / "03_Modeling",
        }
        collector = run_inflator(sample, settings_dir=settings_dir, results_dir=results_dir, pressure_steps=5)
        error = calculate_error(collector.pressures, collector.volumes, out_dirs, settings)
        breakpoint()


if __name__ == "__main__":
    main()
