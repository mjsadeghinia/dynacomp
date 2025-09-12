import argparse
from pathlib import Path
import shutil

import utils

from structlog import get_logger

logger = get_logger()

#%%
def main():
    parser = argparse.ArgumentParser(
        description="2D parameter sweep of (a, a_f) for HeartModelDynaComp"
    )
    parser.add_argument(
        '-n', '--number',
        nargs='*',
        type=int,
        default=None,
        help='Sample number(s) to process. If omitted, all samples will be processed.'
    )
    parser.add_argument(
        "-i",
        "--sample_ID",
        nargs="+",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )
    parser.add_argument(
        '--settings_dir',
        type=Path,
        default=Path('/home/shared/dynacomp/settings'),
        help='Directory where JSON settings files are stored.'
    )
    parser.add_argument(
        '-s', '--scan_type',
        type=str,
        default='TPM',
        help='Scan type; subdirectories will be named accordingly.'
    )
    parser.add_argument(
        '-r', '--results_dir',
        type=Path,
        default=Path('/home/shared/01_results_coarse_mesh'),
        help='Directory where results will be saved.'
    )

    args = parser.parse_args()

    settings_dir = args.settings_dir
    scan_type = args.scan_type
    results_dir = args.results_dir
    

    # Determine samples to process
    if args.sample_ID:
        sample_nums = []
        for id in args.sample_ID:
            id_num = utils.get_num_from_id(id, settings_dir)
            sample_nums.append(id_num)
    elif args.number:
        sample_nums = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(files) + 1))

    for sample in sample_nums:
        settings = utils.load_settings(settings_dir, sample)
        sample_id = settings['id']
        sample_dir = results_dir / sample_id / scan_type
        pv_dir = sample_dir / "01_PVCalibration"
        geo_dir = pv_dir / "Geometries"

        ex3_data_dir = sample_dir / "99_Ex3_Data"
        if not ex3_data_dir.exists():
            ex3_data_dir.mkdir(parents=True, exist_ok=True)
        ex3_pv_dir = ex3_data_dir / "01_PVCalibration"
        if not ex3_pv_dir.exists():
            ex3_pv_dir.mkdir(parents=True, exist_ok=True)
        ex3_geo_dir = ex3_pv_dir / "Geometries"
        if not ex3_geo_dir.exists():
            ex3_geo_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(geo_dir / "geometry_0.h5", ex3_geo_dir / "geometry_0.h5")
        shutil.copy(pv_dir / "ordered_calibrated_pv_data.csv", ex3_pv_dir / "ordered_calibrated_pv_data.csv")
        logger.info(f"Copied data for sample {sample_id} to {ex3_data_dir}")


#%%
if __name__ == "__main__":
    main()