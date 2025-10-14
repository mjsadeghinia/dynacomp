import json
from pathlib import Path
import numpy as np

from structlog import get_logger

logger = get_logger()

def load_settings(settings_dir: Path, sample_num: int) -> dict:
    files = sorted([f for f in settings_dir.iterdir() if f.is_file() and f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)
    
    
def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")

def load_pressure_volumes(pv_dir):
    PV_data_fname = pv_dir / f"ordered_calibrated_pv_data.csv"
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    pressures = PV_data[:, 1] * mmHg_to_kPa
    volumes = PV_data[:, 2]
    return pressures, volumes

def read_edpvr_data(edpvr_dir):
    fname = edpvr_dir / "inflation_results.txt"
    if not fname.exists():
        logger.error(FileNotFoundError(f"EDPVR data file {fname} does not exist."))
        return None
    edpvr_data = np.loadtxt(fname, delimiter=',', skiprows=1)
    return edpvr_data

def read_fiber_data(fiber_dir):
    fname = fiber_dir / "Fiber_results.csv"
    if not fname.exists():
        logger.error(FileNotFoundError(f"Fiber data file {fname} does not exist."))
        return None
    fiber_data = np.loadtxt(fname, delimiter=',', skiprows=1)
    return fiber_data

def get_folders_from_edpvr_data(edpvr_data):
    edpvr_data_sorted = edpvr_data[edpvr_data[:, -1].argsort()]
    folders = []
    for row in edpvr_data_sorted:
        folder_name = f"a_{row[0]}_af_{row[1]}_bf_{row[3]}"
        folders.append(folder_name)
    return folders

def save_settings(settings, settings_dir, sample_name):
    """
    Save the updated settings dictionary to a JSON file.
    """
    settings_fname = settings_dir / f"{sample_name[2:]}.json"
    with open(settings_fname, "w") as file:
        json.dump(settings, file, indent=4)
    return settings_fname
