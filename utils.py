import json
from pathlib import Path


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