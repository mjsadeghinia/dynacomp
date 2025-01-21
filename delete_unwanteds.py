from pathlib import Path
import json
import shutil

from structlog import get_logger

logger = get_logger()

# Specify the input and output folders
settings_folder = Path("/home/shared/dynacomp/settings")  # Replace with the path to your JSON files
keep_folders = ['PV Data', 't3']

# Loop through all JSON files in the input folder
for json_file in sorted(settings_folder.glob("*.json")):
    # Load the JSON data
    with json_file.open("r") as f:
        data = json.load(f)

    # Extract the path field
    file_path = Path(data.get("path", ""))
    logger.info(f"-------------------{file_path.stem}---------------")
    # If the JSON 'path' was found, and it exists as a directory, proceed
    if file_path.exists() and file_path.is_dir():
        # Iterate over all items in file_path
        for folder in file_path.iterdir():
            # Check if it's a directory
            if folder.is_dir():
                # If the folder name is not in keep_folders, remove it
                if folder.name not in keep_folders:
                    logger.info(f"Removing folder and contents: {folder}")
                    shutil.rmtree(folder)
    else:
        print(f"Warning: Path '{file_path}' does not exist or is not a directory.")
    
    logger.info(f"---------------------------------------------")
