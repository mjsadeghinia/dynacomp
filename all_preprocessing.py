from pathlib import Path
import subprocess

settings_dir = Path("/home/shared/dynacomp/settings")

for idx, file_path in enumerate(settings_dir.iterdir()):
    # Process only the first five files
    if idx >= 5:
        break
    
    # Ensure that it's a file (skip directories)
    if file_path.is_file():
        # Extract the file name without extension
        name = file_path.stem
        # Construct the command
        command = f'python3 preprocessing.py -o "test" -n "{name}"'
        # Run the command
        subprocess.run(command, shell=True)
