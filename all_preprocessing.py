from pathlib import Path
import subprocess
from structlog import get_logger

# Set up the logger
logger = get_logger()

# Define the directory containing the files
settings_dir = Path("/home/shared/dynacomp/settings")

# List to keep track of files that failed to process
failed_files = []

# Loop through the first five files in the directory
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
        
        try:
            # Run the command
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            # Log the error with structlog and highlight the sample name
            logger.error(f"Error processing file {name}, due to {e}")
            failed_files.append(name)

# At the end, list all the files that did not process correctly
if failed_files:
    logger.error(f"The following files failed to process {failed_files}")
    logger.error("Some files failed to process. Check the logs for details.")
else:
    logger.info("All files processed successfully.")