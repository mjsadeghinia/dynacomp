from pathlib import Path
import subprocess
from structlog import get_logger
import argparse

# The following files failed for fine mesh to process: ['126_3', '127_1', '128_1', '129_3', '131_1', '132_1', '132_3', '133_1', '136_1', '137_2', '139_2', '140_2', '149_2', '151_1', '158_2', '163_2', '163_3', '164_3', '166_3', '167_1', '168_1', '169_3', '172_2', '183_1', '185_1', '185_2', '187_1']

def main(args=None) -> int:
    """
    Parse the command-line arguments and process files.
    """
    # Set up the logger
    logger = get_logger()

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Process JSON files for preprocessing.")
    
    parser.add_argument(
        "-o",
        "--outdir",
        default="/home/shared/dynacomp/00_data/CineData/",
        type=str,
        help="The full address to the output directory",
    )
    
    parser.add_argument(
        "-m",
        "--mesh_quality",
        type=str,
        help="The mesh quality to be used in the processing",
    )
    
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=str,
        help="Directory where the settings files (.json) are located",
    )
    
    parser.add_argument(
        "--skip_names",
        nargs='*',
        default=['138_3', '144_2', '150_2'],
        help="List of file names to skip during processing (without extensions)",
    )
    
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="Process only the nth file in the sorted list (1-based index).",
    )
    
    # Parse the command-line arguments
    args = parser.parse_args(args)

    # Define the directory containing the files
    settings_dir = Path(args.settings_dir)

    # List to keep track of files that failed to process
    failed_files = []

    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == '.json'])

    # If --number is specified, process only the nth file
    if args.number is not None:
        if 1 <= args.number <= len(sorted_files):
            sorted_files = [sorted_files[args.number - 1]]  # Get only the nth file (convert 1-based to 0-based index)
        else:
            logger.error(f"Invalid value for --number. Must be between 1 and {len(sorted_files)}.")
            return 1
    # Loop through the selected .json files in the sorted list
    for idx, file_path in enumerate(sorted_files):
        
        # Extract the file name without extension
        name = file_path.stem

        # Log the start of processing for this file
        logger.info(f"=======================================================")
        logger.info(f"=========== Start processing file {name} ===========")
        
        # Skip specified sample names
        if name in args.skip_names:
            logger.info(f"Skipping file {name} as it's in the skip list.")
            continue
        
        # Construct the command with the parsed arguments
        command = f'python3 preprocessing.py -o "{args.outdir}" -n "{name}" -m "{args.mesh_quality}"'
        
        try:
            # Run the command
            subprocess.run(command, shell=True, check=True)
            # Log the successful completion of the file
            logger.info(f"=========== Finished processing file {name} ===========")
            logger.info(f"=======================================================")
        except subprocess.CalledProcessError as e:
            # Log the error with structlog and highlight the sample name
            logger.error(f"Error processing file {name}, due to {e}")
            failed_files.append(name)

    # At the end, list all the files that did not process correctly
    if failed_files:
        logger.error(f"The following files failed to process: {failed_files}")
        logger.error("Some files failed to process. Check the logs for details.")
    else:
        logger.info("All files processed successfully.")
    
    return 0


if __name__ == "__main__":
    main()
