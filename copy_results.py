from pathlib import Path
import json
import re
import shutil
import argparse
from structlog import get_logger

logger = get_logger()


def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-r",
        "--results_folder",
        default="t3",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    args = parser.parse_args(args)
    setting_dir = Path(args.settings_dir)
    results_folder = args.results_folder

    # Specify the input and output folders
    input_folder = Path(setting_dir)  # Replace with the path to your JSON files
    output_directory = "/home/shared/00_results_" + results_folder
    output_directory = Path(output_directory)  # Replace with your desired output folder
    output_directory.mkdir(parents=True, exist_ok=True)  # Create the output folder if it doesn't exist

    # List to track missing source files
    missing_files = []

    # Loop through all JSON files in the input folder
    for json_file in input_folder.glob("*.json"):
        # Load the JSON data
        with json_file.open("r") as file:
            data = json.load(file)

        # Extract the path field
        file_path = data.get("path", "")

        # Extract AS, weeks_field, diameter, and the last part using regex
        match = re.search(r"/([^/]+)/([^/]+)/([^/]+)/(OP\d+_\d+)$", file_path)

        # Check if as_field is SHAM
        if "SHAM" in match.groups():
            _, as_field, weeks_field, name = match.groups()
            destination_file_name = f"{as_field}_{weeks_field}_{name}.png"
        else:
            as_field, weeks_field, diameter, name = match.groups()
            destination_file_name = f"{as_field}_{weeks_field}_{diameter}_{name}.png"

        # Construct the source file path
        source_file = Path(file_path) / f"{results_folder}" / "00_Modeling" / "results.png"

        # Check if the source file exists
        if not source_file.exists():
            logger.error(f"Source file {source_file} does not exist, skipping.")
            missing_files.append(destination_file_name)
            continue

        # Construct the destination file path
        destination_file = output_directory / destination_file_name

        # Copy the file to the destination
        shutil.copy(source_file, destination_file)
        logger.info(f"Copied {source_file} to {destination_file}")

    # Save missing files to a text file
    notfound_file = output_directory / "00_notfound.txt"
    with notfound_file.open("w") as file:
        file.write("\n".join(missing_files))

    logger.info(f"Missing files list saved to {notfound_file}")


if __name__ == "__main__":
    main()
