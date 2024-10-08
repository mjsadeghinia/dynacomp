import argparse
import subprocess

def run_commands(n, m, o, c):
    # Preprocessing
    preprocess_cmd = f'python3 preprocessing.py -n "{n}" -m "{m}" -o "{o}"'
    subprocess.run(preprocess_cmd, shell=True)

    # Unloading 
    unload_cmd = f'python3 unloading.py -n "{n}" -o "{o}"'
    subprocess.run(unload_cmd, shell=True)

    # Processing
    process_cmd = f'mpirun -n {c} python3 processing.py -n "{n}" -o "{o}"'
    subprocess.run(process_cmd, shell=True)


def main() -> int:
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        default="156_1",
        type=str,
        help="The sample file name to be p",
    )
    valid_mesh_qualities = ['coarse', 'fine']
    parser.add_argument(
        "-m",
        "--mesh_quality",
        default='coarse',
        choices=valid_mesh_qualities,
        type=str,
        help="The mesh quality. Settings will be loaded accordingly from json file",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        default= "output",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )
    
    parser.add_argument(
        "-c",
        "--cpus_num",
        default= 4,
        type=int,
        help="The number cpu processors for the mpirun in processing.",
    )


    args = parser.parse_args()
    # Run the commands with the provided arguments
    run_commands(args.name, args.mesh_quality, args.output_folder, args.cpus_num)
    
if __name__ == "__main__":
    main()
