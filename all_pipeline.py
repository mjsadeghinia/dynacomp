from pathlib import Path
from structlog import get_logger
from pipeline import meshing_unloading_analysis 

logger = get_logger()


def get_sample_name(sample_num, setting_dir):
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in setting_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    sample_name = sorted_files[sample_num - 1].with_suffix("").name
    return sample_name

number_i = 1
number_f = 57
mesh_quality = "coarse"
output_folder = "coarse_mesh_stiffUnloading_20percent"
cpus_num = 8
setting_dir = Path("/home/shared/dynacomp/settings")
sample_with_error = []

for sample_num in range(number_i, number_f + 1):
    try:
        meshing_unloading_analysis(sample_num, mesh_quality, output_folder, cpus_num)
    except Exception as e:
        sample_name = get_sample_name(sample_num, setting_dir)
        logger.error(f"Sample {sample_name} has not been analyzed: {e}")
        sample_with_error.append(sample_name)


logger.warning(f"Samples with error: {sample_with_error} ")
