import json
from pathlib import Path
from structlog import get_logger

logger = get_logger()

def add_or_update_field(json_file_path, section, field_name, field_values, overwrite=False):
    """
    Add or update a field with values in a specific section of the JSON file.

    :param json_file_path: Path to the JSON file
    :param section: Section of the JSON where the field should be added or updated (e.g., "mesh")
    :param field_name: The name of the field to add or update (e.g., "very_fine")
    :param field_values: A dictionary of values to add or update for the field
    :param overwrite: If True, overwrite the entire field. If False, update keys that exist in field_values
    """
    json_file_path = Path(json_file_path)

    try:
        with json_file_path.open('r') as f:
            data = json.load(f)
        logger.info("Loaded JSON file", path=str(json_file_path))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to load JSON file", path=str(json_file_path), error=str(e))
        return
    
    # Ensure the section exists in the JSON
    if section not in data:
        logger.warning(f'Section "{section}" not found in JSON. Creating new section.', section=section)
        data[section] = {}

    # Handle overwrite or update keys
    if overwrite or field_name not in data[section]:
        # Overwrite the entire field or add the field if it doesn't exist
        data[section][field_name] = field_values
    else:
        # Update only the keys that exist in field_values
        data[section][field_name].update(field_values)
    
    # Save the updated JSON data back to the file
    try:
        with json_file_path.open('w') as f:
            json.dump(data, f, indent=4)
        logger.info("Successfully updated JSON file", path=str(json_file_path), field_name=field_name)
    except Exception as e:
        logger.error("Failed to write to JSON file", path=str(json_file_path), error=str(e))


def process_all_json_files(directory, section, field_name, field_values):
    """
    Process all JSON files in a directory, adding or updating the specified field.

    :param directory: The directory containing the JSON files
    :param section: Section of the JSON where the field should be added or updated (e.g., "mesh")
    :param field_name: The name of the field to add or update (e.g., "very_fine")
    :param field_values: A dictionary of values to add or update for the field
    """
    directory_path = Path(directory)
    
    if directory_path.is_file():
        json_file_path = directory_path
        logger.info("Processing file", filename=json_file_path.name)
        add_or_update_field(json_file_path, section, field_name, field_values)
        return
        
    if not directory_path.is_dir():
        logger.error("Directory not found", directory=str(directory_path))
        return
    
    logger.info("Processing JSON files in directory", directory=str(directory_path))
    
    # Iterate over all JSON files in the directory
    for json_file_path in directory_path.glob("*.json"):
        logger.info("Processing file", filename=json_file_path.name)
        add_or_update_field(json_file_path, section, field_name, field_values)

updated_field_values_very_fine = {
    "seed_num_base_epi": 100,
    "seed_num_base_endo": 100,
    "num_z_sections_epi": 50,
    "num_z_sections_endo": 50,
    "num_mid_layers_base": 5,
    "smooth_level_epi": 0.01,
    "smooth_level_endo": 0.01,
    "num_lax_points": 128,
    "lax_smooth_level_epi": 0.2,
    "lax_smooth_level_endo": 0.2,
    "z_sections_flag_epi": 1,
    "z_sections_flag_endo": 1,
    "seed_num_threshold_epi": 25,
    "seed_num_threshold_endo": 20,
    "scale_for_delauny": 1.5,
    "t_mesh": -1,
    "MeshSizeMin": 0.1,
    "MeshSizeMax": 0.3,
    "SurfaceMeshSizeEndo": .5,
    "SurfaceMeshSizeEpi": .5,
}


updated_field_values_fine = {
    "seed_num_base_epi": 100,
    "seed_num_base_endo": 100,
    "num_z_sections_epi": 50,
    "num_z_sections_endo": 50,
    "num_mid_layers_base": 7,
    "smooth_level_epi": 0.01,
    "smooth_level_endo": 0.01,
    "num_lax_points": 64,
    "lax_smooth_level_epi": 0.2,
    "lax_smooth_level_endo": 0.2,
    "z_sections_flag_epi": 1,
    "z_sections_flag_endo": 1,
    "seed_num_threshold_epi": 25,
    "seed_num_threshold_endo": 20,
    "scale_for_delauny": 1.5,
    "t_mesh": -1,
    "MeshSizeMin": 0.2,
    "MeshSizeMax": 0.5,
    "SurfaceMeshSizeEndo": 1,
    "SurfaceMeshSizeEpi": 1,
}


# Call the function to process all JSON files in the directory
process_all_json_files('/home/shared/dynacomp/settings/100_1.json', 'mesh', 'fine', updated_field_values_fine)
# add_or_update_field('/home/shared/dynacomp/settings/100_1.json', 'mesh', 'very_fine', updated_field_values)
