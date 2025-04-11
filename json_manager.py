import json
from pathlib import Path
from structlog import get_logger

logger = get_logger()

def add_or_update_field(json_file_path, section, field_name, field_values, overwrite=False):
    """
    Add or update a field with values in a specific section of the JSON file.

    :param json_file_path: Path to the JSON file
    :param section: Section of the JSON where the field or values should be added or updated (e.g., "mesh")
    :param field_name: The name of the field to add or update (e.g., "very_fine"). If empty, updates the section itself.
    :param field_values: A dictionary of values to add or update for the field or section
    :param overwrite: If True, overwrite the entire field or section. If False, update keys that exist in field_values
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

    if field_name == "":
        # If field_name is empty, update the entire section
        if overwrite:
            data[section] = field_values  # Overwrite the entire section
        else:
            # Update only the keys that exist in field_values
            data[section].update(field_values)
    else:
        # Handle field update or overwrite
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
        logger.info("Successfully updated JSON file", path=str(json_file_path), field_name=field_name or "entire section")
    except Exception as e:
        logger.error("Failed to write to JSON file", path=str(json_file_path), error=str(e))


def process_all_json_files(directory, section, field_name, field_values, overwrite=False):
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
        add_or_update_field(json_file_path, section, field_name, field_values, overwrite=overwrite)
        return
        
    if not directory_path.is_dir():
        logger.error("Directory not found", directory=str(directory_path))
        return
    
    logger.info("Processing JSON files in directory", directory=str(directory_path))
    
    # Iterate over all JSON files in the directory
    for json_file_path in directory_path.glob("*.json"):
        logger.info("Processing file", filename=json_file_path.name)
        add_or_update_field(json_file_path, section, field_name, field_values, overwrite=overwrite)
        
import json
from pathlib import Path
from structlog import get_logger

logger = get_logger()

def copy_coarse_from_fine(directory):
    """
    Process all JSON files in a directory. For each file, copy the fields from mesh['fine']
    into mesh['coarse'], then update mesh['coarse']['MeshSizeMin'] = 0.5 and
    mesh['coarse']['MeshSizeMax'] = 1. Save the result back to the JSON file.
    
    :param directory: The directory containing the JSON files or a single JSON file path.
    """
    directory_path = Path(directory)

    # If the path is a single file, process just that file
    if directory_path.is_file():
        _process_single_file_coarse_from_fine(directory_path)
        return

    if not directory_path.is_dir():
        logger.error("Directory not found", directory=str(directory_path))
        return

    logger.info("Processing JSON files in directory", directory=str(directory_path))

    # Iterate over all JSON files in the directory
    for json_file_path in directory_path.glob("*.json"):
        logger.info("Processing file", filename=json_file_path.name)
        _process_single_file_coarse_from_fine(json_file_path)


def _process_single_file_coarse_from_fine(json_file_path: Path):
    """
    Helper function to load a single file, copy mesh['fine'] into mesh['coarse'],
    then update mesh['coarse']['MeshSizeMin'] and mesh['coarse']['MeshSizeMax'].
    """
    try:
        with json_file_path.open('r') as f:
            data = json.load(f)
        logger.info("Loaded JSON file", path=str(json_file_path))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to load JSON file", path=str(json_file_path), error=str(e))
        return

    # Safely navigate to data["mesh"]["fine"]
    mesh_section = data.get("mesh", {})
    if "fine" not in mesh_section:
        logger.warning(
            'No "fine" section found in mesh. Skipping file.',
            filename=json_file_path.name
        )
        return

    # If "coarse" does not exist, create it
    if "coarse" not in mesh_section:
        mesh_section["coarse"] = {}

    # Copy all fields from mesh["fine"] to mesh["coarse"]
    # Note that `.update()` modifies in-place. 
    # If you want an exact copy (replacing entire 'coarse'), use:
    # mesh_section["coarse"] = copy.deepcopy(mesh_section["fine"])
    # For now, let's assume we just want to overwrite with the "fine" values.
    mesh_section["coarse"].update(mesh_section["fine"])

    # Finally, update the specific fields in mesh["coarse"]
    mesh_section["coarse"]["MeshSizeMin"] = 0.5
    mesh_section["coarse"]["MeshSizeMax"] = 1

    # Assign back to data in case mesh didn't exist initially
    data["mesh"] = mesh_section

    # Save the updated data back to the file
    try:
        with json_file_path.open('w') as f:
            json.dump(data, f, indent=4)
        logger.info(
            "Successfully updated mesh['coarse'] from mesh['fine']",
            path=str(json_file_path)
        )
    except Exception as e:
        logger.error("Failed to write to JSON file", path=str(json_file_path), error=str(e))

import json
from pathlib import Path
from structlog import get_logger

logger = get_logger()

def add_groups(directory):
    """
    Process all JSON files (or a single JSON file) in 'directory'. 
    For each file:
      - Reads data["path"]
      - Determines 'group' (AS or SHAM)
      - Extracts 'id' (the last part of path)
      - Extracts 'time' (integer before 'weeks')
      - Extracts 'ring_diameter' if group is AS; otherwise None
      - Writes these fields back to the JSON file.
    """

    directory_path = Path(directory)

    # If the path is a single file, process it
    if directory_path.is_file():
        _process_single_file_add_groups(directory_path)
        return

    if not directory_path.is_dir():
        logger.error("Directory not found", directory=str(directory_path))
        return

    logger.info("Processing JSON files in directory", directory=str(directory_path))

    # Iterate over all JSON files in the directory
    for json_file_path in directory_path.glob("*.json"):
        logger.info("Processing file", filename=json_file_path.name)
        _process_single_file_add_groups(json_file_path)


def _process_single_file_add_groups(json_file_path: Path):
    """
    Helper function:
      - Load the JSON
      - Parse data["path"]
      - Add data["group"], data["id"], data["time"], data["ring_diameter"] 
      - Save back to disk
    """
    try:
        with json_file_path.open('r') as f:
            data = json.load(f)
        logger.info("Loaded JSON file", path=str(json_file_path))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Failed to load JSON file", path=str(json_file_path), error=str(e))
        return

    # If 'path' doesn't exist or isn't a string, we skip this file
    path_str = data.get("path")
    if not path_str or not isinstance(path_str, str):
        logger.warning("'path' not found or invalid in JSON file", filename=json_file_path.name)
        return

    # Determine group
    # If the string "SHAM" is in the path, group = SHAM, else AS
    if "SHAM" in path_str.upper():  # Upper-case check handles "Sham"/"SHAM" etc.
        group = "SHAM"
    else:
        group = "AS"

    # Extract ID -> the last component after "/"
    # e.g. OP126_1 or OP100_1
    file_id = path_str.strip("/").split("/")[-1]

    # Extract time -> number before "weeks"
    # For example, "6weeks" -> time=6 or "20weeks" -> time=20
    # We'll scan for the pattern "<number>weeks" anywhere in the path.
    time_val = None
    import re
    match_time = re.search(r'(\d+)weeks', path_str)
    if match_time:
        time_val = int(match_time.group(1))

    # Extract ring_diameter
    # For AS, we want the number right before the last component (file_id).
    # Example for AS: /AS/6weeks/130/OP126_1 -> ring_diameter = 130
    # For SHAM, it's None
    ring_diameter = None
    if group == "AS" and time_val is not None:
        # tokens: [..., 'AS', '6weeks', '130', 'OP126_1']
        # ring_diameter = tokens[-2] if the second last token is numeric
        tokens = path_str.strip("/").split("/")
        if len(tokens) >= 4:
            candidate = tokens[-2]  # e.g. "130"
            # We attempt to convert to float
            try:
                ring_diameter = float(candidate)
            except ValueError:
                # If it fails, ring_diameter remains None
                ring_diameter = None

    # Update the data
    data["group"] = group
    data["id"] = file_id
    data["time"] = time_val  # might be None if not found
    data["ring_diameter"] = ring_diameter

    # Save the updated data back to the file
    try:
        with json_file_path.open('w') as f:
            json.dump(data, f, indent=4)
        logger.info("Successfully updated file with new fields", path=str(json_file_path))
    except Exception as e:
        logger.error("Failed to write to JSON file", path=str(json_file_path), error=str(e))


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

updated_field_values = {"volume_smooth_window_length": 15, "pressure_smooth_window_length": 15}
updated_field_values = {"skip_redundant_data_flag": False}
# Call the function to process all JSON files in the directory
# process_all_json_files('/home/shared/dynacomp/settings/100_1.json', 'mesh', 'fine', updated_field_values_fine)
# add_or_update_field('/home/shared/dynacomp/settings/128_1.json', 'PV', '', updated_field_values)
#process_all_json_files('/home/shared/dynacomp/settings/', 'PV', '', updated_field_values)
# copy_coarse_from_fine('/home/shared/dynacomp/settings/')
# add_groups('/home/shared/dynacomp/settings/')
updated_field_values = {"process_occlusion_flag": True,
                        "Occlusion_data_index_i": 0,
                        "Occlusion_data_index_f": -1,
                        "Occlusion_recording_num": None,
                        "Occlusion_data_skip_index": 1}
process_all_json_files('/home/shared/dynacomp/settings/', 'PV', '', updated_field_values, overwrite=False)