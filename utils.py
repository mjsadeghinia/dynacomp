from structlog import get_logger


logger = get_logger()




def get_mask_settings(sample_name):
    """
    Default mask settings parameters
    """
    settings = {
        'OP130_2': [2,2,2,2,2,2,1,1,1],
        '156_1': [0,0,1,1,1,1,1,1],
        '138_1': [3,3,3,3,3,3,1,1,1,1],
        '129_1': [0,2,2,1,1,1,1,2,1,1],
    }
           
    # Check if sample_name is valid
    if sample_name not in settings:
        logger.error(f"No default mask setting is defined for {sample_name}")

    # Retrieve settings based on sample_name and mesh_quality
    return settings[sample_name]


def get_mesh_settings(mesh_settings=None, sample_name='156_1', mesh_quality='fine'):
    # Get default settings based on sample_name and mesh_quality
    default_mesh_settings = get_default_mesh_settings(sample_name, mesh_quality)

    # If mesh_settings is provided, override the default settings
    if mesh_settings is not None:
        return {key: mesh_settings.get(key, default_mesh_settings[key]) for key in default_mesh_settings}
    else:
        return default_mesh_settings


def get_default_mesh_settings(sample_name, mesh_quality):
    """
    Default mask settings parameters based on sample name and mesh quality.
    """
    settings = {
        '156_1': {
            'coarse': {
                'seed_num_base_epi': 15,
                'seed_num_base_endo': 10,
                'num_z_sections_epi': 10,
                'num_z_sections_endo': 9,
                'num_mid_layers_base': 1,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 0,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 8,
                'scale_for_delauny': 1.2,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': None
            },
            'fine': {
                'seed_num_base_epi': 30,
                'seed_num_base_endo': 26,
                'num_z_sections_epi': 15,
                'num_z_sections_endo': 19,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.08,
                'smooth_level_endo': 0.11,
                'num_lax_points': 32,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 15,
                'scale_for_delauny': 1.2,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': .5
            }
        },
        'OP154_M3': {
            'fine': {
                'seed_num_base_epi': 30,
                'seed_num_base_endo': 26,
                'num_z_sections_epi': 10,
                'num_z_sections_endo': 8,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 15,
                'scale_for_delauny': 1.2,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': 1
            }
        },
        'OP130_2': {
            'coarse': {
                'seed_num_base_epi': 15,
                'seed_num_base_endo': 10,
                'num_z_sections_epi': 10,
                'num_z_sections_endo': 9,
                'num_mid_layers_base': 1,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 0,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 8,
                'scale_for_delauny': 1.5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': None
            },
            'fine': {
                'seed_num_base_epi': 45,
                'seed_num_base_endo': 30,
                'num_z_sections_epi': 20,
                'num_z_sections_endo': 18,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 32,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 2.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 0,
                'seed_num_threshold_epi': 18,
                'seed_num_threshold_endo': 12,
                'scale_for_delauny': 1.5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': .75
            }
        },
        '138_1': {
            'coarse': {
                'seed_num_base_epi': 24,
                'seed_num_base_endo': 15,
                'num_z_sections_epi': 12,
                'num_z_sections_endo': 11,
                'num_mid_layers_base': 1,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 8,
                'scale_for_delauny': .5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': 1.25
            },
            'fine': {
                'seed_num_base_epi': 40,
                'seed_num_base_endo': 25,
                'num_z_sections_epi': 20,
                'num_z_sections_endo': 18,
                'num_mid_layers_base': 2,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 1,
                'seed_num_threshold_epi': 20,
                'seed_num_threshold_endo': 15,
                'scale_for_delauny': .5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': .75
            },
            'very_fine': {
                'seed_num_base_epi': 70,
                'seed_num_base_endo': 40,
                'num_z_sections_epi': 45,
                'num_z_sections_endo': 20,
                'num_mid_layers_base': 5,
                'smooth_level_epi': 0.05,
                'smooth_level_endo': 0.075,
                'num_lax_points': 32,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 0,
                'z_sections_flag_endo': 0,
                'seed_num_threshold_epi': 20,
                'seed_num_threshold_endo': 15,
                'scale_for_delauny': 4.5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': .25
            }
        },
        '129_1': {
            'coarse': {
                'seed_num_base_epi': 15,
                'seed_num_base_endo': 10,
                'num_z_sections_epi': 10,
                'num_z_sections_endo': 13,
                'num_mid_layers_base': 1,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 0,
                'z_sections_flag_endo': 0,
                'seed_num_threshold_epi': 8,
                'seed_num_threshold_endo': 8,
                'scale_for_delauny': 4.5,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': 1.5
            },
            'fine': {
                'seed_num_base_epi': 45,
                'seed_num_base_endo': 15,
                'num_z_sections_epi': 22,
                'num_z_sections_endo': 20,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.07,
                'smooth_level_endo': 0.07,
                'num_lax_points': 16,
                'lax_smooth_level_epi': 1,
                'lax_smooth_level_endo': 1.5,
                'z_sections_flag_epi': 1,
                'z_sections_flag_endo': 0,
                'seed_num_threshold_epi': 35,
                'seed_num_threshold_endo': 12,
                'scale_for_delauny': 1.25,
                't_mesh': -1,
                'MeshSizeMin': None,
                'MeshSizeMax': .75
            }
        }
    }

    # Check if sample_name is valid
    if sample_name not in settings:
        logger.error(f"No default mask setting is defined for {sample_name}")

    # Check if mesh_quality is valid
    if mesh_quality not in settings[sample_name]:
        logger.error(f"Invalid mesh quality '{mesh_quality}' for {sample_name}, it should be either 'coarse' or 'fine'")

    # Retrieve settings based on sample_name and mesh_quality
    return settings[sample_name][mesh_quality]