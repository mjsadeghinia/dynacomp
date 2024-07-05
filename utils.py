from structlog import get_logger


logger = get_logger()


def get_mask_settings(mask_settings, sample_name=None):
    # Use provided mask_settings or default ones if not provided
    default_mask_settings = get_default_mask_settings(sample_name)
    mask_settings = (
        {
            key: mask_settings.get(key, default_mask_settings[key])
            for key in default_mask_settings
        }
        if mask_settings
        else default_mask_settings
    )
    return mask_settings


def get_default_mask_settings(sample_name):
    """
    Default mask settings parameters
    """
    if sample_name is None or sample_name == "156_1":
        default_mask_settings = dict(
            slice_number=6, num_itr_slice_1=0, num_itr_slice_2=1
        )
    elif sample_name == "OP154_M3":
        default_mask_settings = dict(
            slice_number=3, num_itr_slice_1=1, num_itr_slice_2=2
            )
    else:
        logger.error(f"No default mask setting is defined for {sample_name}")
    return default_mask_settings


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
                'num_z_sections_endo': 18,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
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
                'MeshSizeMax': 1
            }
        },
        'OP154_M3': {
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
                'num_z_sections_endo': 18,
                'num_mid_layers_base': 3,
                'smooth_level_epi': 0.1,
                'smooth_level_endo': 0.15,
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
                'MeshSizeMax': 1
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