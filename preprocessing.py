from pathlib import Path
from structlog import get_logger

from mesh_utils import compile_h5, pre_process_mask
from meshing import create_mesh
from create_geometry import create_geometry
from unloading import unloader


logger = get_logger()


# %%
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
    else:
        logger.error(f"No default mask setting is defined for {sample_name}")
    return default_mask_settings


def get_mesh_settings(mesh_settings, sample_name=None):
    # Use provided mask_settings or default ones if not provided
    default_mesh_settings = get_default_mesh_settings(sample_name)
    mesh_settings = (
        {
            key: mesh_settings.get(key, default_mesh_settings[key])
            for key in default_mesh_settings
        }
        if mesh_settings
        else default_mesh_settings
    )
    return mesh_settings


def get_default_mesh_settings(sample_name):
    """
    Default mask settings parameters
    """
    if sample_name is None or sample_name == "156_1":
        default_mask_settings = dict(
            seed_num_base_epi=15,
            seed_num_base_endo=10,
            num_z_sections_epi=10,
            num_z_sections_endo=9,
            num_mid_layers_base=1,
            smooth_level_epi=0.1,
            smooth_level_endo=0.15,
            num_lax_points=16,
            lax_smooth_level_epi=1,
            lax_smooth_level_endo=1.5,
            z_sections_flag_epi=0,
            z_sections_flag_endo=1,
            seed_num_threshold_epi=8,
            seed_num_threshold_endo=8,
            scale_for_delauny = 1.2,
            t_mesh=-1,
        )
    else:
        logger.error(f"No default mask setting is defined for {sample_name}")
    return default_mask_settings


# def main(
#     directory_path,
#     ED_pressure=0.24,
#     mesh_settings=None,
#     mask_settings=None,
#     sample_name=None,
#     h5_overwrite=True,
# ):
# main(directory_path)

# directory_path,
# %%
atrium_pressure = 1.699
mesh_settings = dict(
    seed_num_base_epi=60,
    seed_num_base_endo=40,
    num_z_sections_epi=25,
    num_z_sections_endo=24,
    num_mid_layers_base=5,
    num_lax_points=32,
    seed_num_threshold_epi=20,
    seed_num_threshold_endo=20,
    z_sections_flag_epi=1,
    z_sections_flag_endo=1,
    t_mesh=-1,
)
mesh_settings = dict(
    scale_for_delauny = 3,
    t_mesh=0,
)
mask_settings = None
sample_name = None
h5_overwrite = True

directory_path = Path("00_data/AS/3week/156_1/")
results_folder = "00_Results_LQ_ES"

directory_path = Path(directory_path)
h5_file = compile_h5(directory_path, overwrite=h5_overwrite)
pre_process_mask_settings = get_mask_settings(mask_settings, sample_name)
h5_file = pre_process_mask(h5_file, save_flag=True, settings=pre_process_mask_settings, results_folder=results_folder)
mesh_settings = get_mesh_settings(mesh_settings, sample_name)
LVMesh, meshdir = create_mesh(
    directory_path,
    mesh_settings,
    h5_file,
    plot_flag=True,
    results_folder=results_folder,
)
geometry = create_geometry(meshdir, fiber_angles=None, mesh_fname=None, plot_flag=True)
# Saving the Geometries
if results_folder is not None or not results_folder == "":
    results_folder_dir = directory_path / results_folder
    results_folder_dir.mkdir(exist_ok=True)
else:
    results_folder_dir = directory_path
outdir = results_folder_dir / "Geometry"
fname = outdir / "geometry"
geometry.save(fname.as_posix(), overwrite_file=True)
#%%
unloaded_geometry = unloader(meshdir, atrium_pressure=atrium_pressure, plot_flag=True)
fname = outdir / "unloaded_geometry"
unloaded_geometry.save(fname.as_posix(), overwrite_file=True)

# %%
