from pathlib import Path
from structlog import get_logger

import utils
from mesh_utils import compile_h5, pre_process_mask
from meshing import create_mesh
from create_geometry import create_geometry

logger = get_logger()

# %%
directory_path = Path("00_data/SHAM/3week/OP154_M3")
results_folder = "00_Results"
atrium_pressure = 1
h5_overwrite = True
sample_name = 'OP154_M3'
mesh_quality='fine'
mask_settings = None
mesh_settings = None
fiber_angles = None

directory_path = Path(directory_path)
h5_file = compile_h5(directory_path, overwrite=h5_overwrite)
pre_process_mask_settings = utils.get_mask_settings(mask_settings, sample_name)
h5_file = pre_process_mask(
    h5_file,
    save_flag=True,
    settings=pre_process_mask_settings,
    results_folder=results_folder,
)
mesh_settings = utils.get_mesh_settings(mesh_settings, sample_name=sample_name, mesh_quality=mesh_quality)
LVMesh, meshdir = create_mesh(
    directory_path,
    mesh_settings,
    h5_file,
    plot_flag=True,
    results_folder=results_folder,
)
geometry = create_geometry(
    meshdir, fiber_angles=fiber_angles, mesh_fname=None, plot_flag=True
)
# Saving the Geometries
if results_folder is not None or not results_folder == "":
    results_folder_dir = directory_path / results_folder
    results_folder_dir.mkdir(exist_ok=True)
else:
    results_folder_dir = directory_path
outdir = results_folder_dir / "Geometry"
fname = outdir / "geometry"
geometry.save(fname.as_posix(), overwrite_file=True)
