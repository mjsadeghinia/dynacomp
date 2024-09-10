from pathlib import Path
from structlog import get_logger

import utils
from mesh_utils import compile_h5, pre_process_mask, shift_slice_mask, close_apex, repair_slice, remove_slice
from meshing import create_mesh
from create_geometry import create_geometry

logger = get_logger()

# %%
sample_name = '156_1'
mesh_quality='coarse'
results_folder = "00_Results_test_" + mesh_quality
h5_overwrite = True
setting_dir = "/home/shared/dynacomp/settings"
#%%
setting_dir = Path(setting_dir)
settings_fname = setting_dir / f"{sample_name}.json"
settings = utils.load_settings(settings_fname)
directory_path = Path(settings["path"])
mask_settings = settings["mask"]
mesh_settings = settings["mesh"][mesh_quality]
fiber_angles = None

h5_file = compile_h5(directory_path, overwrite=h5_overwrite)  
h5_file = pre_process_mask(
    h5_file,
    save_flag=True,
    settings=mask_settings,
    results_folder=results_folder,
)
if settings["remove_slice"]:
    h5_file = remove_slice(h5_file, slice_num=0, save_flag=True, results_folder=results_folder)  
    
if settings["shift_slice_mask"]:
    slice_num = 2
    slice_num_ref = 1
    h5_file = shift_slice_mask(h5_file,slice_num,slice_num_ref,save_flag = True,results_folder=results_folder)    

if settings["close_apex"]:
    h5_file = close_apex(h5_file, itr=2, itr_dilation = 3 ,save_flag = True,results_folder=results_folder)    

#%%
LVMesh, meshdir = create_mesh(
    directory_path,
    mesh_settings,
    h5_file,
    plot_flag=True,
    results_folder=results_folder,
)
#%%
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

# %%