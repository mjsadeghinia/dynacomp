from pathlib import Path
from structlog import get_logger

import utils
from mesh_utils import compile_h5, pre_process_mask, shift_slice_mask, close_apex, repair_slice, remove_slice
from meshing import create_mesh
from create_geometry import create_geometry

logger = get_logger()

# %%
paths = {
        'OP130_2': "00_data/SHAM/6week/OP130_2",
        '156_1':'00_data/AS/3week/156_1',
        '129_1':'00_data/AS/6week/129_1',
        '138_1':'00_data/AS/12week/138_1',
}
         
sample_name = '138_1'
results_folder = "00_Results_coarse_unloaded"
h5_overwrite = True
directory_path = Path(paths[sample_name])

mesh_quality='coarse'
mesh_settings = None
fiber_angles = None

directory_path = Path(directory_path)
h5_file = compile_h5(directory_path, overwrite=h5_overwrite)  
pre_process_mask_settings = utils.get_mask_settings(sample_name)
h5_file = pre_process_mask(
    h5_file,
    save_flag=True,
    settings=pre_process_mask_settings,
    results_folder=results_folder,
)
if sample_name in {'129_1'} :
    h5_file = remove_slice(h5_file, slice_num=0, save_flag=True, results_folder=results_folder)  
    
if sample_name in {'OP130_2'}:
    slice_num = 2
    slice_num_ref = 1
    h5_file = shift_slice_mask(h5_file,slice_num,slice_num_ref,save_flag = True,results_folder=results_folder)    

if sample_name in {'138_1', '129_1'} :
    h5_file = close_apex(h5_file, itr=2, itr_dilation = 3 ,save_flag = True,results_folder=results_folder)    

#%%
mesh_settings = utils.get_mesh_settings(mesh_settings, sample_name=sample_name, mesh_quality=mesh_quality)
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