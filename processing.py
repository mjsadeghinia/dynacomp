# %%
import numpy as np
from pathlib import Path
import shutil
from structlog import get_logger

import dolfin
import pulse
from heart_model import HeartModelDynaComp
from datacollector import DataCollector
from coupling_solver import newton_solver
logger = get_logger()

comm = dolfin.MPI.comm_world

# %%

# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03

directory_path = Path("00_data/AS/3week/156_1/")
outdir = directory_path / "00_Modeling"
mesh_outdir = directory_path / "Geometry"
mesh_fname = mesh_outdir / "unloaded_geometry"
bc_params = {"pericardium_spring": 10}
# delet files excepts the
if outdir.is_dir() and comm.Get_rank() == 1:
    shutil.rmtree(outdir)
    outdir.mkdir(exist_ok=True)
comm.Barrier()

#%%
unloaded_geometry = pulse.HeartGeometry.from_file(mesh_fname.as_posix() + ".h5", comm=comm)
heart_model = HeartModelDynaComp(geo=unloaded_geometry, bc_params=bc_params, comm=comm)
collector = DataCollector(outdir=outdir,problem=heart_model)
# Initializing the model
v = heart_model.compute_volume(activation_value=0, pressure_value=0)
collector.collect(
    time=0,
    pressure=0,
    volume=v,
    target_volume=v,
    activation=0.0,
)

# %% Loading PV Data
fname = directory_path / "PV data/PV_data.csv"
PV_data = np.loadtxt(fname.as_posix() ,delimiter=',')
# Converting mmHg to kPa
mmHg_to_kPa = 0.133322
pressures = PV_data[0, :] * mmHg_to_kPa

# Pressurizing up to End Diastole
v = heart_model.compute_volume(activation_value=0, pressure_value=pressures[0])
collector.collect(
    time=1,
    pressure=pressures[0],
    volume=v,
    target_volume=v,
    activation=0.0,
)

# Converting RVU to micro liter based on calculated EDV
RVU_to_microL = v/PV_data[1, 0]
volumes = PV_data[1,:]*RVU_to_microL
#%%
collector = newton_solver(
    heart_model = heart_model,
    pres = pressures[1:-5],
    vols = volumes[1:-5],
    collector = collector,
    start_time = 2,
    comm=comm
)
#%%