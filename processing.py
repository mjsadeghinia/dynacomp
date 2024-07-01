# %%
import numpy as np
from pathlib import Path
import shutil
from structlog import get_logger

import dolfin
import pulse
from heart_model import HeartModelDynaComp
from datacollector import DataCollector

logger = get_logger()


# %%

# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03

directory_path = Path("00_data/AS/3week/156_1/")
outdir = directory_path / "00_Modeling"
mesh_outdir = directory_path / "Geometry"
mesh_fname = mesh_outdir / "unloaded_geometry"
bc_params = {"pericardium_spring": 10}
# delet files excepts the
if outdir.is_dir():
    shutil.rmtree(outdir)
outdir.mkdir()
#%%
unloaded_geometry = pulse.HeartGeometry.from_file(mesh_fname.as_posix() + ".h5")
heart_model = HeartModelDynaComp(geo=unloaded_geometry, bc_params=bc_params)
collector = DataCollector(outdir=outdir,problem=heart_model)
# Initializing the model
v = heart_model.compute_volume(activation_value=0, pressure_value=0)
collector.collect(
    time=0,
    pressure=0,
    volume=v,
    activation=0.0,
)

# %% Loading PV Data
fname = directory_path / "PV data/PV_data.csv"
PV_data = np.loadtxt(fname.as_posix() ,delimiter=',')
# Converting mmHg to kPa
mmHg_to_kPa = 0.133322
pres = PV_data[0, :] * mmHg_to_kPa

# Pressurizing up to End Diastole
v = heart_model.compute_volume(activation_value=0, pressure_value=pres[0])
collector.collect(
    time=1,
    pressure=pres[0],
    volume=v,
    activation=0.0,
)

# Converting RVU to micro liter based on calculated EDV
RVU_to_microL = v/PV_data[0, 0]
vols = PV_data[1]*RVU_to_microL
#%%
max_a = 50
t_res = 10
for t, a in enumerate(np.linspace(0, max_a, t_res)):
    v = heart_model.compute_volume(
        activation_value=a, pressure_value=pres[t+1]
    )
    collector.collect(
        time=t+1,
        pressure=pres[t+1],
        volume=v,
        activation=a,
    )
# %%
deformed_mesh = heart_model.get_deformed_mesh()
# %%
