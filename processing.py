# %%
import numpy as np
from pathlib import Path
import shutil
from structlog import get_logger

import dolfin
import pulse
from heart_model import HeartModelDynaComp

logger = get_logger()


# %%

directory_path = Path("00_data/AS/3week/156_1/")
outdir = directory_path / "00_Modeling"
mesh_outdir = directory_path / "06_Mesh"
mesh_fname = mesh_outdir / "unloaded_geometry"
bc_params = {"pericardium_spring": 0.1}

unloaded_geometry = pulse.HeartGeometry.from_file(mesh_fname.as_posix() + ".h5")
heart_model = HeartModelDynaComp(geo=unloaded_geometry, bc_params=bc_params)

# delet files excepts the 
if outdir.is_dir():
    shutil.rmtree(outdir)
outdir.mkdir()
v = heart_model.compute_volume(activation_value=0, pressure_value=0)
heart_model.save(0,outdir=outdir)

v = heart_model.compute_volume(activation_value=0, pressure_value=.05)
heart_model.save(1,outdir=outdir)

#%%%
target_activation = dolfin.Function(heart_model.activation.ufl_function_space())
max_a = 10000
for t, a in enumerate(np.linspace(0, max_a, 10)):
    target_activation.vector()[:] = a
    v = heart_model.compute_volume(
        activation_value=target_activation, pressure_value=.05+a/max_a
    )
    heart_model.save(t+2,outdir=outdir)