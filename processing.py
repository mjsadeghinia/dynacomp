# %%
import numpy as np
from pathlib import Path
from structlog import get_logger

import dolfin
import pulse
from heart_model import HeartModelPulse
from circ.circulation_model import CirculationModel
from circ.datacollector import DataCollector

from coupling_solver import circulation_solver

logger = get_logger()


# %%

directory_path = Path("00_data/AS/3week/156_1/")
outdir = directory_path / "00_Modeling"
fname = outdir / "unloaded_geometry"
bc_params = {"pericardium_spring": 0.0001}

unloaded_geometry = pulse.HeartGeometry.from_file(fname.as_posix() + ".h5")
heart_model = HeartModelPulse(geo=unloaded_geometry, bc_params=bc_params)
collector = DataCollector(outdir=outdir, problem=heart_model)

v = heart_model.compute_volume(activation_value=0, pressure_value=0)


target_activation = dolfin.Function(heart_model.activation.ufl_function_space())

for t, a in enumerate(np.linspace(0, 200, 50)):
    target_activation.vector()[:] = a
    v = heart_model.compute_volume(
        activation_value=target_activation, pressure_value=0.0
    )
    collector.collect(
        time=t,
        pressure=0,
        volume=v,
        activation=0.0,
        flow=0,
        p_ao=0,
    )

# %%
