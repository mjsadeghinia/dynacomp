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
from unloading import unloading

logger = get_logger()

comm = dolfin.MPI.comm_world

# %%

# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03

def main(path, results_folder, matparams: dict = None):
    
    directory_path = Path(path)
    bc_params = {"base_spring": 1}

    if results_folder is not None or not results_folder == "":
        results_folder_dir = directory_path / results_folder
        results_folder_dir.mkdir(exist_ok=True)
    else:
        results_folder_dir = directory_path
    outdir = results_folder_dir / "00_Modeling"
    mesh_outdir = results_folder_dir / "Geometry"
    unloaded_geometry_fname = mesh_outdir / "unloaded_geometry_with_fibers"

    # delet files for saving again
    if outdir.is_dir() and comm.Get_rank() == 0:
        shutil.rmtree(outdir)
        outdir.mkdir(exist_ok=True)
    comm.Barrier()

    # Loading PV Data
    def caliberate_volumes(ED_geometry_fname, vols, comm=None):
        ED_geometry = pulse.HeartGeometry.from_file(
            ED_geometry_fname.as_posix() + ".h5", comm=comm
        )
        v = ED_geometry.cavity_volume()
        RVU_to_microL = v / vols[0]
        if comm.Get_rank() == 0:
            logger.info(f"Caliberation is done, RVU to micro Liter is {RVU_to_microL}")
        volumes = vols * RVU_to_microL
        return volumes
    
    fname = directory_path / "PV data/PV_data.csv"
    PV_data = np.loadtxt(fname.as_posix(), delimiter=",")
    # Converting mmHg to kPa
    mmHg_to_kPa = 0.133322
    pressures = PV_data[0, :] * mmHg_to_kPa
    # Converting RVU to micro liter based on calculated EDV
    ED_geometry_fname = mesh_outdir / "geometry"
    volumes = caliberate_volumes(ED_geometry_fname, PV_data[1, :], comm=comm)
    # 
    unloaded_geometry = pulse.HeartGeometry.from_file(
        unloaded_geometry_fname.as_posix() + ".h5", comm=comm
    )
    heart_model = HeartModelDynaComp(geo=unloaded_geometry, bc_params=bc_params, matparams=matparams, comm=comm)
    collector = DataCollector(outdir=outdir, problem=heart_model)

    # Initializing the model
    v = heart_model.compute_volume(activation_value=0, pressure_value=0)
    collector.collect(
        time=0,
        pressure=0,
        volume=v,
        target_volume=v,
        activation=0.0,
    )
    # Pressurizing up to End Diastole
    v = heart_model.compute_volume(activation_value=0, pressure_value=pressures[0])
    collector.collect(
        time=1,
        pressure=pressures[0],
        volume=v,
        target_volume=v,
        activation=0.0,
    )

    # 
    collector = newton_solver(
        heart_model=heart_model,
        pres=pressures[1:],
        vols=volumes[1:],
        collector=collector,
        start_time=2,
        comm=comm,
    )
    return collector
    # %%
sample_name = "138_1"
results_folder = "00_Results_coarse_with_unloading"
paths = {
        "OP130_2": "00_data/SHAM/6week/OP130_2",
        "156_1": "00_data/AS/3week/156_1",
        "138_1": "00_data/AS/12week/138_1",
    }

matparams = dict(a=1.726,a_f=7.048,b=1.118,b_f=0.001,a_s=0.0,b_s=0.0,a_fs=0.0,b_fs=0.0)
unloaded_geo =  unloading(paths[sample_name], results_folder, matparams=matparams)
collector = main(paths[sample_name], results_folder, matparams=matparams)