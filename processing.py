# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
import arg_parser
import json

import dolfin
import pulse
from heart_model import HeartModelDynaComp
from datacollector import DataCollector
from coupling_solver import newton_solver

logger = get_logger()

# %%
# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
def get_sample_name(sample_num, setting_dir):
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in setting_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    sample_name = sorted_files[sample_num - 1].with_suffix("").name
    return sample_name

def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def load_pressure_volumes(data_dir, sample_name):
    PV_data_fname = data_dir / f"PV data/PV data/{sample_name}_PV_data.csv"
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    pressures = PV_data[:, 1] * mmHg_to_kPa
    volumes = PV_data[:, 2]
    return pressures, volumes

def caliberate_volumes(mesh_dir, vols, comm=None):
    ED_geometry_fname = mesh_dir / "geometry"
    ED_geometry = pulse.HeartGeometry.from_file(
        ED_geometry_fname.as_posix() + ".h5", comm=comm
    )
    v = ED_geometry.cavity_volume()
    RVU_to_microL = v / vols[0]
    if comm.Get_rank() == 0:
        logger.info(f"Caliberation is done, RVU to micro Liter is {RVU_to_microL}")
    volumes = vols * RVU_to_microL
    return volumes


# %%
def main(args=None) -> int:
    comm = dolfin.MPI.comm_world
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_processing(args)
    else:
        args = arg_parser.update_arguments(args, step="processing")

    sample_num = args.number
    setting_dir = args.settings_dir
    output_folder = args.output_folder
    sample_name = get_sample_name(sample_num, setting_dir)
    settings = load_settings(setting_dir, sample_name)
    bc_params = arg_parser.create_bc_params(args)
    data_dir = Path(settings["path"])
    mesh_dir = data_dir / f"{output_folder}/Geometry"

    # delet files for saving again
    outdir = arg_parser.prepare_oudir_processing(data_dir, output_folder, comm)
    comm.Barrier()

    # Loading PV Data
    pressures, volumes = load_pressure_volumes(data_dir, sample_name)
    volumes = caliberate_volumes(mesh_dir, volumes, comm=comm)
    #
    unloaded_geometry_fname = mesh_dir / "unloaded_geometry_with_fibers.h5"
    unloaded_geometry = pulse.HeartGeometry.from_file(
        unloaded_geometry_fname.as_posix(), comm=comm
    )
    heart_model = HeartModelDynaComp(
        geo=unloaded_geometry,
        bc_params=bc_params,
        matparams=settings["matparams"],
        comm=comm,
    )
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
    # Using newton method to find activation parameters based on PV data
    collector = newton_solver(
        heart_model=heart_model,
        pres=pressures[1:],
        vols=volumes[1:],
        collector=collector,
        start_time=2,
        comm=comm,
    )
    
if __name__ == "__main__":
    main()