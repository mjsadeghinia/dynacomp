# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
import arg_parser
import json

import dolfin
import pulse
from unloading import recreate_geometry_with_fibers
from heart_model import HeartModelDynaComp
from datacollector import DataCollector
from coupling_solver import newton_solver
import utils

logger = get_logger()

# %%
# UNITS:
# [kg]   [mm]    [s]    [mN]     [kPa]       [mN-mm]	    g = 9.806e+03
def load_edpvr_results(edpvr_dir):
    edpvr_data = utils.read_edpvr_data(edpvr_dir)
    edpvr_data_sorted = edpvr_data[edpvr_data[:, -1].argsort()]
    experimets_folders = utils.get_folders_from_edpvr_data(edpvr_data)
    unloaded_geometry_fname  = edpvr_dir / experimets_folders[0] / "unloaded_geometry_0_with_fibers.h5"
    a_matparam = edpvr_data_sorted[0, 0]
    af_matparam = edpvr_data_sorted[0, 1]
    return unloaded_geometry_fname, a_matparam, af_matparam

def update_matparam_settings(settings, a_matparam, af_matparam):
    settings["matparams"]["a"] = a_matparam
    settings["matparams"]["a_f"] = af_matparam
    return settings

def update_fibparam_settings(settings, epi_fiber, endo_fiber):
    settings["fiber_angles"]["alpha_epi_lv"] = epi_fiber
    settings["fiber_angles"]["alpha_endo_lv"] = endo_fiber
    return settings

# %%
def main(args=None) -> int:
    comm = dolfin.MPI.comm_world
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_processing(args)
    else:
        args = arg_parser.update_arguments(args, step="processing")

    sample_num = args.number
    sample_ID = args.ID
    setting_dir = args.settings_dir
    output_folder = args.output_folder
    results_dir = args.results_dir
    scan_type = args.scan_type
    epi_fiber = args.epi_fiber
    endo_fiber = args.endo_fiber
    fiber_modeling_flag = args.fiber_modeling_flag

    if sample_ID is not None:
        sample_num = utils.get_num_from_id(sample_ID, setting_dir)


    settings = utils.load_settings(setting_dir, sample_num)
    sample_name = settings["id"]
    bc_params = arg_parser.create_bc_params(args)

    sample_dir = Path(results_dir) / sample_name / scan_type
    pv_dir = sample_dir / "01_PVCalibration"
    geo_dir = pv_dir / "Geometries"
    edpvr_dir = sample_dir / "02_EDPVR_Modeling_v2"
    outdir = sample_dir / output_folder

    # delet files for saving again
    outdir = arg_parser.prepare_oudir_processing(outdir, comm)
    comm.Barrier()

    # Loading PV Data
    pressures, volumes = utils.load_pressure_volumes(pv_dir)
    if fiber_modeling_flag:
        peak_sys_ind = np.where(pressures == np.max(pressures))[0][0]
        pressures = pressures[:peak_sys_ind + 5]
        if comm.rank == 0:
            logger.warning(f"Fiber modeling enabled, using pressures up to peak systole (+5) peak_systole_pressure = {np.round(pressures[peak_sys_ind], 2)}.")
    
    
    unloaded_geometry_fname, a_matparam, af_matparam = load_edpvr_results(edpvr_dir)
    settings = update_matparam_settings(settings, a_matparam, af_matparam)
    if comm.rank == 0:
        logger.info("Updated settings with EDPVR matparams", a=a_matparam, a_f=af_matparam)

    unloaded_geometry = pulse.HeartGeometry.from_file(
        unloaded_geometry_fname.as_posix(), comm=comm
    )
    settings = update_fibparam_settings(settings, epi_fiber, endo_fiber)
    if comm.rank == 0:
        utils.save_settings(settings, setting_dir, sample_name)

    unloaded_geometry_with_updated_fibers = recreate_geometry_with_fibers(
            unloaded_geometry, settings["fiber_angles"]
        )
    if comm.rank == 0:
        logger.info("Updated settings with fiber angles", epi_fiber=epi_fiber, endo_fiber=endo_fiber)

    heart_model = HeartModelDynaComp(
        geo=unloaded_geometry_with_updated_fibers,
        bc_params=bc_params,
        matparams=settings["matparams"],
        comm=comm,
    )
    save_all = False if fiber_modeling_flag else True
    collector = DataCollector(outdir=outdir, model=heart_model, save_all=save_all)
    # Initializing the model
    v = heart_model.compute_volume(activation_value=0, pressure_value=0)
    collector.collect(
        time=0,
        pressure=0,
        volume=v,
        target_volume=v,
        activation=0.0,
    )
    ED_index_modeling = 0 if "ED_index_modeling" not in settings else settings["ED_index_modeling"]
    # Pressurizing up to End Diastole with 10 steps
    for i in range(1, 11):
        v = heart_model.compute_volume(activation_value=0, pressure_value=pressures[ED_index_modeling] * i / 10)
        collector.collect(
            time=i,
            pressure=pressures[ED_index_modeling] * i / 10,
            volume=v,
        target_volume=v,
        activation=0.0,
    )
    # Using newton method to find activation parameters based on PV data
    collector = newton_solver(
        heart_model=heart_model,
        pres=pressures[ED_index_modeling + 1:],
        vols=volumes[ED_index_modeling + 1:],
        collector=collector,
        start_time=11,
        comm=comm,
    )

if __name__ == "__main__":
    main()