import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

import arg_parser 
import pulse
import dolfin
from heart_model import HeartModelDynaComp
from datacollector import DataCollector_Inflator

from structlog import get_logger

logger = get_logger()
comm = dolfin.MPI.comm_world

# %%
def load_settings(settings_dir, sample_num):
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    settings_fname = sorted_files[sample_num - 1]
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def load_pv_data(pvcalibration_data_dir):
    # These are the paramters we are using: mri_time, regirstered_pressures, regirstered_calibrated_volumes
    fname = pvcalibration_data_dir / "calibrated_pv_data.csv"
    PV_data = np.loadtxt(fname, delimiter=",")
    time, pres_mmHg, vols = PV_data[:, 0], PV_data[:, 1], PV_data[:, 2]
    # Convert mmHg to kPa
    mmHg_to_kPa = 0.133322
    pres = pres_mmHg * mmHg_to_kPa
    return time, pres, vols

def load_edpvr(data_dir):
    PV_data_fname = [fname for fname in data_dir.iterdir() if "EDPVR.csv" in fname.as_posix()][0]
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    pressures = PV_data[:, 0] * mmHg_to_kPa
    volumes = PV_data[:, 1]
    return pressures, volumes

# %%
def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        nargs="+",
        type=int,
        help="The sample number(s), will process all the sample if not indicated",
    )

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        default="/home/shared/00_data",
        type=Path,
        help="The settings directory where data files are stored.",
    )

    parser.add_argument(
        "-s",
        "--scan_type",
        default='TPM',
        type=str,
        help="The scan type. Settings will be loaded accordingly from json file",
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    # Arguments for HeartModel boundary conditions
    parser.add_argument(
        "--pericardium_spring",
        default=0.0001,
        type=float,
        help="HeartModel BC: The stiffness of the spring on the pericardium.",
    )
    parser.add_argument(
        "--base_spring",
        default=1,
        type=float,
        help="HeartModel BC: The stiffness of the spring at the base.",
    )

    args = parser.parse_args(args)

    sample_nums = args.number
    settings_dir = args.settings_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    scan_type = args.scan_type
    bc_params = arg_parser.create_bc_params(args)

    for sample_num in sample_nums:
        settings = load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        pv_data_dir = results_dir / sample_name / "PV Data"
        experiment_data_dir = results_dir / sample_name / scan_type
        pvcalibration_data_dir = results_dir / sample_name / "TPM" / "01_PVCalibration"
        unloading_data_dir = experiment_data_dir / "02_Unloading"
        modeling_outdir = experiment_data_dir / "03_Modeling"
        modeling_outdir.mkdir(parents=True, exist_ok=True)
 
        
        time, pres, vols = load_pv_data(pvcalibration_data_dir)
        edpvr_pres, edpvr_vols = load_edpvr(pv_data_dir)


        unloaded_geometry_fname = unloading_data_dir / "unloaded_geometry_0_with_fibers.h5"
        unloaded_geometry = pulse.HeartGeometry.from_file(
            unloaded_geometry_fname.as_posix(), comm=comm
        )
        heart_model = HeartModelDynaComp(
            geo=unloaded_geometry,
            bc_params=bc_params,
            matparams=settings["matparams"],
            comm=comm,
        )
        collector = DataCollector_Inflator(outdir=modeling_outdir, problem=heart_model)
        # Initializing the model
        v = heart_model.compute_volume(activation_value=0, pressure_value=0)
        collector.collect(
            time=0,
            pressure=0,
            volume=v,
        )
        for p in np.linspace(0, pres[0] * 2, 10):
            v = heart_model.compute_volume(activation_value=0, pressure_value=p)
            collector.collect(
                time=1,
                pressure=p,
                volume=v,
            )

if __name__ == "__main__":
    main()
