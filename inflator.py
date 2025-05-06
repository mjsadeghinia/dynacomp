import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import scipy.stats

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
        modeling_outdir = arg_parser.prepare_oudir_processing(modeling_outdir, comm)
        comm.Barrier()
        
        time, pres, vols = load_pv_data(pvcalibration_data_dir)
        edpvr_pres, edpvr_vols_uncalibrated = load_edpvr(pv_data_dir)
        edpvr_vols = settings["PV"]["calibration"]["a"] * edpvr_vols_uncalibrated + settings["PV"]["calibration"]["b"]


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
        start_time = 1
        for i, p in enumerate(np.linspace(0, pres[0], 10)):
            v = heart_model.compute_volume(activation_value=0, pressure_value=p)
            p_current = heart_model.get_pressure()
            v_current = heart_model.get_volume()
            collector.collect(
                time=i + start_time,
                pressure=p_current,
                volume=v_current,
            )

        if comm.rank == 0:
            # Calculate the x value at which y = 0 using the regression line equation (avoid division by zero)
            res = scipy.stats.linregress(edpvr_vols, edpvr_pres)
            v_0 = -res.intercept / res.slope if res.slope != 0 else float('nan')
            # Calculate the standard error of the slope and intercept
            tinv = lambda p, df: abs(scipy.stats.t.ppf(p/2, df))
            ts = tinv(0.05, len(edpvr_vols)-2)
            # Plotting the data
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(vols, pres, "k", linewidth=1)
            ax.scatter(vols, pres, s=15, c="k", label="PV Data")
            ax.scatter(edpvr_vols, edpvr_pres, s=8, c="r", label="EDPVR")
            ax.plot(collector.volumes, collector.pressures, "g", linewidth=1)
            ax.scatter(collector.volumes, collector.pressures, color="g", s=8, label="Simulation")
            
            plt.xlabel("Volume [micro Liter]")
            plt.ylabel("LV Pressure [mmHg]")

            # Add a title with the slope and intercept
            textstr = (
                    f"slope (95%): {res.slope:.3f} $\pm$ {ts*res.stderr:.3f}\n"
                    f"$v_0$ (P=0): {v_0:.2f}\n"
                    f"$v_0 estimated$ (P=0): {collector.volumes[0]:.2f}"

                )
            ax.text(
                0.05, 0.95, textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            ax.plot(edpvr_vols, res.intercept + res.slope*edpvr_vols, 'b', label='EDVPR')
            ax.axhline(y=0, color='gray', linestyle='--')

            # Add a second y-axis for LV Pressure in kPa
            ax2 = ax.twinx()
            mmHg_to_kPa = 0.133322
            ymin, ymax = ax.get_ylim()
            ax2.set_ylim(ymin * mmHg_to_kPa, ymax * mmHg_to_kPa)
            ax2.set_ylabel("LV Pressure [kPa]")

            ax.legend(loc="lower left")
            fname = modeling_outdir / f"inflation_results.png"
            plt.savefig(fname, dpi=300)
            plt.close()

if __name__ == "__main__":
    main()
