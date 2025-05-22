import argparse
import numpy as np
from pathlib import Path
import json
import structlog
import scipy.interpolate
from matplotlib import pyplot as plt

import arg_parser
import pulse
import dolfin
from heart_model import HeartModelDynaComp
from datacollector import DataCollectorInflator

comm = dolfin.MPI.comm_world
logger = structlog.get_logger()

def load_settings(settings_dir: Path, sample_num: int) -> dict:
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)

def load_pv_data(directory: Path):
    data = np.loadtxt(directory / "calibrated_pv_data.csv", delimiter=',')
    time, pres_mmHg, vols = data.T
    pres = pres_mmHg * 0.133322  # mmHg to kPa
    return time, pres, vols

def load_edpvr(directory: Path):
    path = next(f for f in directory.iterdir() if "EDPVR.csv" in f.name)
    data = np.loadtxt(path, delimiter=',')
    pres = data[:, 0] * 0.133322
    vols = data[:, 1]
    idx_v = np.argsort(vols)
    vols = vols[idx_v]
    pres = pres[idx_v]
    return pres, vols

def calibration_edpvr_vols(edpvr_vols_unc, settings):
    a = settings['PV']['calibration']['a']
    b = settings['PV']['calibration']['b']
    edpvr_vols = a * edpvr_vols_unc + b
    return edpvr_vols

def calculate_error(edpvr_spline, inflation_pres, inflation_vols):
    edpvr_pres_interp = edpvr_spline(inflation_vols)
    error = np.sqrt(np.mean((edpvr_pres_interp - inflation_pres)**2))
    return error

def plot_results(fname, error, matparams, inflation_pres, inflation_vols, edpvr_pres, edpvr_vols, edpvr_spline, pv_vols, pv_pres):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pv_vols, pv_pres, 'k', linewidth=1)
    ax.scatter(pv_vols, pv_pres, s=15, c='k', label='PV Data')
    ax.scatter(edpvr_vols, edpvr_pres, s=8, c='r', label='EDPVR')
    # Regression line
    edpvr_vols_spline = np.linspace(min(edpvr_vols), max(edpvr_vols), 100)
    edpvr_pres_spline = edpvr_spline(edpvr_vols_spline)
    ax.plot(edpvr_vols_spline, edpvr_pres_spline, 'b')
    ax.axhline(0, color='gray', linestyle='--')
    # Simulation placeholders
    ax.plot(inflation_vols, inflation_pres, 'g-', linewidth=1, label='Simulation')
    ax.scatter(inflation_vols, inflation_pres, c='g', s=8)
    # Annotate slope and intercept
    textstr = (
        f'error: {error:.2f}kPa \n'
        f"a = {round(matparams['a'], 3)}\n"
        f"a_f = {round(matparams['a_f'], 3)}\n"
        f"b = {round(matparams['b'], 3)}\n"
        f"b_f = {round(matparams['b_f'], 3)}\n"
    )
    ax.text(
                0.05,
                0.95,
                textstr,
                fontsize=10,
                verticalalignment='top'
            )
    ax.set_xlabel('Volume [microL]')
    ax.set_ylabel('LV Pressure [kPa]')
    ax.legend(loc='lower left')
    fig.savefig(fname, dpi=300)

#%%
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-n',
        '--number',
        nargs='*',
        type=int,
        default=None,
        help='Sample number(s) to process. If omitted, all samples in settings_dir will be processed.'
    )
    parser.add_argument(
        '--settings_dir',
        type=Path,
        default=Path('/home/shared/dynacomp/settings'),
        help='Directory where JSON settings files are stored.'
    )
    parser.add_argument(
        '-d',
        '--data_dir',
        type=Path,
        default=Path('/home/shared/00_data'),
        help='Directory where data files are stored.'
    )
    parser.add_argument(
        '-s',
        '--scan_type',
        type=str,
        default='TPM',
        help='Scan type; subdirectories will be named accordingly.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=Path,
        default=Path('/home/shared/01_results_coarse_mesh'),
        help='Directory where results will be saved.'
    )
    parser.add_argument(
        '--pressure_multiplier',
        type=float,
        default=1.0,
        help='Multiplier for the initial pressure step.'
    )
    parser.add_argument(
        '--pressure_steps',
        type=int,
        default=10,
        help='Number of pressure increments in the inflation simulation.'
    )
    parser.add_argument(
        '--pericardium_spring',
        type=float,
        default=1e-4,
        help='Spring stiffness for pericardium boundary condition.'
    )
    parser.add_argument(
        '--base_spring',
        type=float,
        default=1.0,
        help='Spring stiffness at the heart base boundary condition.'
    )

    parser.add_argument(
        '--a_matparam',
        type=float,
        default=None,
        help='Material parameter a for the heart model.'
    )
    parser.add_argument(
        '--af_matparam',
        type=float,
        default=None,
        help='Material parameter a_f for the heart model.'
    )
    parser.add_argument(
        '--b_matparam',
        type=float,
        default=None,
        help='Material parameter a for the heart model.'
    )
    parser.add_argument(
        '--bf_matparam',
        type=float,
        default=None,
        help='Material parameter a_f for the heart model.'
    )

    parser.add_argument(
        '--spline_smoothness',
        type=float,
        default=0.5,
        help='Smoothness parameter for the spline interpolation of EDPVR data.'
    )

    parser.add_argument(
        '-p',
        '--plot_flag',
        action='store_true',
        help='Flag to indicate whether to plot the results.'
    )
    parser.add_argument(
        '-l',
        '--logging_flag',
        action='store_true',
        default='logging the results',
        help='Flag to indicate whether to log the results.'
        )

    args = parser.parse_args()

    number = args.number
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    scan_type = args.scan_type
    pressure_multiplier = args.pressure_multiplier
    pressure_steps = args.pressure_steps
    input_matparams = arg_parser.prepare_matparams(args)
    bc_params = arg_parser.create_bc_params(args)
    spline_smoothness = args.spline_smoothness
    plot_flag = args.plot_flag
    logging_flag = args.logging_flag
    # Determine sample list
    if number:
        sample_nums = number
    else:
        settings_files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(settings_files) + 1))

    # Run inflation for each sample
    for sample_num in sample_nums:
        settings = load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        sample_dir = results_dir / sample_name / scan_type

        if not sample_dir.exists():
            continue

        # Prepare output directory
        output_dir = sample_dir / "03_Inflation"
        if comm.rank == 0:
            arg_parser.prepare_oudir_processing(output_dir, comm)
        comm.Barrier()

        # Load PV and EDPVR data
        _, pv_pres, pv_vols = load_pv_data(sample_dir / "01_PVCalibration")
        EDV = pv_vols[0]
        edpvr_pres, edpvr_vols_unc = load_edpvr(sample_dir.parent / "PV Data")
        edpvr_vols = calibration_edpvr_vols(edpvr_vols_unc, settings)
        edpvr_spline = scipy.interpolate.UnivariateSpline(edpvr_vols, edpvr_pres, s=spline_smoothness, k=2)
        # Creating FE model
        geometry = pulse.HeartGeometry.from_file(
        (sample_dir / '02_Unloading/unloaded_geometry_0_with_fibers.h5').as_posix(), comm=comm
        )
        # Set material properties
        matparams = settings['matparams']
        for key, value in input_matparams.items():
            matparams[key] = value
        # Initialize heart model
        model = HeartModelDynaComp(
            geo=geometry,
            bc_params=bc_params,
            matparams=matparams,
            comm=comm,
        )

        # Setup pressure for inflation
        pressures = np.linspace(0, pv_pres[0] * pressure_multiplier, pressure_steps)
        # Run inflation steps
        inflation_pres = []
        inflation_vols = []

        for i, p in enumerate(pressures):
            v = model.compute_volume(activation_value=0, pressure_value=p, logging_flag=False)
            inflation_pres.append(p)
            inflation_vols.append(v)
            if v>EDV:
                # If the volume exceeds EDV, break the loop
                logger.info(f"Volume exceeded EDV at pressure {p:.2f} kPa. Stopping inflation.")
                break
            if comm.rank == 0:
                logger.info(f"Inflation step {i}: ", pressure=round(p,3), volume=round(v,3))
        
        error = calculate_error(edpvr_spline, inflation_pres, inflation_vols)
        if comm.rank == 0:
            logger.info(f"Inflation RMS error: {error:.3f} kPa")
            if plot_flag:
                fname = output_dir / f"inflation_results.png"
                plot_results(fname, error, matparams, inflation_pres, inflation_vols, edpvr_pres, edpvr_vols, edpvr_spline, pv_vols, pv_pres)
            if logging_flag:
                # Save results to a file
                fname = output_dir.parent / f"inflation_results.txt"
                if not fname.exists():
                    header = "a,a_f,b,b_f,error\n"
                    fname.write_text(header, encoding="utf-8")
                with fname.open("a", encoding="utf-8") as f:
                    f.write(f"{model.material.parameters['a']},"
                            f"{model.material.parameters['a_f']},"
                            f"{model.material.parameters['b']},"
                            f"{model.material.parameters['b_f']},"
                            f"{round(error,3)}\n")

if __name__ == '__main__':
    main()