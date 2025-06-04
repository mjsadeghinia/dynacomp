import argparse
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import scipy.stats
import structlog
import arg_parser
from heart_model import HeartModelDynaComp

import pulse
import dolfin
from inflator import run_inflator, load_edpvr, load_pv_data

logger = structlog.get_logger()
comm = dolfin.MPI.comm_world


def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)


def calculate_error(
    pressures: np.ndarray,
    volumes: np.ndarray,
    pv_outdir: dict,
    settings: dict
) -> float:
    """
    Compute the mean absolute error between the simulated pressures and
    the interpolated EDPVR pressures at the simulated volumes.
    """
    edpvr_pres, edpvr_vols_unc = load_edpvr(pv_outdir)
    a = settings['PV']['calibration']['a']
    b = settings['PV']['calibration']['b']
    edpvr_vols = a * edpvr_vols_unc + b

    idx = np.argsort(edpvr_vols)
    edpvr_pres = edpvr_pres[idx]
    edpvr_vols = edpvr_vols[idx]

    interp_pres = np.interp(volumes, edpvr_vols, edpvr_pres)
    error = np.mean(np.abs(pressures - interp_pres))
    return error


def objective(
    x: np.ndarray,
    model: HeartModelDynaComp,
    pressures: np.array,
    edpvr_slope: float,
    comm: dolfin.MPI.comm_world
) -> float:
    updated_matparams = {'a': x[0], 'a_f': x[1]}
    # Update the material parameters in the model
    model.update_matparams(updated_matparams)
    # Run the inflation simulation
    res_pres, res_vols = run_inflator(model, pressures, comm=comm) 
    res_slope = np.mean(np.diff(res_pres) / np.diff(res_vols))
    # Calculate the error as a percentage of the EDPVR slope
    error = np.abs(res_slope - edpvr_slope)/edpvr_slope*100
    if comm.rank == 0:
        logger.info("objective_evaluation", a=x[0], a_f=x[1], error=error)
    return error


def main():
    parser = argparse.ArgumentParser(
        description="Estimate material parameters (a, a_f) by fitting an inflation model to EDPVR data"
    )
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
        default=2,
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
    args = parser.parse_args()

    settings_dir = args.settings_dir
    data_dir = args.data_dir
    scan_type = args.scan_type
    results_dir = args.results_dir
    pressure_multiplier = args.pressure_multiplier
    pressure_steps = args.pressure_steps
    pericardium_spring = args.pericardium_spring
    base_spring = args.base_spring

    if args.number:
        sample_list = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(files) + 1))

    ftol = 1e-3
    maxiter = 30

    for sample in sample_list:
        settings = load_settings(settings_dir, sample)
        sample_id = settings['id']
        # Prepare directories
        out_dirs = {
            'pv': results_dir / sample_id / 'PV Data',
            'calib': results_dir / sample_id / scan_type / '01_PVCalibration',
            'unload': results_dir / sample_id / scan_type / '02_Unloading',
            'model': results_dir / sample_id / scan_type / '03_Modeling'
        }
        # Load PV and EDPVR data
        time, pres, vols = load_pv_data(out_dirs['calib'])
        edpvr_pres, edpvr_vols_unc = load_edpvr(out_dirs['pv'])
        a = settings['PV']['calibration']['a']
        b = settings['PV']['calibration']['b']
        edpvr_vols = a * edpvr_vols_unc + b

        # sort EDPVR data for the interpolation
        idx = np.argsort(edpvr_vols)
        edpvr_pres = edpvr_pres[idx]
        edpvr_vols = edpvr_vols[idx]

        res = scipy.stats.linregress(edpvr_vols, edpvr_pres)
        edpvr_slope = res.slope
        # Initialize heart model
        bc_params = arg_parser.create_bc_params(
            argparse.Namespace(
                pericardium_spring=pericardium_spring,
                base_spring=base_spring
            )
        )
        geometry = pulse.HeartGeometry.from_file(
            (out_dirs['unload'] / 'unloaded_geometry_0_with_fibers.h5').as_posix(), comm=comm
        )

        # Set material properties
        matparams = settings['matparams']
        # Initialize the heart model
        model = HeartModelDynaComp(
            geo=geometry,
            bc_params=bc_params,
            matparams=matparams,
            comm=comm,
        )

        a0 = settings['matparams']['a']
        af0 = settings['matparams']['a_f']
        x0 = np.array([a0, af0])
        bounds = [(a0/5, 5*a0), (af0/5, 5*af0)]
        
        if comm.rank == 0:
            logger.info("--------------")
            logger.info("starting_optimization", sample=sample, a0=a0, a_f0=af0)

        pressures = np.linspace(0, pres[0] * pressure_multiplier, pressure_steps)


        fun = lambda x: objective(
            x,
            model,
            pressures,
            edpvr_slope,
            comm
        )

        res = minimize(
            fun,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': ftol, 'maxiter': maxiter}
        )

        if res.success:
            best_a, best_af = res.x
            logger.info(
                "optimization_converged",
                sample=sample,
                a=best_a,
                a_f=best_af,
                error=res.fun
            )
        else:
            logger.error(
                "optimization_failed",
                sample=sample,
                message=res.message
            )

if __name__ == '__main__':
    main()
