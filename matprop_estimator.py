import argparse
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import structlog

from inflator import run_inflator, load_edpvr

logger = structlog.get_logger()

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
    sample: int,
    settings_dir: Path,
    results_dir: Path,
    scan_type: str
) -> float:
    """
    Objective for optimization: run inflation with given a, a_f and return error.
    """
    a_opt, af_opt = float(x[0]), float(x[1])
    mat_override = {'a': a_opt, 'a_f': af_opt}

    collector = run_inflator(
        sample,
        settings_dir=settings_dir,
        results_dir=results_dir,
        scan_type=scan_type,
        matparams=mat_override,
    )

    settings = load_settings(settings_dir, sample)
    sample_id = settings['id']
    pv_outdir = results_dir / sample_id / 'PV Data'

    pressures = np.array(collector.pressures)
    volumes = np.array(collector.volumes)
    err = calculate_error(pressures, volumes, pv_outdir, settings)
    logger.debug("objective_evaluation", sample=sample, a=a_opt, a_f=af_opt, error=err)
    return err


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
    args = parser.parse_args()

    settings_dir = args.settings_dir
    data_dir = args.data_dir
    scan_type = args.scan_type
    results_dir = args.results_dir

    if args.number:
        sample_list = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(files) + 1))

    ftol = 1e-3
    maxiter = 30

    for sample in sample_list:
        settings = load_settings(settings_dir, sample)
        a0 = settings['matparams']['a']
        af0 = settings['matparams']['a_f']
        x0 = np.array([a0, af0])
        bounds = [(a0/5, 5*a0), (af0/5, 5*af0)]
        
        logger.info("--------------")
        logger.info("starting_optimization", sample=sample, a0=a0, a_f0=af0)

        res = minimize(
            objective,
            x0,
            args=(sample, settings_dir, results_dir, scan_type),
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
