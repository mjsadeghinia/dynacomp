# param_sweep.py

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats
import structlog
import dolfin
import pulse
import arg_parser
from heart_model import HeartModelDynaComp
from inflator import run_inflator, load_edpvr, load_pv_data

logger = structlog.get_logger()
comm = dolfin.MPI.comm_world


# %%
def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)


def sweep_parameters(
    model: HeartModelDynaComp, pressures: np.ndarray, edpvr_slope: float, a_vals: np.ndarray, af_vals: np.ndarray, comm
):
    """
    Returns meshgrid (A, AF) and error_grid of shape (len(af_vals), len(a_vals))
    """
    A, AF = np.meshgrid(a_vals, af_vals, indexing="xy")
    error_grid = np.zeros_like(A)
    for i, a in enumerate(a_vals):
        for j, af in enumerate(af_vals):
            # update the two parameters
            model.update_matparams({"a": a, "a_f": af})
            model.problem._init_forms()
            model.compute_volume(activation_value=0, pressure_value=0, logging_flag=False)
            # run the same inflation as in objective()
            pres_out, vol_out = run_inflator(model, pressures, comm=comm)
            slope_out = np.mean(np.diff(pres_out) / np.diff(vol_out))
            error_grid[j, i] = np.round(np.abs(slope_out - edpvr_slope) / edpvr_slope * 100, 2)
            if comm.rank == 0:
                logger.info("sweep_step", a=model.problem.material.a.values()[0], a_f=model.problem.material.a_f.values()[0], error=error_grid[j, i])
    return A, AF, error_grid


def plot_contours(A, AF, error_grid, out_path: Path):
    plt.figure()
    cp = plt.contourf(A, AF, error_grid, levels=20)
    plt.colorbar(cp, label="Error (%)")
    plt.xlabel("a")
    plt.ylabel("a_f")
    plt.title("Parameter-sweep error contours")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# %%
def main():
    parser = argparse.ArgumentParser(description="2D parameter sweep of (a, a_f) for HeartModelDynaComp")
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
        default=3,
        help='Number of pressure increments in the inflation simulation.'
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=4, 
        help="Number of points along each axis"
        )
    parser.add_argument(
        '--a_min',
        default=.25,
        type=float,
        help='Minimum a-value for the sweep grid.'
    )
    parser.add_argument(
        '--a_max',
        type=float,
        default=2,
        help='Maximum a-value for the sweep grid.'
    )
    parser.add_argument(
        '--af_min',
        type=float,
        default=1,
        help='Minimum a_f-value for the sweep grid.'
    )
    parser.add_argument(
        '--af_max',
        type=float,
        default=8,
        help='Maximum a_f-value for the sweep grid.'
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
    scan_type = args.scan_type
    results_dir = args.results_dir
    pressure_multiplier = args.pressure_multiplier
    pressure_steps = args.pressure_steps
    grid_size = args.grid_size
    pericardium_spring = args.pericardium_spring
    base_spring = args.base_spring
    a_min, a_max = args.a_min, args.a_max
    af_min, af_max = args.af_min, args.af_max

    if args.number:
        sample_list = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(files) + 1))

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

        # prepare pressures to pass to run_inflator
        _, pres, _ = load_pv_data(out_dirs["calib"])
        pressures = np.linspace(0, pres[0] * pressure_multiplier, pressure_steps)

        # build model once
        bc_params = arg_parser.create_bc_params(
            argparse.Namespace(pericardium_spring=pericardium_spring, base_spring=base_spring)
        )
        geo = pulse.HeartGeometry.from_file(
            (out_dirs['unload'] / 'unloaded_geometry_0_with_fibers.h5').as_posix(), comm=comm
        )
        model = HeartModelDynaComp(geo=geo, bc_params=bc_params, matparams=settings["matparams"], comm=comm)

        # initial guesses
        a0 = settings["matparams"]["a"]
        af0 = settings["matparams"]["a_f"]
        # sweep ranges
        a_vals = np.round(np.linspace(a_min, a_max, grid_size), 2)
        af_vals = np.round(np.linspace(af_min, af_max, grid_size), 2)
        A, AF, err = sweep_parameters(model, pressures, edpvr_slope, a_vals, af_vals, comm)

        # only rank 0 plots
        if comm.rank == 0:
            plot_path = results_dir / sample_id / "param_sweep_contours.png"
            plot_contours(A, AF, err, plot_path)
            logger.info("saved_contour_plot", path=str(plot_path))


if __name__ == "__main__":
    main()
