# main.py
import argparse
import numpy as np
from pathlib import Path
import json

import arg_parser
import pulse
import dolfin
from heart_model import HeartModelDynaComp
from datacollector import DataCollectorInflator

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
    return pres, vols

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
        default=2.0,
        help='Multiplier for the initial pressure step.'
    )
    parser.add_argument(
        '--pressure_steps',
        type=int,
        default=20,
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

    # Determine which samples to process
    if args.number:
        sample_list = args.number
    else:
        settings_files = sorted([f for f in args.settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(settings_files) + 1))

    bc_params = arg_parser.create_bc_params(args)
    comm = dolfin.MPI.comm_world

    for sample in sample_list:
        settings = load_settings(args.settings_dir, sample)
        name = settings['id']
        out_dirs = {
            'pv': args.results_dir / name / 'PV Data',
            'calib': args.results_dir / name / args.scan_type / '01_PVCalibration',
            'unload': args.results_dir / name / args.scan_type / '02_Unloading',
            'model': args.results_dir / name / args.scan_type / '03_Modeling'
        }
        if comm.rank == 0:
            arg_parser.prepare_oudir_processing(out_dirs['model'], comm)
        comm.Barrier()

        # Load data
        time, pres, vols = load_pv_data(out_dirs['calib'])
        edpvr_pres, edpvr_vols_unc = load_edpvr(out_dirs['pv'])
        a = settings['PV']['calibration']['a']
        b = settings['PV']['calibration']['b']
        edpvr_vols = a * edpvr_vols_unc + b

        # Set up model
        geometry = pulse.HeartGeometry.from_file(
            (out_dirs['unload'] / 'unloaded_geometry_0_with_fibers.h5').as_posix(), comm=comm
        )
        model = HeartModelDynaComp(
            geo=geometry,
            bc_params=bc_params,
            matparams=settings['matparams'],
            comm=comm,
        )

        # Initialize data collector with live plotting
        collector = DataCollectorInflator(
            outdir=out_dirs['model'],
            problem=model,
            pv_vols=vols,
            pv_pres=pres,
            edpvr_vols=edpvr_vols,
            edpvr_pres=edpvr_pres,
            live_plot=True
        )

        # Initial state
        v0 = model.compute_volume(activation_value=0, pressure_value=0)
        collector.collect(time=0, pressure=0, volume=v0)

        # Simulation loop
        pressures = np.linspace(0, pres[0] * args.pressure_multiplier, args.pressure_steps)
        for i, p in enumerate(pressures, start=1):
            model.compute_volume(activation_value=0, pressure_value=p)
            collector.collect(time=i, pressure=model.get_pressure(), volume=model.get_volume())

        # Finalize plot session
        if comm.rank == 0:
            collector.finalize_plot()

    return 0


if __name__ == '__main__':
    main()