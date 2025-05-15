import argparse
import numpy as np
from pathlib import Path
import json
import structlog



import arg_parser
import pulse
import dolfin
from heart_model import HeartModelDynaComp
from datacollector import DataCollectorInflator

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
    return pres, vols

def run_inflator_with_collector(
    sample: int,
    settings_dir: Path,
    results_dir: Path,
    scan_type: str = 'TPM',
    pressure_multiplier: float = 1.0,
    pressure_steps: int = 2,
    pericardium_spring: float = 1e-4,
    base_spring: float = 1.0,
    matparams: dict = None,
    live_plot: bool = True
) -> DataCollectorInflator:
    """
    Run an inflation simulation for a given sample index using explicit parameters.

    Parameters:
        sample: sample number (1-based index into sorted settings_dir JSON files)
        settings_dir: directory containing JSON settings files
        results_dir: base output directory for results
        scan_type: name of scan-type subdirectory
        pressure_multiplier: multiplier for initial pressure
        pressure_steps: number of pressure increments
        pericardium_spring: pericardium BC stiffness
        base_spring: base BC stiffness
        live_plot: whether to show and update a live plot

    Returns:
        DataCollectorInflator with collected data and saved figures
    """
    # Load settings
    settings = load_settings(settings_dir, sample)
    sample_id = settings['id']

    # Prepare directories
    out_dirs = {
        'pv': results_dir / sample_id / 'PV Data',
        'calib': results_dir / sample_id / scan_type / '01_PVCalibration',
        'unload': results_dir / sample_id / scan_type / '02_Unloading',
        'model': results_dir / sample_id / scan_type / '03_Modeling'
    }
    comm = dolfin.MPI.comm_world
    if comm.rank == 0:
        arg_parser.prepare_oudir_processing(out_dirs['model'], comm)
    comm.Barrier()

    # Load PV and EDPVR data
    time, pres, vols = load_pv_data(out_dirs['calib'])
    edpvr_pres, edpvr_vols_unc = load_edpvr(out_dirs['pv'])
    a = settings['PV']['calibration']['a']
    b = settings['PV']['calibration']['b']
    edpvr_vols = a * edpvr_vols_unc + b

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
    if matparams is None:
        matparams = settings['matparams']
    else:
        matparams_default = settings['matparams']
        for key, value in matparams.items():
            matparams_default[key] = value
        matparams = matparams_default

    model = HeartModelDynaComp(
        geo=geometry,
        bc_params=bc_params,
        matparams=matparams,
        comm=comm,
    )

    if comm.rank == 0:
        logger.info("Current material paramters", a=model.material.parameters['a'], a_f=model.material.parameters['a_f'])

    # Set up data collector with live plotting
    collector = DataCollectorInflator(
        outdir=out_dirs['model'],
        problem=model,
        pv_vols=vols,
        pv_pres=pres,
        edpvr_vols=edpvr_vols,
        edpvr_pres=edpvr_pres,
        live_plot=live_plot
    )

    # Run inflation steps
    v0 = model.compute_volume(activation_value=0, pressure_value=0)
    collector.collect(time=0, pressure=0, volume=v0)

    pressures = np.linspace(0, pres[0] * pressure_multiplier, pressure_steps)
    for i, p in enumerate(pressures, start=1):
        model.compute_volume(activation_value=0, pressure_value=p)
        collector.collect(time=i, pressure=model.get_pressure(), volume=model.get_volume())

    if comm.rank == 0:
        collector.finalize_plot()

    return collector

def run_inflator(
    model: HeartModelDynaComp,
    pressures: np.array,
    comm: dolfin.MPI.comm_world
) -> np.array:
    """
    Run an inflation simulation using explicit parameters.

    Parameters:
        model: HeartModelDynaComp instance,
        pressures: pressure values for inflation steps,
        comm: MPI communicator

    Returns:
        pressure and volumes
    """

    # Run inflation steps
    res_pres = []
    res_vols = []
    for i, p in enumerate(pressures):
        v = model.compute_volume(activation_value=0, pressure_value=p, logging_flag=False)
        res_pres.append(p)
        res_vols.append(v)
        if comm.rank == 0:
            logger.info(f"Inflation step {i}: ", pressure=round(p,3), volume=round(v,3))

    return res_pres, res_vols

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

    # Determine sample list
    if args.number:
        sample_list = args.number
    else:
        settings_files = sorted([f for f in args.settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(settings_files) + 1))

    # Run inflator for each sample with explicit params
    for sample in sample_list:
        run_inflator_with_collector(
            sample,
            settings_dir=args.settings_dir,
            results_dir=args.results_dir,
            scan_type=args.scan_type,
            pressure_multiplier=args.pressure_multiplier,
            pressure_steps=args.pressure_steps,
            pericardium_spring=args.pericardium_spring,
            base_spring=args.base_spring,
            live_plot=True
        )

if __name__ == '__main__':
    main()