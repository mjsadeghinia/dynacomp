# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
import json
import shutil

import arg_parser
from fenics_plotly import plot
import pulse
import dolfin
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.filterwarnings("ignore", category=QuadratureRepresentationDeprecationWarning)


logger = get_logger()
comm = dolfin.MPI.comm_world


# %%
def unloader(outdir, geo_fname, atrium_pressure, matparams, bcs_parameters,  plot_flag=False, comm=None):
    if comm is None:
        comm = dolfin.MPI.comm_world
    geo = pulse.HeartGeometry.from_file(geo_fname, comm=comm)
    if comm.Get_rank() == 0:
        logger.info(f"Original geometry loaded {geo_fname} ...")
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    marker_functions = pulse.MarkerFunctions(ffun=geo.ffun)
    geometry = pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )

    # ffun_fname = outdir / f"{geo_fname}_ffun.xdmf"
    # if not ffun_fname.exists():
    #     with dolfin.XDMFFile(comm, ffun_fname.as_posix()) as f:
    #         f.write(geometry.mesh)
    material = pulse.HolzapfelOgden(
        active_model="active_stress",
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # Parameter for the cardiac boundary conditions
    bcs_parameters = bcs_parameters
    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs_parameters=bcs_parameters)

    # Suppose geometry is loaded with a pressure of 1.776 mmHg (0.24kPa) based on PV loop of D3-2
    # and create the unloader
    unloading_params = {
        "maxiter": 10,
        "tol": 1e-2,
        "lb": 0.5,
        "ub": 2.0,
        "regen_fibers": False,
        "solve_tries": 20,
    }
    unloader = pulse.FixedPointUnloader(
        problem=problem, pressure=atrium_pressure, options=unloading_params
    )

    # Unload the geometry
    unloader.unload()

    # Get the unloaded geometry
    unloaded_geometry = unloader.unloaded_geometry
    # if plot_flag:
    #     fig = plot(geometry.mesh, opacity=0.0, show=False, wireframe=True)
    #     fig.add_plot(
    #         plot(unloaded_geometry.mesh, opacity=0.5, color="grey", show=False)
    #     )
    #     fig.show()
        # Saving ffun

    return unloaded_geometry


def recreate_geometry_with_fibers(geo, fiber_angles):
    import ldrb

    # Convert markers to correct format
    markers = {
        "base": geo.markers["BASE"][0],
        "lv": geo.markers["ENDO"][0],
        "epi": geo.markers["EPI"][0],
    }
    # Choose space for the fiber fields
    # This is a string on the form {family}_{degree}
    fiber_space = "Quadrature_4"

    # Compute the microstructure
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=geo.mesh,
        fiber_space=fiber_space,
        ffun=geo.ffun,
        markers=markers,
        **fiber_angles,
    )
    if comm.Get_rank() == 1:
        logger.info("---------- Fibers regenerated ----------")

    microstructure = pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)
    marker_functions = pulse.MarkerFunctions(ffun=geo.ffun)

    return pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )


def load_settings(settings_dir, sample_num):
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    settings_fname = sorted_files[sample_num - 1]
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")

def load_atrium_pressure(pv_dir):
    fname = pv_dir / "calibrated_pv_data.csv"
    PV_data = np.loadtxt(fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    atrium_pressure = PV_data[0, 1] * mmHg_to_kPa
    return atrium_pressure


def export_unloaded_geometry(geo_dir, unloaded_geometry_with_corrected_fibers):
    geo_fname = 'geometry_0'
    fname = geo_dir.as_posix() + f"/unloaded_{geo_fname}_with_fibers.h5"
    unloaded_geometry_with_corrected_fibers.save(fname, overwrite_file=True)

    fname = geo_dir.as_posix() + f"/unloaded_{geo_fname}_with_fibers_ffun.xdmf"
    with dolfin.XDMFFile(comm, fname) as f:
        f.write(unloaded_geometry_with_corrected_fibers.mesh)


# %%
def main(args=None) -> int:
    comm = dolfin.MPI.comm_world
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_unloading(args)
    else:
        args = arg_parser.update_arguments(args, step="unloading")

    number = args.number
    sample_ID = args.ID
    bcs_parameters = arg_parser.create_bc_params(args)
    settings_dir = args.settings_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    scan_type = args.scan_type
    mesh_quality = args.mesh_quality
    output_folder = args.output_folder
    input_matparams = arg_parser.prepare_matparams(args)
    
    if sample_ID is not None:
        sample_nums = []
        for id in sample_ID:
            id_num = get_num_from_id(id, settings_dir)
            sample_nums.append(id_num)
    elif number:
        sample_nums = number
    else:
        settings_files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(settings_files) + 1))

    # Run inflation for each sample
    for sample_num in sample_nums:
        settings = load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        sample_dir = results_dir / sample_name / scan_type
        output_dir = sample_dir / output_folder
        output_dir.mkdir(parents=True, exist_ok=True)

        if not sample_dir.exists():
            continue
        pv_dir = sample_dir / "01_PVCalibration/"
        geo_dir = pv_dir / "Geometries"
        if not geo_dir.exists():
            logger.warning(f"Geometries not found for {sample_name}")
            continue
        
        atrium_pressure = load_atrium_pressure(pv_dir)
        logger.info(f"Sample {sample_name} atrium pressure: {atrium_pressure:.2f} kPa")
        geo_fname = geo_dir / "geometry_0.h5"

        # Set material properties
        matparams = settings['matparams']
        for key, value in input_matparams.items():
            matparams[key] = value

        unloaded_geometry = unloader(
            output_dir,
            geo_fname,
            atrium_pressure,
            matparams=matparams,
            bcs_parameters=bcs_parameters,
            plot_flag=True,
            comm=comm,
        )

        unloaded_geometry_with_corrected_fibers = recreate_geometry_with_fibers(
            unloaded_geometry, settings["fiber_angles"]
        )
        export_unloaded_geometry(output_dir, unloaded_geometry_with_corrected_fibers)

if __name__ == "__main__":
    main()
