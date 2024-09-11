# %%
import numpy as np
from pathlib import Path
from structlog import get_logger
import json

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
def unloader(outdir, atrium_pressure, matparams, plot_flag=False, comm=None):
    if comm is None:
        comm = dolfin.MPI.comm_world

    h5_fname = outdir / "geometry.h5"
    logger.info(f"Original geometry loaded {h5_fname} ...")
    geo = pulse.HeartGeometry.from_file(h5_fname, comm=comm)
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    marker_functions = pulse.MarkerFunctions(ffun=geo.ffun)
    geometry = pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )

    ffun_fname = outdir / "geometry_ffun.xdmf"
    if not ffun_fname.exists():
        with dolfin.XDMFFile(comm, ffun_fname.as_posix()) as f:
            f.write(geometry.mesh)
    material = pulse.HolzapfelOgden(
        active_model="active_stress",
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    # Parameter for the cardiac boundary conditions
    bcs_parameters = pulse.MechanicsProblem.default_bcs_parameters()
    bcs_parameters["base_spring"] = 1.0
    bcs_parameters["base_bc"] = "fix_x"
    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs_parameters=bcs_parameters)

    # Suppose geometry is loaded with a pressure of 1.776 mmHg (0.24kPa) based on PV loop of D3-2
    # and create the unloader
    unloading_params = {
        "maxiter": 10,
        "tol": 1e-4,
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
    if plot_flag:
        fig = plot(geometry.mesh, opacity=0.0, show=False, wireframe=True)
        fig.add_plot(
            plot(unloaded_geometry.mesh, opacity=0.5, color="grey", show=False)
        )
        fig.show()
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
    fiber_space = "P_1"

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


def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


def load_atrium_pressure(data_dir):
    PV_data_fname = data_dir / "PV data/PV_data.csv"
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    atrium_pressure = PV_data[0, 0] * mmHg_to_kPa
    return atrium_pressure


def export_unloaded_geometry(geo_dir, unloaded_geometry_with_corrected_fibers):
    fname = geo_dir.as_posix() + "/unloaded_geometry_with_fibers.h5"
    unloaded_geometry_with_corrected_fibers.save(fname, overwrite_file=True)

    fname = geo_dir.as_posix() + "/unloaded_geometry_with_fibers_ffun.xdmf"
    with dolfin.XDMFFile(comm, fname) as f:
        f.write(unloaded_geometry_with_corrected_fibers.mesh)


# %%
def main(args=None) -> int:
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_unloading(args)
    else:
        args = arg_parser.update_arguments(args, step="unloading")

    sample_name = args.name
    setting_dir = args.settings_dir
    output_folder = args.output_folder

    settings = load_settings(setting_dir, sample_name)
    data_dir = Path(settings["path"])

    atrium_pressure = load_atrium_pressure(data_dir)

    geo_dir = data_dir / f"{output_folder}/Geometry"

    unloaded_geometry = unloader(
        geo_dir,
        atrium_pressure,
        matparams=settings["matparams"],
        plot_flag=True,
        comm=comm,
    )

    unloaded_geometry_with_corrected_fibers = recreate_geometry_with_fibers(
        unloaded_geometry, settings["fiber_angles"]
    )
    export_unloaded_geometry(geo_dir, unloaded_geometry_with_corrected_fibers)


if __name__ == "__main__":
    main()
