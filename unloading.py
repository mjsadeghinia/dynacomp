# %%
import numpy as np
from pathlib import Path
from structlog import get_logger

from fenics_plotly import plot
import pulse
import dolfin
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning

warnings.filterwarnings("ignore", category=QuadratureRepresentationDeprecationWarning)


logger = get_logger()
# logging.getLogger("pulse").setLevel(logging.WARNING)

comm = dolfin.MPI.comm_world


# %%
def get_h5_fname(meshdir, h5_fname=None):
    if h5_fname is not None:
        return meshdir.as_posix() + "/" + h5_fname
    else:
        meshdir = Path(meshdir)
        # find the msh file in the meshdir
        h5_files = list(meshdir.glob("*.h5"))
        # Exclude files containing "fiber", "ffun", "cfun", "laplace"or  "microstructure"
        h5_files = [
            file
            for file in h5_files
            if not any(
                substr in file.name
                for substr in ["fiber", "ffun", "cfun", "laplace", "microstructure"]
            )
        ]

        if len(h5_files) > 1:
            logger.warning(
                f'There are {len(h5_files)} mesh files in the folder. The first mesh "{h5_files[0].as_posix()}" is being used. Otherwise, specify h5_fname.'
            )
        h5_fname = h5_files[0].as_posix()
        return h5_fname


def unloader(outdir, atrium_pressure, matparams, plot_flag=False, comm=None, h5_fname=None):
    if comm is None:
        comm = dolfin.MPI.comm_world

    h5_fname = get_h5_fname(outdir, h5_fname=h5_fname)
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

    fname = outdir / "geometry_ffun.xdmf"
    if not fname.exists():
        with dolfin.XDMFFile(comm, fname.as_posix()) as f:
            f.write(geometry.mesh)

    matparams = get_matparams(matparams)

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


def get_matparams(matparams: dict = dict()):
        # Use provided fiber_angles or default ones if not provided
        default_matparams = get_default_matparams()
        matparams = (
            {
                key: matparams.get(key, default_matparams[key])
                for key in default_matparams
            }
            if matparams
            else default_matparams
        )
        return matparams

def get_default_matparams():
    """
    Default material parameters for the left ventricle
    """
    return dict(
        a=10.726,
        a_f=7.048,
        b=2.118,
        b_f=0.001,
        a_s=0.0,
        b_s=0.0,
        a_fs=0.0,
        b_fs=0.0,
        )

# %%
def create_geometry(geo, fiber_angles):
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


def get_fiber_angles(fiber_angles):
    # Use provided fiber_angles or default ones if not provided
    default_fiber_angles = get_default_fiber_angles()
    fiber_angles = (
        {
            key: fiber_angles.get(key, default_fiber_angles[key])
            for key in default_fiber_angles
        }
        if fiber_angles
        else default_fiber_angles
    )
    return fiber_angles


def get_default_fiber_angles():
    """
    Default fiber angles parameter for the left ventricle
    """
    angles = dict(
        alpha_endo_lv=60,  # Fiber angle on the LV endocardium
        alpha_epi_lv=-60,  # Fiber angle on the LV epicardium
        beta_endo_lv=-15,  # Sheet angle on the LV endocardium
        beta_epi_lv=15,  # Sheet angle on the LV epicardium
    )
    return angles

def unloading(path, results_folder, fiber_angles: dict = None, matparams: dict = None, comm=None):

    directory_path = Path(path)
    fname = directory_path / "PV data/PV_data.csv"
    PV_data = np.loadtxt(fname.as_posix(), delimiter=",")
    mmHg_to_kPa = 0.133322
    atrium_pressure = PV_data[0, 0] * mmHg_to_kPa

    if results_folder is not None or not results_folder == "":
        results_folder_dir = directory_path / results_folder
        results_folder_dir.mkdir(exist_ok=True)
    else:
        results_folder_dir = directory_path

    outdir = results_folder_dir / "Geometry"

    h5_fname = "geometry.h5"
    unloaded_geometry = unloader(
        outdir,
        atrium_pressure,
        matparams=matparams,
        plot_flag=True,
        comm=comm,
        h5_fname=h5_fname,
    )

    fiber_angles = get_fiber_angles(fiber_angles)
    unloaded_geometry_with_corrected_fibers = create_geometry(
        unloaded_geometry, fiber_angles
    )
    fname = outdir.as_posix() + "/unloaded_geometry_with_fibers.h5"
    # if comm.Get_rank()==0:
    unloaded_geometry_with_corrected_fibers.save(fname, overwrite_file=True)

    fname = outdir.as_posix() + "/unloaded_geometry_with_fibers_ffun.xdmf"
    with dolfin.XDMFFile(comm, fname) as f:
        f.write(unloaded_geometry.mesh)
    
    return unloaded_geometry_with_corrected_fibers

