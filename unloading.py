# %%
from pathlib import Path
from structlog import get_logger

from fenics_plotly import plot
import pulse

logger = get_logger()


# %%
def get_h5_fname(meshdir, h5_fname=None):
    if h5_fname is not None:
        return h5_fname
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


def unloader(meshdir, atrium_pressure=0.24, plot_flag=False):
    h5_fname = get_h5_fname(meshdir, h5_fname=None)
    geo = pulse.HeartGeometry.from_file(h5_fname)
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    marker_functions = pulse.MarkerFunctions(ffun=geo.ffun)
    geometry = pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        microstructure=microstructure,
        marker_functions=marker_functions,
    )

    material = pulse.NeoHookean(parameters=dict(mu=1.5))
    # Parameter for the cardiac boundary conditions
    bcs_parameters = pulse.MechanicsProblem.default_bcs_parameters()
    bcs_parameters["base_spring"] = 1.0
    bcs_parameters["base_bc"] = "fix_x"

    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs_parameters=bcs_parameters)

    # Suppose geometry is loaded with a pressure of 1.776 mmHg (0.24kPa) based on PV loop of D3-2
    # and create the unloader
    unloader = pulse.FixedPointUnloader(problem=problem, pressure=atrium_pressure)

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

    return unloaded_geometry
