# %%
import numpy as np
from pathlib import Path
from structlog import get_logger

from fenics_plotly import plot
import pulse
import dolfin
import logging
import warnings
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning  

warnings.filterwarnings("ignore", category=QuadratureRepresentationDeprecationWarning)


logger = get_logger()
# logging.getLogger("pulse").setLevel(logging.WARNING)

comm = dolfin.MPI.comm_world

# %%
def get_h5_fname(meshdir, h5_fname=None):
    if h5_fname is not None:
        return meshdir.as_posix() + '/' + h5_fname
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


def unloader(outdir, atrium_pressure=0.24, plot_flag=False, comm=None, h5_fname=None):
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
    
    fname = outdir / 'geometry_ffun.xdmf'
    if not fname.exists():
        with dolfin.XDMFFile(comm, fname.as_posix()) as f:
            f.write(geometry.mesh)

    matparams = dict(
            a=1.726,
            a_f=7.048,
            b=1.118,
            b_f=0.001,
            a_s=0.0,
            b_s=0.0,
            a_fs=0.0,
            b_fs=0.0,
        )

    material =  pulse.HolzapfelOgden(
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
        # Saving ffun

    return unloaded_geometry

#%%
directory_path = Path("00_data/AS/3week/156_1/")
results_folder = "00_Results"
fname = directory_path / "PV data/PV_data.csv"
PV_data = np.loadtxt(fname.as_posix() ,delimiter=',')
mmHg_to_kPa = 0.133322
atrium_pressure = PV_data[0,0] * mmHg_to_kPa

if results_folder is not None or not results_folder == "":
    results_folder_dir = directory_path / results_folder
    results_folder_dir.mkdir(exist_ok=True)
else:
    results_folder_dir = directory_path
    
outdir = results_folder_dir / "Geometry"

h5_fname='geometry.h5'
unloaded_geometry = unloader(outdir, atrium_pressure=atrium_pressure, plot_flag=False, comm=comm, h5_fname=h5_fname)

fname = outdir.as_posix() +  '/unloaded_geometry.h5'
unloaded_geometry.save(fname, overwrite_file=True)

fname = outdir.as_posix() +  '/unloaded_geometry_ffun.xdmf'
with dolfin.XDMFFile(comm, fname) as f:
    f.write(unloaded_geometry.mesh)

#%%