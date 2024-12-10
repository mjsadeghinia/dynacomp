# %%
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from structlog import get_logger
import logging

import meshio
import dolfin
import pulse
from fenics_plotly import plot
import ldrb

logger = get_logger()


# %%
def get_mesh_fname(meshdir, mesh_fname=None):
    meshdir = Path(meshdir)
    # find the msh file in the meshdir
    mesh_files = list(meshdir.glob("*.msh"))
    if len(mesh_files) > 1:
        logger.warning(
            f'There are {len(mesh_files)} mesh files in the folder. The first mesh "{mesh_files[0].as_posix()}" is being used. Otherwise, specify mesh_fname.'
        )

    if mesh_fname is None:
        mesh_fname = mesh_files[0].as_posix()
    return mesh_fname


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
    # Check if the default values are used
    if not fiber_angles:
        logger.warning("Default fiber angles are being used.")
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


def create_geometry(
    mesh_fname, fiber_angles: dict = None, plot_flag=False
):
    # mesh_fname = get_mesh_fname(meshdir, mesh_fname=mesh_fname)
    # Reading the gmsh file and create a xdmf to be read by dolfin
    msh = meshio.read(mesh_fname)
    # Find the indices for 'tetra' and 'triangle' cells
    tetra_index = next(i for i, item in enumerate(msh.cells) if item.type == "tetra")
    # Extract the corresponding cells
    tetra_cells = msh.cells[tetra_index].data
    # Write the mesh and mesh function
    fname = mesh_fname[:-4] + ".xdmf"
    meshio.write(fname, meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells}))

    # reading xdmf file and create pvd and initializing the mesh
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(fname) as infile:
        infile.read(mesh)
    fname = mesh_fname[:-4] + ".pvd"
    dolfin.File(fname).write(mesh)

    # initialize the connectivity between facets and cells
    tdim = mesh.topology().dim()
    fdim = tdim - 1
    mesh.init(fdim, tdim)

    # Creating the pulse geometry and setting ffun
    geometry = pulse.HeartGeometry(mesh=mesh)

    # Defining face function (ffun)
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    # Extract face indices from 'msh'
    epi_face_indices = msh.cells[0].data
    endo_face_indices = msh.cells[1].data
    base_face_indices = msh.cells[2].data

    # Get vertex coordinates
    vertex_coordinates = msh.points

    def triangle_key(coords, tol=1e-6):
        """
        Generates a hashable key for a triangle's coordinates.
        """
        rounded_coords = np.round(coords / tol).astype(int)
        sorted_coords = np.sort(rounded_coords, axis=0)
        key = tuple(sorted_coords.flatten())
        return key

    def build_face_keys(face_indices, vertex_coordinates, tol=1e-6):
        """
        Builds a set of unique keys for a group of faces.
        """
        keys = set()
        for indices in face_indices:
            coords = vertex_coordinates[indices]
            key = triangle_key(coords, tol=tol)
            keys.add(key)
        return keys

    # Build sets of keys for each face group
    epi_keys = build_face_keys(epi_face_indices, vertex_coordinates)
    endo_keys = build_face_keys(endo_face_indices, vertex_coordinates)
    base_keys = build_face_keys(base_face_indices, vertex_coordinates)
    # Annotate the mesh function using the keys
    for fc in dolfin.facets(mesh):
        if fc.exterior():
            coord = mesh.coordinates()[fc.entities(0)]
            key = triangle_key(coord)
            if key in epi_keys:
                ffun[fc] = 7
            elif key in endo_keys:
                ffun[fc] = 6
            elif key in base_keys:
                ffun[fc] = 5

    if plot_flag:
        fname = mesh_fname[:-4] + "_plotly.html"
        # plotting the face function
        fig = plot(ffun, wireframe=True, show=False)
        fig.save(fname)

    # Saving ffun
    fname = mesh_fname[:-4] + "_ffun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(ffun)

    logger.info("Creating a pulse geometry...")

    marker_functions = pulse.MarkerFunctions(ffun=ffun)
    markers = {"BASE": [5, 2], "ENDO": [6, 2], "EPI": [7, 2]}
    geometry = pulse.HeartGeometry(
        mesh=geometry.mesh, markers=markers, marker_functions=marker_functions
    )
    #
    # Decide on the angles you want to use
    angles = get_fiber_angles(fiber_angles)

    # Convert markers to correct format
    markers = {
        "base": geometry.markers["BASE"][0],
        "lv": geometry.markers["ENDO"][0],
        "epi": geometry.markers["EPI"][0],
    }
    # Choose space for the fiber fields
    # This is a string on the form {family}_{degree}
    fiber_space = "DG_0"

    # Compute the microstructure
    logger.info("Computing fiber angles...")
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=geometry.mesh,
        fiber_space=fiber_space,
        ffun=geometry.ffun,
        markers=markers,
        log_level=30,
        **angles,
    )
    fname = mesh_fname[:-4] + "_fiber"

    ldrb.fiber_to_xdmf(fiber, fname)

    pulse_logger = logging.getLogger("pulse")
    pulse_logger.setLevel(logging.WARNING)
    geometry.microstructure = pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)
    fname = mesh_fname[:-4]
    geometry.save(fname, overwrite_file=True)
    return geometry
