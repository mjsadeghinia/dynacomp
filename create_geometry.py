# %%
import numpy as np
from scipy.spatial import KDTree
import itertools

from pathlib import Path
from structlog import get_logger

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


def dfs(graph, node, visited):
    visited.add(node)
    for neighbour in graph[node]:
        if neighbour not in visited:
            dfs(graph, neighbour, visited)


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


def create_geometry(
    meshdir, fiber_angles: dict = None, mesh_fname=None, plot_flag=False
):
    mesh_fname = get_mesh_fname(meshdir, mesh_fname=mesh_fname)
    # Reading the gmsh file and create a xdmf to be read by dolfin
    msh = meshio.read(mesh_fname)

    # Find the Epi, Endo and Base triangle indices
    Epi_triangles = msh.cell_sets_dict['Epi']['triangle']
    Endo_triangles = msh.cell_sets_dict['Endo']['triangle']
    Base_triangles = msh.cell_sets_dict['Base']['triangle']

    # Find the indices for 'tetra' and 'triangle' cells
    tetra_index = next(i for i, item in enumerate(msh.cells) if item.type == "tetra")
    # Extract the corresponding cells
    tetra_cells = msh.cells[tetra_index].data
    # Find the indices for 'triangle' cells (surface elements)
    triangle_index = next(i for i, item in enumerate(msh.cells) if item.type == "triangle")
    triangle_cells = msh.cells[triangle_index].data
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
    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)

    # Assuming msh.cells[i].data contains the indices of the vertices for each face
    epi_face_indices = msh.cells[0].data 
    endo_face_indices = msh.cells[1].data 
    base_face_indices = msh.cells[2].data  
    # msh.points contains the coordinates of each vertex
    vertex_coordinates = msh.points  

    # Get the coordinates for each face by indexing msh.points with face_indices
    epi_face_coordinates = vertex_coordinates[epi_face_indices] 
    endo_face_coordinates = vertex_coordinates[endo_face_indices]
    base_face_coordinates = vertex_coordinates[base_face_indices]

    def are_triangles_similar(tri1, tri2, tol=1e-6):
        """
        Checks whether two triangles (arrays of shape (3, 3)) contain the same set of points,
        regardless of order, within a specified tolerance.
        """
        for perm in itertools.permutations(range(3)):
            tri2_perm = tri2[list(perm)]
            if np.allclose(tri1, tri2_perm, atol=tol):
                return True
        return False

    def check_face_similarity(face, coord, tol=1e-6):
        """
        Checks if coord is similar to any set of 3 points in face.

        Parameters:
        - face: NumPy array of shape (n, 3, 3), where each face[i] is a set of 3 points.
        - coord: NumPy array of shape (3, 3), representing a set of 3 points.
        - tol: Numerical tolerance for floating-point comparisons.

        Returns:
        - True if coord is similar to any face[i] in face; False otherwise.
        """
        for i in range(face.shape[0]):
            if are_triangles_similar(face[i], coord, tol=tol):
                return True
        return False

    # Annotating the base mesh function
    # we set the markers as base=5, endo=6, epi=7
    # First we find all the exterior surface with z coords equal to 0 which corresponds to the base facets
    # facet_exterior_all is the index of facets on the exterior surfaces and coord_exterior_all is the coordinates
    # coord_exterior_all=[]
    for fc in dolfin.facets(geometry.mesh):
        if fc.exterior():
            idx = fc.index()
            #print(set(msh.point_data['gmsh:dim_tags'][:,1]))
            #print(mesh.coordinates()[fc.entities(0)])
            #breakpoint()
            coord = mesh.coordinates()[fc.entities(0)]
            if check_face_similarity(epi_face_coordinates, coord, tol=1e-6):
                ffun[fc] = 7
            elif check_face_similarity(endo_face_coordinates, coord, tol=1e-6):
                ffun[fc] = 6
            elif check_face_similarity(base_face_coordinates, coord, tol=1e-6):
                ffun[fc] = 5
#            if idx in set(Base_triangles):
#                ffun[fc] = 5
#            elif idx in set(Endo_triangles):
#                ffun[fc] = 6
#            elif idx in set(Epi_triangles):
#                ffun[idx] = 7
    if plot_flag:
        fname = mesh_fname[:-4] + "_plotly"
        # plotting the face function
        plot(ffun, wireframe=True, filename=fname)

    # Saving ffun
    fname = mesh_fname[:-4] + "_ffun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(ffun)
    breakpoint()
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
    fiber_space = "Quadrature_4"

    # Compute the microstructure
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=geometry.mesh,
        fiber_space=fiber_space,
        ffun=geometry.ffun,
        markers=markers,
        **angles,
    )
    fname = mesh_fname[:-4] + "_fiber"

    ldrb.fiber_to_xdmf(fiber, fname)

    geometry.microstructure = pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)
    fname = mesh_fname[:-4]
    geometry.save(fname, overwrite_file=True)
    return geometry
