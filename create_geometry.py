# %%
import numpy as np
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

    # Find the indices for 'tetra' and 'triangle' cells

    tetra_index = next(i for i, item in enumerate(msh.cells) if item.type == "tetra")
    # triangle_index = next(
    #     i for i, item in enumerate(msh.cells) if item.type == "triangle"
    # )

    # Extract the corresponding cells
    tetra_cells = msh.cells[tetra_index].data
    # triangle_cells = msh.cells[triangle_index].data

    # Extract the corresponding cell data
    # triangle_cell_data = msh.cell_data["gmsh:geometrical"][triangle_index]

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

    # Annotating the base mesh function
    # we set the markers as base=5, endo=6, epi=7
    # First we find all the exterior surface with z coords equal to 0 which corresponds to the base facets
    # facet_exterior_all is the index of facets on the exterior surfaces and coord_exterior_all is the coordinates
    facet_exterior_all = []
    # coord_exterior_all=[]
    for fc in dolfin.facets(geometry.mesh):
        if fc.exterior():
            facet_exterior_all.append(fc.index())
            # coord_exterior_all.append(geometry.mesh.coordinates()[fc.entities(0), 2])
            z_coords = np.mean(geometry.mesh.coordinates()[fc.entities(0), 2])
            if dolfin.near(z_coords, 0):
                ffun[fc] = 5

    # Finding the exterior facets without the base for annotating the epi and endo
    # facet_exterior is the index of facets on the exterior surfaces excluding the base and coord_exterior is the coordinates and nodes are a n*3 matrix of node numbers of each corresponding facet
    facet_exterior = []
    node_exterior = []
    for fc in dolfin.facets(geometry.mesh):
        if fc.exterior() and not (dolfin.near(ffun[fc], 5)):
            facet_exterior.append(fc.index())
            node_exterior.extend(fc.entities(0))

    # Creating a dictionary (a graph in fact) to find all the connected facets with each other
    node_exterior = [node_exterior[i : i + 3] for i in range(0, len(node_exterior), 3)]
    graph = {fc: set() for fc in facet_exterior}
    for i, nodes_i in enumerate(node_exterior):
        for j, nodes_j in enumerate(node_exterior):
            if i != j and set(nodes_i).intersection(set(nodes_j)):
                graph[facet_exterior[i]].add(facet_exterior[j])

    #

    # we find a first set as facet_1, however we do not know if it is epi or endo
    facet_1 = set()
    facet_i = list(graph.keys())[0]
    dfs(graph, facet_i, facet_1)
    facet_exterior_set = set(facet_exterior)
    facet_2 = facet_exterior_set - facet_1

    # Determining the endo and epi based on area comparison

    facet_1_id = np.array(list(dolfin.facets(geometry.mesh)))[list(facet_1)]
    facet_2_id = np.array(list(dolfin.facets(geometry.mesh)))[list(facet_2)]
    all_cells = np.array(list(dolfin.cells(geometry.mesh)))

    f_to_c = mesh.topology()(fdim, tdim)
    c_to_f = mesh.topology()(tdim, fdim)

    facet_1_area = []
    for facet in facet_1_id:
        cell = all_cells[f_to_c(facet.index())[0]]
        local_facets = c_to_f(cell.index())
        local_index = np.flatnonzero(local_facets == facet.index())
        area = cell.facet_area(local_index)
        facet_1_area.append(area)

    facet_2_area = []
    for facet in facet_2_id:
        cell = all_cells[f_to_c(facet.index())[0]]
        local_facets = c_to_f(cell.index())
        local_index = np.flatnonzero(local_facets == facet.index())
        area = cell.facet_area(local_index)
        facet_2_area.append(area)

    if np.sum(facet_1_area) > np.sum(facet_2_area):
        facet_endo = facet_2
        facet_epi = facet_1
    else:
        facet_endo = facet_1
        facet_epi = facet_2

    facet_endo_id = np.array(list(dolfin.facets(geometry.mesh)))[list(facet_endo)]
    facet_epi_id = np.array(list(dolfin.facets(geometry.mesh)))[list(facet_epi)]
    for facet in facet_endo_id:
        ffun[facet] = 6
    for facet in facet_epi_id:
        ffun[facet] = 7
    if plot_flag:
        # plotting the face function
        plot(ffun, wireframe=True)

    # Saving ffun
    fname = mesh_fname[:-4] + "_ffun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(ffun)

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
    fiber_space = "P_1"

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
