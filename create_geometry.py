#%%
from fenics_plotly import plot
import pulse
import meshio
import numpy as np
import meshio
import dolfin
from pathlib import Path
from structlog import get_logger

logger = get_logger()

#%%
dir = Path('00_data/AS/3week/156_1/')
meshdir = dir / '06_Mesh'
mesh_fname = None

# find the msh file in the meshdir
mesh_files = list(meshdir.glob('*.msh'))
if len(mesh_files)>1:
    logger.warning(f'There are {len(mesh_files)} mesh files in the folder. The first mesh "{mesh_files[0].as_posix()}" is being used. Otherwise, specify mesh_fname.')

if mesh_fname is None:
    mesh_fname = mesh_files[0].as_posix()
# Reading the gmsh file and create a xdmf to be read by dolfin
msh = meshio.read(mesh_fname)

# Find the indices for 'tetra' and 'triangle' cells

tetra_index = next(i for i, item in enumerate(msh.cells) if item.type == "tetra")
triangle_index = next(i for i, item in enumerate(msh.cells) if item.type == "triangle")

# Extract the corresponding cells
tetra_cells = msh.cells[tetra_index].data
triangle_cells = msh.cells[triangle_index].data

# Extract the corresponding cell data
triangle_cell_data = msh.cell_data["gmsh:geometrical"][triangle_index]

# Write the mesh and mesh function
fname = mesh_fname[:-4] + '.xdmf'
meshio.write(fname, meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells}))
# %%
#%% reading xdmf file and create pvd and initializing the mesh
mesh = dolfin.Mesh()
with dolfin.XDMFFile(fname) as infile:
    infile.read(mesh)
fname = mesh_fname[:-4] + '.pvd'
dolfin.File(fname).write(mesh)

#initialize the connectivity between facets and cells  
tdim = mesh.topology().dim()
fdim = tdim - 1
mesh.init(fdim, tdim)


#%% Creating the pulse geometry and setting ffun
geometry = pulse.HeartGeometry(mesh=mesh)
ffun = df.MeshFunction('size_t', mesh, 2)
ffun.set_all(0)

#%% Annotating the base mesh function
# we set the markers as base=5, endo=6, epi=7
# First we find all the exterior surface with z coords equal to 0 which corresponds to the base facets
# facet_exterior_all is the index of facets on the exterior surfaces and coord_exterior_all is the coordinates
facet_exterior_all=[]
# coord_exterior_all=[]
for fc in df.facets(geometry.mesh):
    if fc.exterior():
        facet_exterior_all.append(fc.index())
        # coord_exterior_all.append(geometry.mesh.coordinates()[fc.entities(0), 2])
        z_coords=np.mean(geometry.mesh.coordinates()[fc.entities(0), 2])
        if df.near(z_coords,0):
            ffun[fc]=5
        
# x_exterior_all = [np.mean(array) for array in coord_exterior_all]
# x_base = np.min(x_exterior_all)
# x_apex = np.max(x_exterior_all)

# for fc in df.facets(geometry.mesh):
#     x=np.mean(geometry.mesh.coordinates()[fc.entities(0), 0])
#     if df.near(x,x_base):
#         ffun[fc]=5


# plot(ffun,wireframe=True)
#%% Finding the exterior facets without the base for annotating the epi and endo
# facet_exterior is the index of facets on the exterior surfaces excluding the base and coord_exterior is the coordinates and nodes are a n*3 matrix of node numbers of each corresponding facet
facet_exterior=[]
# coord_exterior=[]
node_exterior=[]
for fc in df.facets(geometry.mesh):
    if fc.exterior() and not(df.near(ffun[fc],5)):
        facet_exterior.append(fc.index())
        # coord_exterior.append(geometry.mesh.coordinates()[fc.entities(0), 0])
        node_exterior.extend(fc.entities(0))

# Creating a dictionary (a graph in fact) to find all the connected facets with each other 
node_exterior = [node_exterior[i:i+3] for i in range(0, len(node_exterior), 3)]
graph = {fc: set() for fc in facet_exterior}       
for i, nodes_i in enumerate(node_exterior):
    for j, nodes_j in enumerate(node_exterior):
        if i != j and set(nodes_i).intersection(set(nodes_j)):
            graph[facet_exterior[i]].add(facet_exterior[j])       