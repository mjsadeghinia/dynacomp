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
