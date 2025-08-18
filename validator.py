import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


import pulse
import dolfin

from processing import load_edpvr_results
import utils

from structlog import get_logger

logger = get_logger()
# %%
def get_infaltion_time(modeling_dir):
    # find the duration of the inflation phase, considering that during this time the activation is exactly zero
    data = np.loadtxt(modeling_dir / "results_data.csv", delimiter=',', skiprows=1)
    ind = np.where(abs(data[:,1])>0)[0][0] - 1
    return ind



def load_mesh_from_file(mesh_fname: Path):
    # Read the mesh
    mesh_fname = Path(mesh_fname)
    with dolfin.XDMFFile(mesh_fname.as_posix()) as xdmf:
        mesh = dolfin.Mesh()
        xdmf.read(mesh)
    return mesh

def load_displacement_function_from_file(
    displacement_fname: Path, t: float, mesh: dolfin.mesh
):
    displacement_fname = Path(displacement_fname)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    u = dolfin.Function(V)
    with dolfin.XDMFFile(displacement_fname.as_posix()) as xdmf:
        xdmf.read_checkpoint(u, "Displacement", t)
    return u

def copy_facet_markers_to_mesh(src_ffun: dolfin.cpp.mesh.MeshFunctionSizet,
                               dst_mesh: dolfin.Mesh) -> dolfin.MeshFunction:
    """Clone a MeshFunction('size_t') of facets from src mesh to an identical dst mesh."""
    # Create a destination facet function on the dst_mesh
    dst_ffun = dolfin.MeshFunction("size_t", dst_mesh, dst_mesh.topology().dim()-1, 0)
    # Copy by facet index (works because dst_mesh is an identical copy of the src mesh)
    dst_ffun.array()[:] = src_ffun.array()
    return dst_ffun

def plot_facet_tag_to_html(mesh: dolfin.Mesh,
                           ffun: dolfin.MeshFunction,
                           tag: int,
                           out_html: Path,
                           name: str = 'unnamed',
                           color: str = 'blue',
                           title: str = "Facet tag surface"):
    """
    Build a Plotly Mesh3d for all facets with ffun == tag and write to HTML.
    Assumes a 3D tetrahedral mesh (triangular facets).
    """
    mesh.init(2, 0)  # ensure facet->vertex connectivity

    coords = mesh.coordinates()
    gdim = mesh.geometry().dim()
    assert gdim == 3, "This routine expects a 3D mesh."

    # Collect the triangles (vertex indices) for facets with the given tag
    tris = []
    vert_used = set()

    # Iterate facets
    for f in dolfin.facets(mesh):
        if ffun[f.index()] == tag:
            vs = f.entities(0)  # vertex indices of this facet
            if len(vs) != 3:
                # If not triangular, skip (Plotly Mesh3d expects triangles)
                continue
            tris.append(tuple(vs))
            vert_used.update(vs)

    if not tris:
        raise RuntimeError(f"No facets found with tag {tag}")

    # Reindex vertices to a compact 0..N-1 set for Plotly
    vert_used = sorted(vert_used)
    global_to_local = {g: i for i, g in enumerate(vert_used)}

    x = [coords[g][0] for g in vert_used]
    y = [coords[g][1] for g in vert_used]
    z = [coords[g][2] for g in vert_used]

    i = [global_to_local[a] for (a, b, c) in tris]
    j = [global_to_local[b] for (a, b, c) in tris]
    k = [global_to_local[c] for (a, b, c) in tris]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=i, j=j, k=k,
                opacity=0.5,
                flatshading=True,
                name=name,
                color=color,  
                # Do not set colors explicitly (lets Plotly pick)
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn")

# %%
def main(args=None) -> int:
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="The sample number(s), will process all the sample if not indicated",
    )

    parser.add_argument(
        "-i",
        "--ID",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
    )

    parser.add_argument(
        "--scan_type",
        default='TPM',
        type=str,
        help="The scan type. Settings will be loaded accordingly from json file",
    )

    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )
    args = parser.parse_args(args)

    sample_num = args.number
    sample_ID = args.ID
    scan_type = args.scan_type
    settings_dir = args.settings_dir
    results_dir = args.results_dir

    if sample_ID is not None:
        sample_num = utils.get_num_from_id(sample_ID, settings_dir)

    settings = utils.load_settings(settings_dir, sample_num)
    sample_name = settings["id"]

    sample_dir = Path(results_dir) / sample_name / scan_type
    pv_dir = sample_dir / "01_PVCalibration"
    geo_dir = pv_dir / "Geometries"
    edpvr_dir = sample_dir / "02_EDPVR_Modeling_v2"
    modeling_dir = sample_dir / "03_Active_Modeling"

    pressures, volumes = utils.load_pressure_volumes(pv_dir)
    peak_sys_ind = np.where(pressures == np.max(pressures))[0][0]
    inflation_time = get_infaltion_time(modeling_dir) 

    peak_sys_ind_simulation = peak_sys_ind + inflation_time
    # Load Simulation peak systole geometry
    unloaded_geometry_fname, _, _ = load_edpvr_results(edpvr_dir)
    unloaded_geometry = pulse.HeartGeometry.from_file(
        unloaded_geometry_fname.as_posix()
    )
    unloaded_mesh = unloaded_geometry.mesh
    
    displacement_fname = modeling_dir / "displacement.xdmf"
    u = load_displacement_function_from_file(displacement_fname, peak_sys_ind_simulation, unloaded_geometry.mesh)
    u.set_allow_extrapolation(True)
    # Make a copy of the mesh 
    peak_sys_mesh = dolfin.Mesh(unloaded_mesh)
    ffun_peak = copy_facet_markers_to_mesh(unloaded_geometry.ffun, peak_sys_mesh)
    # Apply displacement
    dolfin.ALE.move(peak_sys_mesh, u, )


    #Loading peak_systole mesh from MRI data
    mri_peak_sys_mesh_fname = geo_dir / f"geometry_{peak_sys_ind}.h5"
    mri_geometry = pulse.HeartGeometry.from_file(
        mri_peak_sys_mesh_fname.as_posix()
    )
    mri_mesh = mri_geometry.mesh

    out_html = sample_dir / "peak_sys_ffun.html"
    plot_facet_tag_to_html(
        mesh=peak_sys_mesh,
        ffun=ffun_peak,
        tag=7,
        out_html=out_html,
        name="Epi",
        color="blue",
    )


if __name__ == "__main__":
    main()