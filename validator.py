import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from typing import Tuple, List



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

def facet_tag_trace(mesh: dolfin.Mesh,
                    ffun: dolfin.MeshFunction,
                    tag: int,
                    name: str,
                    color: str,
                    opacity: float = 0.5):
    """
    Return a Plotly Mesh3d trace for facets with ffun == tag.
    """
    mesh.init(2, 0)  # ensure facet->vertex connectivity

    coords = mesh.coordinates()
    gdim = mesh.geometry().dim()
    assert gdim == 3, "This routine expects a 3D mesh."

    tris = []
    vert_used = set()
    for f in dolfin.facets(mesh):
        if ffun[f.index()] == tag:
            vs = f.entities(0)
            if len(vs) != 3:
                continue
            tris.append(tuple(vs))
            vert_used.update(vs)

    if not tris:
        raise RuntimeError(f"No facets found with tag {tag}")

    vert_used = sorted(vert_used)
    global_to_local = {g: i for i, g in enumerate(vert_used)}

    x = [coords[g][0] for g in vert_used]
    y = [coords[g][1] for g in vert_used]
    z = [coords[g][2] for g in vert_used]

    i = [global_to_local[a] for (a, b, c) in tris]
    j = [global_to_local[b] for (a, b, c) in tris]
    k = [global_to_local[c] for (a, b, c) in tris]

    mesh_trace = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=opacity,
        flatshading=True,
        name=name,
        color=color,
        showlegend=True
    )

    # Wireframe (edges) overlay
    edge_x, edge_y, edge_z = [], [], []
    for (a, b, c) in tris:
        for u, v in [(a, b), (b, c), (c, a)]:
            edge_x += [coords[u][0], coords[v][0], None]
            edge_y += [coords[u][1], coords[v][1], None]
            edge_z += [coords[u][2], coords[v][2], None]
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color=color, width=1),
        name=f"{name} edges",
        showlegend=False
    )

    return mesh_trace, edge_trace
def surface_vertices(mesh: dolfin.Mesh, ffun: dolfin.MeshFunction, tag: int) -> Tuple[np.ndarray, List[int]]:
    """
    Return (coords_of_surface_vertices, list_of_global_vertex_ids) for facets with marker == tag.
    """
    mesh.init(2, 0)
    coords = mesh.coordinates()
    vert_ids = set()
    for f in dolfin.facets(mesh):
        if ffun[f.index()] == tag:
            vs = f.entities(0)
            if len(vs) == 3:
                vert_ids.update(vs)
    vert_ids = sorted(vert_ids)
    pts = coords[vert_ids]
    return pts, vert_ids

def surface_triangles(mesh: dolfin.Mesh, ffun: dolfin.MeshFunction, tag: int) -> np.ndarray:
    """
    Return triangles as Nx3x3 array of coordinates for facets with marker == tag.
    tri_coords[n] = [[ax,ay,az],[bx,by,bz],[cx,cy,cz]]
    """
    mesh.init(2, 0)
    coords = mesh.coordinates()
    tris = []
    for f in dolfin.facets(mesh):
        if ffun[f.index()] == tag:
            vs = f.entities(0)
            if len(vs) == 3:
                a, b, c = vs
                tris.append(np.vstack([coords[a], coords[b], coords[c]]))
    if not tris:
        raise RuntimeError(f"No triangular facets for tag {tag}")
    return np.asarray(tris)  # (N,3,3)

# --- geometry helpers: closest distance point->triangle ---
def point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> float:
    """
    Compute shortest distance from point p (3,) to triangle tri (3,3) with rows a,b,c.
    Robust algorithm adapted from Christer Ericson, “Real-Time Collision Detection”.
    """
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(ap)  # barycentric (1,0,0)

    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(bp)  # barycentric (0,1,0)

    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        proj = a + v * ab
        return np.linalg.norm(p - proj)  # edge AB

    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(cp)  # barycentric (0,0,1)

    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        proj = a + w * ac
        return np.linalg.norm(p - proj)  # edge AC

    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        proj = b + w * (c - b)
        return np.linalg.norm(p - proj)  # edge BC

    # Inside face region
    n = np.cross(ab, ac)
    n_norm2 = np.dot(n, n)
    if n_norm2 == 0.0:
        # Degenerate triangle; fallback to min distance to vertices
        return min(np.linalg.norm(ap), np.linalg.norm(bp), np.linalg.norm(cp))
    dist = abs(np.dot(ap, n)) / np.sqrt(n_norm2)
    return dist

def distances_points_to_surface(points: np.ndarray, tri_coords: np.ndarray) -> np.ndarray:
    """
    Brute-force distances from each point to the closest triangle in tri_coords.
    points: (M,3), tri_coords: (N,3,3)
    Returns: (M,) distances
    """
    M = points.shape[0]
    N = tri_coords.shape[0]
    d = np.empty(M, dtype=float)
    for i in range(M):
        p = points[i]
        # Compute distance to all triangles; take min
        # (If slow for your mesh size, we can add KD-tree accel later.)
        mind = np.inf
        for n in range(N):
            dist = point_triangle_distance(p, tri_coords[n])
            if dist < mind:
                mind = dist
        d[i] = mind
    return d

def plot_error_histogram(errors, fname, color, xlim=None, ylim=None, title_prefix=""):
    avg_error = np.mean(errors)
    std_error  = np.std(errors)
    line = f'{title_prefix} Error Distribution (Avg: {avg_error:.2f} ± {std_error:.2f})'

    plt.figure()
    plt.hist(errors, bins=30, edgecolor='black', color=color)
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title(line)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

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
        "-o",
        "--output_folder",
        default= "03_Active_Modeling",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        '-l',
        '--logging_flag',
        action='store_true',
        default='logging the results',
        help='Flag to indicate whether to log the results.'
        )
    
    args = parser.parse_args(args)

    sample_num = args.number
    sample_ID = args.ID
    scan_type = args.scan_type
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    output_folder = args.output_folder
    logging_flag = args.logging_flag

    if sample_ID is not None:
        sample_num = utils.get_num_from_id(sample_ID, settings_dir)

    settings = utils.load_settings(settings_dir, sample_num)
    sample_name = settings["id"]

    sample_dir = Path(results_dir) / sample_name / scan_type
    pv_dir = sample_dir / "01_PVCalibration"
    geo_dir = pv_dir / "Geometries"
    edpvr_dir = sample_dir / "02_EDPVR_Modeling_v2"
    modeling_dir = sample_dir / output_folder

    pressures, volumes = utils.load_pressure_volumes(pv_dir)
    peak_sys_ind = np.where(pressures == np.max(pressures))[0][0]
    inflation_time = get_infaltion_time(modeling_dir) 

    result_path = sample_dir / "03_Active_Modeling" / "results_data.csv"
    sample_data = np.loadtxt(result_path, delimiter=",", skiprows=1)

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
    ffun_mri = copy_facet_markers_to_mesh(mri_geometry.ffun, mri_mesh)

    # Traces for simulation mesh
    epi_mesh, epi_edges   = facet_tag_trace(peak_sys_mesh, ffun_peak, 7, name="Epi (sim)",  color="blue")
    endo_mesh, endo_edges = facet_tag_trace(peak_sys_mesh, ffun_peak, 6, name="Endo (sim)", color="red")

    # Traces for MRI mesh (both epi & endo in grey)
    epi_mri_mesh, epi_mri_edges   = facet_tag_trace(mri_mesh, ffun_mri, 7, name="Epi (MRI)",  color="grey", opacity=0.3)
    endo_mri_mesh, endo_mri_edges = facet_tag_trace(mri_mesh, ffun_mri, 6, name="Endo (MRI)", color="grey", opacity=0.3)

    # Combine all into one figure
    fig = go.Figure(data=[
        epi_mesh, epi_edges,
        endo_mesh, endo_edges,
        epi_mri_mesh, epi_mri_edges,
        endo_mri_mesh, endo_mri_edges
    ])

    fig.update_layout(
        title=f"{sample_name} — peak systole: Simulation vs MRI surfaces",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_html = modeling_dir / "peak_sys_ffun.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    # Extract sim surface nodes for each tag
    epi_pts_sim, _  = surface_vertices(peak_sys_mesh, ffun_peak, 7)
    endo_pts_sim, _ = surface_vertices(peak_sys_mesh, ffun_peak, 6)

    # Extract MRI surface triangles for each tag
    epi_tris_mri  = surface_triangles(mri_mesh, ffun_mri, 7)  # (N_e,3,3)
    endo_tris_mri = surface_triangles(mri_mesh, ffun_mri, 6)  # (N_i,3,3)

    # Compute distances: sim nodes -> MRI surfaces
    epi_dists  = distances_points_to_surface(epi_pts_sim,  epi_tris_mri)
    endo_dists = distances_points_to_surface(endo_pts_sim, endo_tris_mri)
    hist_epi_png  = modeling_dir / "hist_epi.png"
    hist_endo_png = modeling_dir / "hist_endo.png"
    plot_error_histogram(
        epi_dists, hist_epi_png, color="blue",
        xlim=(0, np.max(epi_dists)*1.05), ylim=None,
        title_prefix="Epi (ffun=7)"
    )
    plot_error_histogram(
        endo_dists, hist_endo_png, color="red",
        xlim=(0, np.max(endo_dists)*1.05), ylim=None,
        title_prefix="Endo (ffun=6)"
    )
    
    if logging_flag:
        epi_avg_error = np.mean(epi_dists)
        epi_std_error  = np.std(epi_dists)
        endo_avg_error = np.mean(endo_dists)
        endo_std_error  = np.std(endo_dists)
        total_avg_error = np.mean(np.concatenate([epi_dists, endo_dists]))
        total_std_error  = np.std(np.concatenate([epi_dists, endo_dists]))
        # Save results to a file
        fname = modeling_dir.parent / f"Fiber_results.csv"
        if not fname.exists():
            header = "a, a_f, epi_fib, endo_fib, maximum Activation (kPa), epi_distance (mean), epi_distance (STD), endo_distance (mean), endo_distance(STD), total_distance (mean), total_distance (STD)\n"
            fname.write_text(header, encoding="utf-8")
        with fname.open("a", encoding="utf-8") as f:
            f.write(f"{settings['matparams']['a']},"
                    f"{settings['matparams']['a_f']},"
                    f"{settings['fiber_angles']['alpha_epi_lv']},"
                    f"{settings['fiber_angles']['alpha_endo_lv']},"
                    f"{np.max(sample_data[:,1])},"
                    f"{epi_avg_error},"
                    f"{epi_std_error},"
                    f"{endo_avg_error},"
                    f"{endo_std_error},"
                    f"{total_avg_error},"
                    f"{total_std_error}\n")

if __name__ == "__main__":
    main()