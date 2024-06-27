# %%
from pathlib import Path
import plotly.graph_objects as go


from ventric_mesh.create_mesh import read_data_h5
import ventric_mesh.mesh_utils as mu
from ventric_mesh.mesh_utils import (
    check_mesh_quality,
    generate_3d_mesh_from_stl,
)
import matplotlib.pyplot as plt


# %%
def create_mesh(directory_path, mesh_settings, h5_file, plot_flag=True):
    mask, T_array, slice_thickness, resolution = read_data_h5(h5_file)
    mask_epi, mask_endo = mu.get_endo_epi(mask)

    tck_epi = mu.get_shax_from_mask(
        mask_epi, resolution, slice_thickness, mesh_settings["smooth_level_epi"]
    )
    tck_endo = mu.get_shax_from_mask(
        mask_endo, resolution, slice_thickness, mesh_settings["smooth_level_endo"]
    )

    T = len(tck_epi[0])
    K = len(tck_epi[0][0])
    K_endo = len(tck_endo[0][0])
    if plot_flag:
        outdir = directory_path / "02_ShaxBSpline"
        outdir.mkdir(exist_ok=True)
        for t in range(T):
            for k in range(K):
                mu.plot_shax_with_coords(
                    mask_epi, tck_epi, t, k, resolution, new_plot=True
                )
                if k < K_endo:
                    mu.plot_shax_with_coords(
                        mask_endo, tck_endo, t, k, resolution, color="b"
                    )
                fnmae = outdir.as_posix() + "/" + str(t) + "_" + str(k) + ".png"
                plt.savefig(fnmae)
                plt.close()

    sample_points_epi = mu.get_sample_points_from_shax(
        tck_epi, mesh_settings["num_lax_points"]
    )
    sample_points_endo = mu.get_sample_points_from_shax(
        tck_endo, mesh_settings["num_lax_points"]
    )

    apex_threshold = mu.get_apex_threshold(sample_points_epi, sample_points_endo)
    LAX_points_epi, apex_epi = mu.create_lax_points(
        sample_points_epi, apex_threshold, slice_thickness
    )
    LAX_points_endo, apex_endo = mu.create_lax_points(
        sample_points_endo, apex_threshold, slice_thickness
    )
    tck_lax_epi = mu.get_lax_from_laxpoints(
        LAX_points_epi, mesh_settings["lax_smooth_level_epi"]
    )
    tck_lax_endo = mu.get_lax_from_laxpoints(
        LAX_points_endo, mesh_settings["lax_smooth_level_endo"]
    )
    if plot_flag:
        outdir = directory_path / "03_LaxBSpline"
        outdir.mkdir(exist_ok=True)
        for t in range(T):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            mu.plot_3d_LAX(
                ax,
                t,
                range(int(mesh_settings["num_lax_points"] / 2)),
                tck_lax_epi,
                tck_endo=tck_lax_endo,
            )
            fnmae = outdir.as_posix() + "/" + str(t) + "_" + str(k) + ".png"
            plt.savefig(fnmae)
            plt.close()

    tck_shax_epi = mu.get_shax_from_lax(
        tck_lax_epi,
        apex_epi,
        mesh_settings["num_z_sections_epi"],
        mesh_settings["z_sections_flag_epi"],
    )
    tck_shax_endo = mu.get_shax_from_lax(
        tck_lax_endo,
        apex_endo,
        mesh_settings["num_z_sections_endo"],
        mesh_settings["z_sections_flag_endo"],
    )
    if plot_flag:
        outdir = directory_path / "04_Contours"
        outdir.mkdir(exist_ok=True)
        for t in range(T):
            fig = go.Figure()
            fig = mu.plotly_3d_contours(
                fig, t, tck_shax_epi, tck_lax_epi, tck_shax_endo, tck_lax_endo
            )
            fnmae = outdir.as_posix() + "/" + str(t) + ".html"
            fig.write_html(fnmae)

    points_cloud_epi = mu.create_point_cloud(
        tck_shax_epi,
        apex_epi,
        mesh_settings["seed_num_base_epi"],
        seed_num_threshold=mesh_settings["seed_num_threshold_epi"],
    )
    points_cloud_endo = mu.create_point_cloud(
        tck_shax_endo,
        apex_endo,
        mesh_settings["seed_num_base_endo"],
        seed_num_threshold=mesh_settings["seed_num_threshold_endo"],
    )
    if plot_flag:
        outdir = directory_path / "05_Point Cloud"
        outdir.mkdir(exist_ok=True)
        for t in range(T):
            fig = go.Figure()
            for points in points_cloud_epi[t]:
                mu.plot_3d_points_on_figure(points, fig=fig)
            fnmae = outdir.as_posix() + "/" + str(t) + "_epi.html"
            fig.write_html(fnmae)
            fig = go.Figure()
            for points in points_cloud_endo[t]:
                mu.plot_3d_points_on_figure(points, fig=fig)
            fnmae = outdir.as_posix() + "/" + str(t) + "_endo.html"
            fig.write_html(fnmae)

    outdir = directory_path / "06_Mesh/"
    outdir.mkdir(exist_ok=True)
    LVmesh = mu.VentricMesh(
        points_cloud_epi,
        points_cloud_endo,
        mesh_settings["t_mesh"],
        mesh_settings["num_mid_layers_base"],
        save_flag=True,
        result_folder=outdir.as_posix() + "/",
    )

    check_mesh_quality(LVmesh)
    stl_path = outdir.as_posix() + "/Mesh.stl"
    mesh_3d_path = outdir.as_posix() + "/Mesh_3D.msh"
    generate_3d_mesh_from_stl(stl_path, mesh_3d_path)

    return LVmesh, outdir
