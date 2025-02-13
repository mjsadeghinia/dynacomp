# %%
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

import mesh_utils
from ventric_mesh.create_mesh import read_data_h5
import ventric_mesh.mesh_utils as mu
import ventric_mesh.utils as ventric_utils
import matplotlib.pyplot as plt
from structlog import get_logger

logger = get_logger()

# %%
def create_mesh(
    directory_path, scan_type, mesh_settings, h5_file, plot_flag=True, results_folder="00_Results", fname_prefix = "0"
):
    if scan_type == "TPM":
        mask, T_array, slice_thickness, resolution, I = mesh_utils.read_data_h5_TPM(h5_file)
        mask_epi, mask_endo = mu.get_endo_epi(mask[:, :, :, mesh_settings["t_mesh"]])
        coords_epi = mu.get_coords_from_mask(mask_epi, resolution, slice_thickness)
        coords_endo = mu.get_coords_from_mask(mask_endo, resolution, slice_thickness)
    elif scan_type == "CINE":
        coords_endo,coords_epi,slice_thickness,resolution, I = mesh_utils.read_data_h5_CINE(h5_file)
        coords_epi = mesh_utils.transform_to_img_cs_for_all_slices(coords_epi, resolution, slice_thickness, I)
        coords_endo = mesh_utils.transform_to_img_cs_for_all_slices(coords_endo, resolution, slice_thickness,  I)
        if len(coords_endo) == len(coords_epi):
            coords_epi, coords_endo = mesh_utils.close_apex_coords(coords_epi, coords_endo)
    else:
        logger.error(f"The scan type should be either TPM or CINE now it is {scan_type}")
    tck_epi = mu.get_shax_from_coords(
            coords_epi, mesh_settings["smooth_level_epi"]
        )
    tck_endo = mu.get_shax_from_coords(
        coords_endo, mesh_settings["smooth_level_endo"]
    )
    K = len(tck_epi)
    if plot_flag:
        outdir = results_folder / "02_ShaxBSpline"
        outdir.mkdir(exist_ok=True)
        K_endo = len(tck_endo)
        for k in range(K):
            mu.plot_shax_with_coords(
                coords_epi, tck_epi, k, new_plot=True
            )
            if k < K_endo:
                mu.plot_shax_with_coords(
                    coords_endo, tck_endo, k, color="b"
                )
            fnmae = outdir.as_posix() + "/" + fname_prefix + "_" + str(k) + ".png"
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
        outdir = results_folder / "03_LaxBSpline"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        mu.plotly_3d_LAX(
            fig,
            range(int(mesh_settings["num_lax_points"] / 2)),
            tck_lax_epi,
            tck_endo=tck_lax_endo,
        )
        fnmae = outdir.as_posix() + "/" + fname_prefix + ".html"
        fig.write_html(fnmae)
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
        outdir = results_folder / "04_Contours"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        fig = mu.plotly_3d_contours(
            fig, tck_shax_epi, tck_lax_epi, tck_shax_endo, tck_lax_endo
        )
        fnmae = outdir.as_posix() + "/" + fname_prefix + ".html"
        fig.write_html(fnmae)
    points_cloud_epi, k_apex_epi = mu.create_point_cloud(
        tck_shax_epi,
        apex_epi,
        mesh_settings["seed_num_base_epi"],
        seed_num_threshold=mesh_settings["seed_num_threshold_epi"],
    )
    points_cloud_endo, k_apex_endo = mu.create_point_cloud(
        tck_shax_endo,
        apex_endo,
        mesh_settings["seed_num_base_endo"],
        seed_num_threshold=mesh_settings["seed_num_threshold_endo"],
    )
    if plot_flag:
        outdir = results_folder / "05_Point Cloud"
        outdir.mkdir(exist_ok=True)
        fig = go.Figure()
        for points in points_cloud_epi:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/" + fname_prefix + "_epi.html"
        fig.write_html(fnmae)
        fig = go.Figure()
        for points in points_cloud_endo:
            mu.plot_3d_points_on_figure(points, fig=fig)
        fnmae = outdir.as_posix() + "/" + fname_prefix + "_endo.html"
        fig.write_html(fnmae)
    
    
    # Calculate normals
    normals_list_endo = mu.calculate_normals(points_cloud_endo, k_apex_endo)
    normals_list_epi = mu.calculate_normals(points_cloud_epi, k_apex_epi)
    if directory_path.stem =='OP138_3':
        normals_list_epi = mu.calculate_normals(points_cloud_epi, k_apex_epi, base_ind=7)
    
    outdir = results_folder / "06_Mesh"
    outdir.mkdir(exist_ok=True)
    mesh_epi_filename, mesh_endo_filename, mesh_base_filename = mu.VentricMesh_poisson(
        points_cloud_epi,
        points_cloud_endo,
        mesh_settings["num_mid_layers_base"],
        SurfaceMeshSizeEpi=mesh_settings["SurfaceMeshSizeEpi"],
        SurfaceMeshSizeEndo=mesh_settings["SurfaceMeshSizeEndo"],
        normals_list_epi = normals_list_epi,
        normals_list_endo = normals_list_endo,
        save_flag=True,
        filename_suffix="",
        result_folder=outdir.as_posix() + "/",
    )
    output_mesh_filename = outdir / 'Mesh_3D.msh'
    mu.generate_3d_mesh_from_seperate_stl(mesh_epi_filename, mesh_endo_filename, mesh_base_filename, output_mesh_filename.as_posix(),  MeshSizeMin=mesh_settings["MeshSizeMin"], MeshSizeMax=mesh_settings["MeshSizeMax"])
    if plot_flag:
        fig = ventric_utils.plot_coords_and_mesh(coords_epi, coords_endo, mesh_epi_filename, mesh_endo_filename)
        fname = outdir.as_posix() + "/Mesh_vs_Coords.html"
        fig.write_html(fname)
        
        
    # making error report 
    errors_epi = ventric_utils.calculate_error_between_coords_and_mesh(coords_epi, mesh_epi_filename)
    errors_endo = ventric_utils.calculate_error_between_coords_and_mesh(coords_endo, mesh_endo_filename)
    
    all_errors = np.concatenate([errors_epi, errors_endo])
    xlim = (np.min(all_errors), np.max(all_errors))
    hist_epi, _ = np.histogram(errors_epi, bins=30)
    hist_endo, _ = np.histogram(errors_endo, bins=30)
    max_y = max(np.max(hist_epi), np.max(hist_endo))
    ylim = (0, max_y + max_y * 0.1)  # Add 10% padding for aesthetics

    fname_epi = outdir / "Epi_mesh_errors.png"
    fname_endo = outdir / "Endo_mesh_errors.png"

    ventric_utils.plot_error_histogram(
        errors=errors_epi,
        fname=fname_epi,
        color='red',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Epi', 
        resolution=resolution
    )

    ventric_utils.plot_error_histogram(
        errors=errors_endo,
        fname=fname_endo,
        color='blue',
        xlim=xlim,
        ylim=ylim,
        title_prefix='Endo', 
        resolution=resolution
    )
    fname_epi = fname_epi.as_posix()[:-4] + ".txt"
    ventric_utils.save_error_distribution_report(errors_epi,fname_epi, n_bins=10, surface_name="Epicardium", resolution=resolution)
    fname_endo = fname_endo.as_posix()[:-4] + ".txt"
    ventric_utils.save_error_distribution_report(errors_endo, fname_endo, n_bins=10,  surface_name="Endocardium", resolution=resolution)

    return output_mesh_filename.as_posix()

# %%
