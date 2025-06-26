from pathlib import Path
import json

import arg_parser
import mesh_utils
import meshing
import create_geometry

from structlog import get_logger

logger = get_logger()


# %%
def load_settings(setting_dir, sample_num):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    settings_fname = sorted_files[sample_num - 1]
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")



# %%
def main(args=None) -> int:
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_pre(args)
    else:
        args = arg_parser.update_arguments(args)

    sample_num = args.number
    sample_ID = args.ID
    setting_dir = args.settings_dir
    data_dir = args.data_dir
    results_dir = args.results_dir
    time_mesh = args.time_mesh
    scan_type = args.scan_type
    mesh_quality = args.mesh_quality
    h5_overwrite = args.h5_overwrite

    if sample_ID is not None:
        sample_num = get_num_from_id(sample_ID, setting_dir)

    settings = load_settings(setting_dir, sample_num)
    sample_name = settings["id"]
    logger.info(f"Loaded settings from {sample_name}")
    mesh_settings = settings[scan_type]["mesh"][mesh_quality]

    sample_dir = data_dir / sample_name / scan_type
    # creating the output folder
    if time_mesh is None:
        time_mesh = range(0, 100)

    for t in time_mesh:
        output_dir = Path(results_dir) / sample_name / scan_type / "00_Meshes" / f"time_{t}"
        output_dir = arg_parser.prepare_outdir(output_dir)
        mesh_settings["t_mesh"] = t

        if t == time_mesh[0]:
            # Creating the mesh settings
            h5_file = mesh_utils.compile_h5(
                sample_dir,
                scan_type,
                overwrite=h5_overwrite,
                is_inverted=settings[scan_type]["is_inverted"],
            )
            h5_file = mesh_utils.prepare_h5_files(scan_type, h5_file, output_dir, settings)

        mesh_fname = meshing.create_mesh(
            data_dir,
            scan_type,
            mesh_settings,
            h5_file,
            plot_flag=True,
            output_dir=output_dir,
        )
        if mesh_fname is None:
            arg_parser.prepare_outdir(output_dir)
            output_dir.rmdir()  
            break
        geometry = create_geometry.create_geometry(mesh_fname, fiber_angles=settings["fiber_angles"], plot_flag=True)

        geo_outdir = output_dir / "Geometry"
        geo_fname = geo_outdir / "geometry"
        geometry.save(geo_fname.as_posix(), overwrite_file=True)
        logger.info(f"=========== Mesh generated for {sample_name} at t={t} =========")
        logger.info(f"=======================================================")

if __name__ == "__main__":
    main()
