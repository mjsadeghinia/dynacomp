from pathlib import Path
import json

import arg_parser
import mesh_utils
import meshing
import create_geometry


# %%
def load_settings(setting_dir, sample_name):
    settings_fname = setting_dir / f"{sample_name}.json"
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings

def get_sample_name(sample_num, setting_dir):
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted(
        [
            file
            for file in setting_dir.iterdir()
            if file.is_file() and file.suffix == ".json"
        ]
    )
    sample_name = sorted_files[sample_num - 1].with_suffix("").name
    return sample_name

# %%
def main(args=None) -> int:
    # Getting the arguments
    if args is None:
        args = arg_parser.parse_arguments_pre(args)
    else:
        args = arg_parser.update_arguments(args)

    sample_num = args.number
    setting_dir = args.settings_dir
    sample_name = get_sample_name(sample_num, setting_dir)

    mesh_quality = args.mesh_quality
    h5_overwrite = args.h5_overwrite
    output_folder = args.output_folder

    settings = load_settings(setting_dir, sample_name)
    data_dir = Path(settings["path"])
    mesh_settings = settings["mesh"][mesh_quality]
    # Creating outdir, a folder with the name of output_folder in the data_dir for saving the results
    outdir = arg_parser.prepare_outdir(data_dir, output_folder)
    h5_file = mesh_utils.compile_h5(
        data_dir,
        settings["scan_type"],
        overwrite=h5_overwrite,
        is_inverted=settings["is_inverted"],
    )

    if settings["scan_type"] == "TPM":
        h5_file = mesh_utils.prepare_mask(h5_file, outdir, settings)
    if settings["scan_type"] == "CINE":
        h5_file = mesh_utils.prepare_coords(h5_file, outdir, settings)

    mesh_fname = meshing.create_mesh(
        data_dir,
        settings["scan_type"],
        mesh_settings,
        h5_file,
        plot_flag=True,
        results_folder=outdir,
    )
    geometry = create_geometry.create_geometry(
        mesh_fname, fiber_angles=settings["fiber_angles"], plot_flag=True
    )

    geo_outdir = outdir / "Geometry"
    geo_fname = geo_outdir / "geometry"
    geometry.save(geo_fname.as_posix(), overwrite_file=True)


if __name__ == "__main__":
    main()
