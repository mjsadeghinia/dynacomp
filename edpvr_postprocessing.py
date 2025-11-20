import argparse
import json
from pathlib import Path
import utils_post
import numpy as np
import csv
import scipy.stats
import pulse

import utils

from structlog import get_logger

logger = get_logger()


# %%
def parse_arguments(args=None):
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        nargs="+",
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
        "-x",
        "--exclusion",
        action="store_true",
        help="The flag for excluding samples with high error",
    )

    parser.add_argument(
        "--error_threshold",
        default=0.5,
        type=float,
        help="The value used for excluding samples with high error",
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

    parser.add_argument(
        "-o",
        "--output_dir",
        default="/home/shared/02_post_processing/02_EDPVR_Results",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )

    parser.add_argument(
        "-f",
        "--folder",
        default="02_EDPVR_Modeling",
        type=str,
        help="The folder containing the EDPVR results."
    )

    parser.add_argument(
        "--fibrosis_path",
        type=str,
        default="/home/shared/00_data/fibrosis_data.csv",
        help="The path to the fibrosis data file (csv)."
    )

    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Version number (e.g., 3)",
    )

    parser.add_argument(
        '--exclude',
        nargs='*',
        type=str,
        default=None,
        help='Sample ID(s) to exclude from processing.'
    )

    return parser.parse_args(args)

def load_edpvr_calibrated_shifted(pv_directory: Path):
    path = next(f for f in pv_directory.iterdir() if "EDPVR_calibrated_shifted.csv" in f.name)
    data = np.loadtxt(path, delimiter=',')
    pres = data[:, 0] * 0.133322
    vols = data[:, 1]
    idx_v = np.argsort(vols)
    vols = vols[idx_v]
    pres = pres[idx_v]
    return pres, vols

def get_v0_edpvr(pv_directory: Path):
    pres, vols = load_edpvr_calibrated_shifted(pv_directory)
    res = scipy.stats.linregress(vols, pres)
    v_0 = -res.intercept / res.slope if res.slope != 0 else float('nan')
    return v_0

def get_v0_sim(sim_directory: Path):
    unloaded_geometry_fname = sim_directory / "unloaded_geometry_0_with_fibers.h5"
    unloaded_geometry = pulse.HeartGeometry.from_file(
        unloaded_geometry_fname.as_posix()
    )
    v_0 = unloaded_geometry.cavity_volume()
    return v_0

def prepare_results_dict(data_dict, ordered_keys=None, round_flag=True):
    data_dict = utils_post.flatten_data_dict(data_dict)
    if round_flag:
        data_dict = {k: np.round(v,3) for k, v in data_dict.items() if v}
    # Determine ordering of keys
    if ordered_keys is not None:
        ordered = [k for k in ordered_keys if k in data_dict]
    else:
        ordered = list(data_dict.keys())
    # Reorder data_dict according to ordered list
    data_dict = {k: data_dict[k] for k in ordered}
    return data_dict

def get_fibrosis_data(ids, fibrosis_path):
    fibrosis = dict()
    fibrosis_slice1 = dict()
    fibrosis_slice2 = dict()
    data = np.loadtxt(fibrosis_path, delimiter=',', skiprows=1, dtype=str)
    for key in ids.keys():
        if key not in fibrosis:
            fibrosis[key] = []
            fibrosis_slice1[key] = []
            fibrosis_slice2[key] = []
        if ids[key]:
            for id in ids[key]:
                try:
                    ind = np.where(data[:,0]==id[2:])[0][0]
                    fibrosis[key].append(float(data[ind][-1]))
                    fibrosis_slice1[key].append(float(data[ind][-3]))
                    fibrosis_slice2[key].append(float(data[ind][-2]))
                except IndexError:
                    logger.error(f"Sample {id} not found in fibrosis data.")
                    fibrosis[key].append(np.nan)
                    fibrosis_slice1[key].append(np.nan)
                    fibrosis_slice2[key].append(np.nan)
    return fibrosis, fibrosis_slice1, fibrosis_slice2

# %%
def main(args=None) -> int:
    if args is None:
        args = parse_arguments()
    else:
        # updating arguments if called by function
        default_args = parse_arguments()
        default_args = vars(default_args)
        for key, value in vars(args).items():
            if value is not None:
                default_args[key] = value
        args = argparse.Namespace(**default_args)
    # Getting the arguments
    sample_num = args.number
    sample_ID = args.ID
    exclusion_flag = args.exclusion
    error_threshold = args.error_threshold
    settings_dir = args.settings_dir
    v = f"_{args.version}" if args.version else ""
    results_dir = f"{args.results_dir}{v}"
    edpvr_folder = f"{args.folder}{v}"
    fibrosis_path = args.fibrosis_path
    output_dir = args.output_dir
    output_dir = output_dir.parent / (output_dir.name + v)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the results dicts
    group_list = ["SHAM", "AS"]
    time_list = [6, 12]
    diameter_list = [107, 130, 150]

    ids = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    a_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    af_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    a_af_matparam = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    err = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    v0_edpvr = utils_post.initialize_results_dict(group_list, time_list, diameter_list)
    v0_sim = utils_post.initialize_results_dict(group_list, time_list, diameter_list)


    # Load settings
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])

    if sample_ID is not None:
        sample_num = utils.get_num_from_id(sample_ID, settings_dir)
        sample_nums = [sample_num]
    elif sample_num is None:
        sample_nums = range(1, len(sorted_files) + 1)

    if args.exclude:
        exclude_nums = []
        for ex in args.exclude:
            ex_num = utils.get_num_from_id(ex, args.settings_dir)
            exclude_nums.append(ex_num)
        sample_nums = [s for s in sample_nums if s not in exclude_nums]
        print("--------------------------")
        logger.warning(f"Excluding samples: {args.exclude}")
        print("--------------------------")

    for n in sample_nums:
        settings = utils.load_settings(settings_dir, n)
        sample_name = settings["id"]
        edpvr_dir = Path(results_dir) / sample_name / "TPM" / edpvr_folder
        pv_dir = Path(results_dir) / sample_name / "TPM" / "01_PVCalibration"
        if not edpvr_dir.exists():
            continue
        group = settings["group"]
        time = settings["time"]
        diameter = settings.get("ring_diameter", None)
        edpvr_data = utils.read_edpvr_data(edpvr_dir)
        edpvr_data_sorted = edpvr_data[edpvr_data[:, -1].argsort()]
        a = edpvr_data_sorted[0][0]
        af = edpvr_data_sorted[0][1]
        bf = edpvr_data_sorted[0][-2]
        error = edpvr_data_sorted[0][-1]
        sim_dir = Path(f"{edpvr_dir}/a_{a}_af_{af}_bf_{bf}")

        if exclusion_flag and error>error_threshold :
            logger.warning(f"Sample {sample_name} with the error of {error} is ignored")
            continue
        if diameter is None:
            ids[group][time].append(sample_name)
            a_matparam[group][time].append(a)
            af_matparam[group][time].append(af)
            a_af_matparam[group][time].append((np.round(a/af, 3)))
            v0_edpvr[group][time].append(get_v0_edpvr(pv_dir))
            v0_sim[group][time].append(get_v0_sim(sim_dir))
            err[group][time].append(error)
        else:
            ids[group][time][diameter].append(sample_name)
            a_matparam[group][time][diameter].append(a)
            af_matparam[group][time][diameter].append(af)
            a_af_matparam[group][time][diameter].append((np.round(a/af, 3)))
            v0_edpvr[group][time][diameter].append(get_v0_edpvr(pv_dir))
            v0_sim[group][time][diameter].append(get_v0_sim(sim_dir))
            err[group][time][diameter].append(error)

    # Save the results
    ordered_keys = ["SHAM_6", "SHAM_12", "SHAM_20", "AS_6_150", "AS_12_150", "AS_6_130", "AS_12_130", "AS_12_107"]
    fname = output_dir / "ECM_Stiffness.png"
    utils_post.plot_bar_with_data(a_matparam, fname, ylabel="ECM Stiffness [kPa]", ordered_keys=ordered_keys)
    fname = output_dir / "Myocyte_Stiffness.png"
    utils_post.plot_bar_with_data(af_matparam, fname, ylabel="Myocyte Stiffness [kPa]", ordered_keys=ordered_keys)
    fname = output_dir / "ECM_Myocyte_Stiffness_Ratio.png"
    utils_post.plot_bar_with_data(a_af_matparam, fname, ylabel="ECM/Myocyte Stiffness Ratio", ordered_keys=ordered_keys)
    fname = output_dir / "EDPVR_Error.png"
    utils_post.plot_bar_with_data(err, fname, ylabel="EDPVR Error", ordered_keys=ordered_keys, ylim=(0, 1))

    ids = prepare_results_dict(ids, ordered_keys=ordered_keys, round_flag=False)
    a_matparam = prepare_results_dict(a_matparam, ordered_keys=ordered_keys)
    af_matparam = prepare_results_dict(af_matparam, ordered_keys=ordered_keys)
    a_af_matparam = prepare_results_dict(a_af_matparam, ordered_keys=ordered_keys)
    v0_edpvr = prepare_results_dict(v0_edpvr, ordered_keys=ordered_keys)
    v0_sim = prepare_results_dict(v0_sim, ordered_keys=ordered_keys)
    err = prepare_results_dict(err, ordered_keys=ordered_keys)

    fname = output_dir / "V0_Comparison.png"
    slope, intercept, r_squared, p_value, std_err = utils_post.plot_maximums_with_regression(fname.as_posix(), v0_edpvr, v0_sim, case='v0')

    fibrosis, fibrosis_slice1, fibrosis_slice2 = get_fibrosis_data(ids, fibrosis_path)
    fname = output_dir / "Fibrosis_Comparison.png"
    slope_fibrosis, intercept_fibrosis, r_squared_fibrosis, p_value_fibrosis, std_err_fibrosis = utils_post.plot_maximums_with_regression(fname.as_posix(), fibrosis, a_matparam, case='fibrosis', x1=fibrosis_slice1, x2=fibrosis_slice2)
    # Save the results to a csv file

    fname = output_dir / "EDPVR_Results.csv"
    with open(fname, 'w', newline='') as csvfile:
        # Define header
        header = ["Group", "ID", "a_matparam [kPa]", "af_matparam [kPa]", "a_af_matparam", "v0_sim [microL]" , "v0_edpvr [microL]", "Fibrosis Slice I [%]", "Fibrosis Slice II[%]", "Fibrosis [%]","err [kPa]"]
        writer = csv.writer(csvfile)
        writer.writerow(header)

        # Loop over groups
        for group, id_list in ids.items():
            for i, sample_id in enumerate(id_list):
                row = [
                    group,
                    sample_id,
                    a_matparam.get(group, [None])[i] if group in a_matparam and len(a_matparam[group]) > i else None,
                    af_matparam.get(group, [None])[i] if group in af_matparam and len(af_matparam[group]) > i else None,
                    a_af_matparam.get(group, [None])[i] if group in a_af_matparam and len(a_af_matparam[group]) > i else None,
                    v0_sim.get(group, [None])[i] if group in v0_sim and len(v0_sim[group]) > i else None,
                    v0_edpvr.get(group, [None])[i] if group in v0_edpvr and len(v0_edpvr[group]) > i else None,
                    fibrosis_slice1.get(group, [None])[i] if group in fibrosis_slice1 and len(fibrosis_slice1[group]) > i else None,
                    fibrosis_slice2.get(group, [None])[i] if group in fibrosis_slice2 and len(fibrosis_slice2[group]) > i else None,
                    fibrosis.get(group, [None])[i] if group in fibrosis and len(fibrosis[group]) > i else None,
                    err.get(group, [None])[i] if group in err and len(err[group]) > i else None,
                ]
                writer.writerow(row)
             
# %%
if __name__ == "__main__":
    main()