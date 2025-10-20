import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pulse
import dolfin
import h5py
import json
import shutil
import ast
import scipy.stats


import arg_parser
from structlog import get_logger

logger = get_logger()


# %%
def load_settings(settings_dir, sample_num):
    sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    settings_fname = sorted_files[sample_num - 1]
    with open(settings_fname, "r") as file:
        settings = json.load(file)
    return settings


def get_sample_name(sample_num, setting_dir):
    # Get the list of .json files in the directory and sort them by name
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    sample_name = sorted_files[sample_num - 1].with_suffix("").name
    return sample_name

def update_settings(settings, volume_shift, pressure_shift):

    settings["PV"]["EDPVR_shift"] = {
        "volume": volume_shift,
        "pressure": pressure_shift,
    }
    return settings


def save_settings(settings, settings_dir, sample_name):
    """
    Save the updated settings dictionary to a JSON file.
    """
    settings_fname = settings_dir / f"{sample_name[2:]}.json"
    with open(settings_fname, "w") as file:
        json.dump(settings, file, indent=4)
    return settings_fname


def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")

def load_calibrated_pressure_volumes(data_dir):
    PV_data_fname = [fname for fname in data_dir.iterdir() if "ordered_calibrated_pv_data" in fname.stem][0]
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    time = PV_data[:, 0] * 1000
    pressures = PV_data[:, 1]
    volumes = PV_data[:, 2]
    return time, pressures, volumes

def load_edpvr(data_dir):
    PV_data_fname = [fname for fname in data_dir.iterdir() if "EDPVR.csv" in fname.as_posix()][0]
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    pressures = PV_data[:, 0]
    volumes = PV_data[:, 1]
    return pressures, volumes

def get_sv_and_sp(registered_volumes, registered_pressures):
    """
    Calculate the stroke volume and stroke pressure from the registered volumes and pressures.
    """
    stroke_volume = []
    stroke_pressure = []
    for cycle_pres, cycle_vols in zip(registered_pressures, registered_volumes):
        # Calculate stroke volume and stroke pressure
        stroke_vols = np.max(cycle_vols) - np.min(cycle_vols)
        stroke_pres = np.max(cycle_pres) - np.min(cycle_pres)
        stroke_volume.append(stroke_vols)
        stroke_pressure.append(stroke_pres)

    return np.array(stroke_volume), np.array(stroke_pressure)

def correlate_pv_to_edpvr(registered_calibrated_volumes, registered_pressures, calibrated_edpvr_volumes_all, edpvr_pressures_all):
    edpvr_stroke_volume, edpvr_stroke_pressure = get_sv_and_sp(
            calibrated_edpvr_volumes_all, edpvr_pressures_all
        )

    pv_stroke_volume, pv_stroke_pressure = get_sv_and_sp(
        [registered_calibrated_volumes], [registered_pressures]
    )

    corr_sv = ((edpvr_stroke_volume - pv_stroke_volume)/edpvr_stroke_volume)**2
    corr_sp = ((edpvr_stroke_pressure - pv_stroke_pressure)/edpvr_stroke_pressure)**2
    corr = corr_sv + corr_sp
    cycle_num = np.argmin(corr)
    return cycle_num

def get_end_diastole_ind(
    pressures, volumes, pressure_threshold_percent=0.1, volume_threshold_percent=0.05
):
    volumes = np.asarray(volumes)
    pressures = np.asarray(pressures)
    # Calculate the thresholds for pressure and volume
    pressure_min = np.min(pressures)
    # Define the range for end-diastole
    pressure_threshold = pressure_min + pressure_threshold_percent * (
        np.max(pressures) - pressure_min
    )
    # Find indices where pressure is below the threshol
    valid_pressure_indices = np.where((pressures <= pressure_threshold))[0]
    new_volumes = volumes[valid_pressure_indices]
    new_volume_max = np.max(new_volumes)
    new_volume_min = np.min(new_volumes)
    volume_threshold = new_volume_max - volume_threshold_percent * (
        new_volume_max - new_volume_min
    )

    # Find indices where conditions are met
    valid_indices = np.where(
        (pressures <= pressure_threshold) & (volumes >= volume_threshold)
    )[0]
    # Find the index of the maximum volume in the valid region
    index = valid_indices[np.argmax(volumes[valid_indices])]

    return index



# %%
def main(args=None) -> int:
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
        "-pv",
        "--pv_folder",
        default="PV Data",
        type=Path,
        help="The folder where PV data is stored.",
    )

    parser.add_argument(
        '-o',
        "--output_folder",
        default="01_PVCalibration",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )


    args = parser.parse_args(args)

    sample_nums = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    results_dir = args.results_dir
    pv_folder = args.pv_folder
    output_folder = args.output_folder

    if sample_ID is not None:
        sample_nums = [get_num_from_id(sample_ID, settings_dir)]

    # Get the list of .json files in the directory and sort them by name
    if sample_nums is None:
        sorted_files = sorted([file for file in settings_dir.iterdir() if file.is_file() and file.suffix == ".json"])
        sample_nums = range(1, len(sorted_files) + 1)

    for sample_num in sample_nums:
        settings = load_settings(settings_dir, sample_num)
        sample_name = settings["id"]

        pv_data_dir = results_dir / sample_name / pv_folder
        tpm_data_dir = results_dir / sample_name / "TPM"
        pv_calibrated_data_dir = tpm_data_dir / output_folder

        if not pv_calibrated_data_dir.exists():
            continue

        a = settings["PV"]["calibration"]["a"]
        b = settings["PV"]["calibration"]["b"]

        if not settings["PV"]["EDPVR_shift_flags"]["pressure"] and not settings["PV"]["EDPVR_shift_flags"]["volume"]:
            logger.info(f"Sample {settings['id']} needs no shift, according to settings.")
            # Load the EDPVR data
            edpvr_pressures, edpvr_volumes = load_edpvr(pv_data_dir)
            calibrated_edpvr_volumes = a * edpvr_volumes + b
            fname = pv_calibrated_data_dir / f"{sample_name}_EDPVR_calibrated_shifted.csv"
            np.savetxt(fname, np.vstack((edpvr_pressures, calibrated_edpvr_volumes)).T, delimiter=",")
            src = pv_calibrated_data_dir / "registered_edpvr_with_calibrated_cather_volume.png"
            fname = pv_calibrated_data_dir / "shifted_registered_edpvr_with_calibrated_cather_volume.png"
            shutil.copy(src, fname)
            logger.info(f"--------------------------------")
            continue

        # Load PV calibration data
        registered_time, registered_pressures, registered_calibrated_volumes = load_calibrated_pressure_volumes(pv_calibrated_data_dir)
        # loaded the EDPVR PV data
        fname = pv_data_dir / f"{sample_name}_EDPVR_pressure_data.csv"
        with open(fname, 'r') as f:
            text = f.read()
            edpvr_pressures_all = ast.literal_eval(text)

        fname = pv_data_dir / f"{sample_name}_EDPVR_volume_data.csv"
        with open(fname, 'r') as f:
            text = f.read()
            edpvr_volumes_all = ast.literal_eval(text)        

        calibrated_edpvr_volumes_all = [
            [a * v + b for v in volume_cycle]
            for volume_cycle in edpvr_volumes_all
        ]

        if "manual_pv_to_edpvr" in settings["PV"]:
            cycle_num = settings["PV"]["manual_pv_to_edpvr"]
            logger.warning(f"Using manual PV to EDPVR correlation, {cycle_num} is selected for EDPVR.")
        else:
            cycle_num = correlate_pv_to_edpvr(
                registered_calibrated_volumes, registered_pressures,
                calibrated_edpvr_volumes_all, edpvr_pressures_all
            )
            logger.info(f"Cycle number {cycle_num} selected based on correlation.")

        # Get the end-diastole index
        ED_index = 0
        EDP, EDV = registered_pressures[ED_index], registered_calibrated_volumes[ED_index]

        # ED_index_edpvr = get_end_diastole_ind(
        #     edpvr_pressures_all[cycle_num],
        #     calibrated_edpvr_volumes_all[cycle_num]
        # )
        # EDP_edpvr, EDV_edpvr = edpvr_pressures_all[cycle_num][ED_index_edpvr], calibrated_edpvr_volumes_all[cycle_num][ED_index_edpvr]

        # Load the EDPVR data
        edpvr_pressures, edpvr_volumes = load_edpvr(pv_data_dir)
        calibrated_edpvr_volumes = a * edpvr_volumes + b

        EDP_edpvr, EDV_edpvr = edpvr_pressures[cycle_num], calibrated_edpvr_volumes[cycle_num]
        
        if settings["PV"]["EDPVR_shift_flags"]["volume"]:
            volume_diff = EDV - EDV_edpvr
        else:
            volume_diff = 0 
            logger.info(f"Sample {settings['id']} needs no volume shift, according to settings.")

        pressure_diff = EDP - EDP_edpvr
        logger.info(f"Sample {sample_name} Pressure Difference: {pressure_diff:.2f}, Volume Difference: {volume_diff:.2f}({EDV-EDV_edpvr:.2f})")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(registered_calibrated_volumes, registered_pressures, "k", linewidth=1)
        ax.scatter(registered_calibrated_volumes, registered_pressures, s=15, c="k")
        ax.scatter(registered_calibrated_volumes[ED_index], registered_pressures[ED_index], s=15, c="r", label="ED Point")
        for n, (p,v) in enumerate(zip(edpvr_pressures_all, calibrated_edpvr_volumes_all)):
            color = "r" if n == cycle_num else "k"
            linewidth = 0.5 if n == cycle_num else 0.05
            ax.plot(v, p, c=color, linewidth=linewidth)
        ax.scatter(calibrated_edpvr_volumes[cycle_num], edpvr_pressures[cycle_num], s=5, c="r")
        plt.xlabel("Volume [micro Liter]")
        plt.ylabel("LV Pressure [mmHg]")
        fname = pv_calibrated_data_dir / f"EDPVR_shift.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        settings = update_settings(settings, volume_diff, pressure_diff)
        settings_fname = save_settings(settings, settings_dir, sample_name)

        shifted_calibrated_edpvr_volumes_all = [
            [v + volume_diff for v in volume_cycle]
            for volume_cycle in calibrated_edpvr_volumes_all
        ] if settings["PV"]["EDPVR_shift_flags"]["volume"] else calibrated_edpvr_volumes_all
        shifted_edpvr_pressures_all = [
            [p + pressure_diff for p in pressure_cycle]
            for pressure_cycle in edpvr_pressures_all
        ] if settings["PV"]["EDPVR_shift_flags"]["pressure"] else edpvr_pressures_all   

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(registered_calibrated_volumes, registered_pressures, "k", linewidth=1)
        ax.scatter(registered_calibrated_volumes, registered_pressures, s=15, c="k")
        ax.scatter(registered_calibrated_volumes[ED_index], registered_pressures[ED_index], s=15, c="r", label="ED Point")
        for n, (p,v) in enumerate(zip(shifted_edpvr_pressures_all, shifted_calibrated_edpvr_volumes_all)):
            color = "r" if n == cycle_num else "k"
            linewidth = 0.5 if n == cycle_num else 0.05
            ax.plot(v, p, c=color, linewidth=linewidth)
        ax.scatter(calibrated_edpvr_volumes[cycle_num], edpvr_pressures[cycle_num], s=5, c="r")
        plt.xlabel("Volume [micro Liter]")
        plt.ylabel("LV Pressure [mmHg]")
        fname = pv_calibrated_data_dir / f"Shifted_EDPVR.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        # write the EDPVR_pv_data data to a file
        fname = pv_calibrated_data_dir / f"{sample_name}_shifted_EDPVR_pressure_data.csv"
        with open(fname, 'w') as f:
            json.dump(shifted_edpvr_pressures_all, f)

        fname = pv_calibrated_data_dir / f"{sample_name}_shifted_EDPVR_volume_data.csv"
        with open(fname, 'w') as f:
            json.dump(shifted_calibrated_edpvr_volumes_all, f)


        
        # Shift the EDPVR volumes and pressures
        shifted_calibrated_edpvr_volumes = [v + volume_diff for v in
            calibrated_edpvr_volumes
        ] if settings["PV"]["EDPVR_shift_flags"]["volume"] else calibrated_edpvr_volumes
        shifted_edpvr_pressures = [
            p + pressure_diff for p in edpvr_pressures
        ] if settings["PV"]["EDPVR_shift_flags"]["pressure"] else edpvr_pressures
        # Calculate the x value at which y = 0 using the regression line equation (avoid division by zero)
        res = scipy.stats.linregress(shifted_calibrated_edpvr_volumes, shifted_edpvr_pressures)
        v_0 = -res.intercept / res.slope if res.slope != 0 else float('nan')
        # Calculate the standard error of the slope and intercept
        tinv = lambda p, df: abs(scipy.stats.t.ppf(p/2, df))
        ts = tinv(0.05, len(shifted_calibrated_edpvr_volumes)-2)

        registered_calibrated_volumes_cycle = np.append(registered_calibrated_volumes, registered_calibrated_volumes[0])
        registered_pressures_cycle = np.append(registered_pressures, registered_pressures[0])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(registered_calibrated_volumes_cycle, registered_pressures_cycle, "k", linewidth=1)
        ax.scatter(registered_calibrated_volumes_cycle, registered_pressures_cycle, s=15, c="k")
        ax.scatter(registered_calibrated_volumes_cycle[ED_index], registered_pressures_cycle[ED_index], s=15, c="m", label="ED Point")
        for n, (p,v) in enumerate(zip(shifted_edpvr_pressures_all, shifted_calibrated_edpvr_volumes_all)):
            color = "r" if n == cycle_num else "k"
            linewidth = 0.5 if n == cycle_num else 0.05
            ax.plot(v, p, c=color, linewidth=linewidth)

        ax.scatter(shifted_calibrated_edpvr_volumes, shifted_edpvr_pressures, s=8, c="r")
        plt.xlabel("Volume [micro Liter]")
        plt.ylabel("LV Pressure [mmHg]")

        # Add a title with the slope and intercept
        textstr = (
                f"slope (95%): {res.slope:.3f} $\pm$ {ts*res.stderr:.3f}\n"
                f"$v_0$ (P=0): {v_0:.2f}"
            )
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            # bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax.plot(shifted_calibrated_edpvr_volumes, res.intercept + res.slope*np.array(shifted_calibrated_edpvr_volumes), 'b', label='EDVPR')
        ax.scatter(shifted_calibrated_edpvr_volumes[cycle_num], shifted_edpvr_pressures[cycle_num], s=5, c="r")
        ax.axhline(y=0, color='gray', linestyle='--')

        # Add a second y-axis for LV Pressure in kPa
        ax2 = ax.twinx()
        mmHg_to_kPa = 0.133322
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * mmHg_to_kPa, ymax * mmHg_to_kPa)
        ax2.set_ylabel("LV Pressure [kPa]")

        fname =  pv_calibrated_data_dir / f"shifted_registered_edpvr_with_calibrated_cather_volume.png"
        plt.savefig(fname, dpi=300)
        plt.close()


        fname = pv_calibrated_data_dir / f"{sample_name}_EDPVR_calibrated_shifted.csv"
        np.savetxt(fname, np.vstack((shifted_edpvr_pressures, shifted_calibrated_edpvr_volumes)).T, delimiter=",")
        logger.info(f"--------------------------------")

if __name__ == "__main__":
    main()
