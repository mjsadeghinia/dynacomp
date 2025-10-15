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
import logging


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

def get_num_from_id(sample_ID, setting_dir):
    sorted_files = sorted([file for file in setting_dir.iterdir() if file.is_file() and file.suffix == ".json"])
    for i, file in enumerate(sorted_files):
        with open(file, "r") as f:
            settings = json.load(f)
            if settings["id"][2:] == sample_ID:
                return i + 1
    raise ValueError(f"Sample ID {sample_ID} not found in settings directory.")


def calculate_cavity_volume_sliced(geometry):
    """
    This functions takes a subset from the endo ffun based ont he facets that are below z = 0 surface (based on MR acqusition), and cacluate the cavity volume of the enclosed surface with the plane at z=0
    """
    mesh = geometry.mesh
    ffun = geometry.ffun
    endo_marker = geometry.markers["ENDO"]
    geometry = slice_ffun(geometry)

    ds = dolfin.Measure("exterior_facet", subdomain_data=ffun, domain=mesh)(endo_marker)
    X = dolfin.SpatialCoordinate(mesh)
    N = dolfin.FacetNormal(mesh)
    vol_form = (-1.0 / 3.0) * dolfin.dot(X, N)
    return dolfin.assemble(vol_form * ds)


def slice_ffun(geometry):
    """
    This function will change the ffun value from 6 to 8 if the ffun is above the plane at z=0
    """
    mesh = geometry.mesh
    ffun = geometry.ffun

    for fc in dolfin.facets(mesh):
        if fc.exterior() and ffun[fc] == 6:
            coord = mesh.coordinates()[fc.entities(0)]
            center_coord = np.mean(coord, 0)
            if center_coord[2] > 0:
                ffun[fc] = 8

    fname = "test_ffun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(ffun)
    return geometry


def calculate_tissue_volume_sliced(geometry):
    """
    This functions calculates the tissue volumes below the plane at z=0
    """
    mesh = geometry.mesh
    geometry = slice_cfun(geometry)
    cfun = geometry.cfun
    tissue_volume = dolfin.assemble(
        dolfin.Constant(1) * dolfin.dx(domain=mesh, subdomain_data=cfun, subdomain_id=(0, 3))
    )
    return tissue_volume


def slice_cfun(geometry):
    """
    This function will slice the cfun so the cells above the z=0 plane are sliced
    """
    mesh = geometry.mesh
    cfun = geometry.cfun
    cfun.set_all(0)
    for c in dolfin.cells(mesh):
        coord = mesh.coordinates()[c.entities(0)]
        center_coord = np.mean(coord, 0)
        if center_coord[2] > 0:
            cfun[c] = 1
    fname = "test_cfun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(cfun)
    return geometry


def load_mr_cardiac_cycle_duration(h5_dir):
    # Finding the h5 file:
    h5_files = list(h5_dir.glob("*.h5"))
    if len(h5_files) > 1:
        logger.error("There are multiple h5 files!")
        return

    with h5py.File(h5_files[0], "r") as f:
        CC_duration = f.attrs["cardiac_cycle_duration"]

    return CC_duration


def load_pressure_volumes(data_dir):
    PV_data_fname = [fname for fname in data_dir.iterdir() if "PV_data" in fname.stem][0]
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


def find_best_mri_shift(mri_time, mri_volumes, pv_time, pv_volumes, N=5):
    """
    Find the best roll (shift) for mri_volumes so that its path aligns best with pv_volumes.
    Returns:
      best_shift   : int
                     The shift value (between 0 and N) that gives the highest correlation.
      best_corr    : float
                     The Pearson correlation coefficient at the best shift.
    """
    best_shift = 0
    best_corr = -np.inf
    for shift in range(N + 1):
        rolled_volumes = np.roll(mri_volumes, shift)
        # Interpolate the rolled mri_volumes onto the pv_time scale.
        aligned_volumes = np.interp(pv_time, mri_time, rolled_volumes)
        # Calculate the Pearson correlation coefficient between the aligned mri volumes and pv_volumes.
        corr = np.corrcoef(aligned_volumes, pv_volumes)[0, 1]
        # Update the best_shift if this shift gives a higher correlation.
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift, best_corr


def calibrate_pv_to_mri(mri_time, mri_volumes, pv_time, pv_volumes, weights=None):
    pv_interp = np.interp(mri_time, pv_time, pv_volumes)
    if weights is None:
        weights = np.ones_like(mri_time)
    mean_pv = np.average(pv_interp, weights=weights)
    mean_mri = np.average(mri_volumes, weights=weights)
    a = np.sum(weights * (pv_interp - mean_pv) * (mri_volumes - mean_mri)) / np.sum(
        weights * (pv_interp - mean_pv) ** 2
    )
    b = mean_mri - a * mean_pv

    calibrated_pv_volumes = a * pv_volumes + b
    return a, b, calibrated_pv_volumes


def update_settings(settings, a, b):
    """
    Update the settings dictionary with the calibration coefficients.
    """
    settings["PV"]["calibration"] = {
        "a": a,
        "b": b,
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

def append_additional_volumes(mri_time, registered_volumes, registered_pressures, pv_volumes, pv_pressures):
    stroke_volume = max(registered_volumes) - min(registered_volumes)
    if registered_volumes[0]-registered_volumes[-1] > stroke_volume * 0.05:
        logger.warning("The first and last volume are not close enough, appending data from the PV curves.")
        last_reg_vol = registered_volumes[-1]
        ind = np.where(pv_volumes < last_reg_vol)[0][-1] + 1
        pv_pres = pv_pressures[ind:]
        pv_vols = pv_volumes[ind:]
        missing_range = (registered_volumes[0]-registered_volumes[-1]) / stroke_volume
        num_additional_data = int(missing_range / 0.03)
        skip_interval = int(len(pv_vols) / num_additional_data)
        registered_pressures = np.append(registered_pressures, pv_pres[::skip_interval])
        registered_volumes = np.append(registered_volumes, pv_vols[::skip_interval])
        mri_time = np.append(mri_time, mri_time[-1] + (mri_time[-1] - mri_time[-2]) * np.arange(1, len(registered_volumes)-len(mri_time) + 1))
    return mri_time, registered_volumes, registered_pressures


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
        "-d",
        "--data_dir",
        default="/home/shared/00_data",
        type=Path,
        help="The settings directory where data files are stored.",
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

    parser.add_argument(
        "-r",
        "--results_dir",
        default="/home/shared/01_results_coarse_mesh",
        type=Path,
        help="The results folder where the processed data should be saved.",
    )
    args = parser.parse_args(args)

    sample_nums = args.number
    sample_ID = args.ID
    settings_dir = args.settings_dir
    data_dir = args.data_dir
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
        if "TPM" not in settings:
            logger.warning(f"TPM not found in settings for {sample_name}")
            continue
        pv_data_dir = results_dir / sample_name / pv_folder
        tpm_data_dir = results_dir / sample_name / "TPM"
        meshes_data_dir = tpm_data_dir / "00_Meshes"
        h5_dir = data_dir / sample_name / "TPM"
        output_dir = results_dir / sample_name / "TPM" / output_folder
        output_dir = arg_parser.prepare_outdir(output_dir)

        pv_time, pv_pressures, pv_volumes = load_pressure_volumes(pv_data_dir)

        cc_duration = load_mr_cardiac_cycle_duration(h5_dir)
        mri_time_total = np.mean(cc_duration) * 1000
        mri_time_total_std = np.std(cc_duration) * 1000
        if mri_time_total_std / mri_time_total > 0.05:
            logger.warning(
                f"The cardiac cyclee duration between stacks have a STD/AVE > 5%, Ave: {mri_time_total}ms and STD: {mri_time_total_std}ms"
            )

        mri_time_series = [file for file in meshes_data_dir.iterdir() if file.is_dir()]
        # Sorting numerically based on the number in 'time_X'
        mri_time_series = sorted(
            mri_time_series,
            key=lambda p: int(p.name.split("_")[-1]),  # Extract and convert the number
        )

        mri_volumes_original = []
        pulse_logger = logging.getLogger("pulse")
        pulse_logger.setLevel(logging.WARNING)
        for folder in mri_time_series:
            mesh_fname = folder / "geometry/Geometry.h5"
            geo = pulse.HeartGeometry.from_file(mesh_fname.as_posix())
            mri_volumes_original.append(geo.cavity_volume())
        mri_time = np.linspace(0, mri_time_total, len(mri_volumes_original))
        best_shift, _ = find_best_mri_shift(mri_time, mri_volumes_original, pv_time, pv_volumes, N=20)
        mri_volumes = mri_volumes_original.copy()
        mri_volumes = np.roll(mri_volumes, best_shift)
        if best_shift > 0:
            logger.info(f"MRI data has been shifted by {best_shift} in time")

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(mri_time, mri_volumes, color="black", linewidth=1)
        ax1.scatter(mri_time, mri_volumes, color="black", s=15, label="Shifted MRI Volumes")
        ax1.plot(mri_time, mri_volumes_original, color="gray", linewidth=1)
        ax1.scatter(mri_time, mri_volumes_original, color="gray", s=15, label="Original MRI Volumes")
        # Original PV volumes in tab:orange on the right y-axis.
        ax2 = ax1.twinx()
        ax2.scatter(pv_time, pv_volumes, s=15, label="PV Volumes", color="tab:orange")
        ax2.plot(pv_time, pv_volumes, color="tab:orange")
        ax2.set_ylabel("PV Volume [RVU]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
        plt.tight_layout()
        plt.title(f"MRI data has been shifted by {best_shift} in time")
        plt.savefig(output_dir / "shifted_mri.png", dpi=300)
        plt.close()

        regirstered_pressures = np.interp(mri_time, pv_time, pv_pressures)
        # adding a possibility for adjusting the ED pressure just, USE with caution! only for the samples where the original PV loop makes sense to adjust the ED pressure
        regirstered_pressures[0] = regirstered_pressures[0] if not "ED_pressure_offset" in settings["PV"] else regirstered_pressures[0] + settings["PV"]["ED_pressure_offset"]
        N = len(mri_time)
        weights = np.ones(len(mri_time))
        weights[: int(0.25 * N)] = 5
        weights[-int(0.25 * N) :] = 5
        a, b, calibrated_pv_volumes = calibrate_pv_to_mri(mri_time, mri_volumes, pv_time, pv_volumes, weights=weights)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        # MRI volumes in black and calibrated PV volumes in tab:blue on the left y-axis.
        ax1.scatter(mri_time, mri_volumes, s=15, label="MRI Volumes", color="black")
        ax1.plot(mri_time, mri_volumes, color="black")
        ax1.plot(pv_time, calibrated_pv_volumes, color="tab:blue", linewidth=1)
        ax1.scatter(pv_time, calibrated_pv_volumes, label="Calibrated PV Volumes", s=15, color="tab:blue")
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("MRI / Calibrated PV Volume", color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        # Original PV volumes in tab:orange on the right y-axis.
        ax2 = ax1.twinx()
        ax2.scatter(pv_time, pv_volumes, s=15, label="PV Volumes", color="tab:orange")
        ax2.plot(pv_time, pv_volumes, color="tab:orange")
        ax2.set_ylabel("PV Volume [RVU]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")
        plt.tight_layout()
        fname = output_dir / f"calibrated_volumes.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mri_volumes, regirstered_pressures, "k", linewidth=1)
        ax.scatter(mri_volumes, regirstered_pressures, s=15, c="k")
        ax.scatter(mri_volumes[0], regirstered_pressures[0], c="r", s=20)
        plt.xlabel("Volume [micro Liter]")
        plt.ylabel("LV Pressure [mmHg]")

        # Add a second y-axis for LV Pressure in kPa
        ax2 = ax.twinx()
        mmHg_to_kPa = 0.133322
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * mmHg_to_kPa, ymax * mmHg_to_kPa)
        ax2.set_ylabel("LV Pressure [kPa]")

        fname = output_dir / f"registered_pv.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        # fname = output_dir / "registered_pv_data.csv"
        # np.savetxt(fname, np.vstack((mri_time, regirstered_pressures, mri_volumes)).T, delimiter=",")

        settings = update_settings(settings, a, b)
        settings_fname = save_settings(settings, settings_dir, sample_name)
        logger.info(f"Updated settings saved to {settings_fname}")

        # Calibrate EDPVR data
        fname = pv_data_dir / f"{sample_name}_EDPVR.csv"
        np.loadtxt(fname, delimiter=",")

        # Calibrating the EDPVR data
        # Load the EDPVR data
        edpvr_pressures, edpvr_volumes = load_edpvr(pv_data_dir)
        calibrated_edpvr_volumes = a * edpvr_volumes + b
        # Calculate the x value at which y = 0 using the regression line equation (avoid division by zero)
        res = scipy.stats.linregress(calibrated_edpvr_volumes, edpvr_pressures)
        v_0 = -res.intercept / res.slope if res.slope != 0 else float('nan')
        # Calculate the standard error of the slope and intercept
        tinv = lambda p, df: abs(scipy.stats.t.ppf(p/2, df))
        ts = tinv(0.05, len(calibrated_edpvr_volumes)-2)
        # loaded the EDPVR PV data
        fname = pv_data_dir / f"{sample_name}_EDPVR_pressure_data.csv"
        with open(fname, 'r') as f:
            text = f.read()
            edpvr_pressures_all = ast.literal_eval(text)

        fname = pv_data_dir / f"{sample_name}_EDPVR_volume_data.csv"
        with open(fname, 'r') as f:
            text = f.read()
            edpvr_volumes_all = ast.literal_eval(text)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mri_volumes, regirstered_pressures, "k", linewidth=1)
        ax.scatter(mri_volumes, regirstered_pressures, s=15, c="k")
        ax.scatter(calibrated_edpvr_volumes, edpvr_pressures, s=8, c="r")
        for p,v in zip(edpvr_pressures_all, edpvr_volumes_all):
            v_calibrated = a * np.array(v) + b
            ax.plot(v_calibrated, p, c="k", linewidth=0.05)
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
        ax.plot(calibrated_edpvr_volumes, res.intercept + res.slope*calibrated_edpvr_volumes, 'b', label='EDVPR')
        ax.axhline(y=0, color='gray', linestyle='--')
        # Add a second y-axis for LV Pressure in kPa
        ax2 = ax.twinx()
        mmHg_to_kPa = 0.133322
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * mmHg_to_kPa, ymax * mmHg_to_kPa)
        ax2.set_ylabel("LV Pressure [kPa]")

        fname = output_dir / f"registered_edpvr.png"
        plt.savefig(fname, dpi=300)
        plt.close()

        pv_volumes_calibrated = a * pv_volumes + b
        regirstered_calibrated_volumes = np.interp(mri_time, pv_time, pv_volumes_calibrated)
        # mri_time, regirstered_calibrated_volumes,  regirstered_pressures= append_additional_volumes(mri_time, regirstered_calibrated_volumes, regirstered_pressures, pv_volumes_calibrated, pv_pressures)
        ED_offset_index = settings["PV"]["ED_offset_index"] if "ED_offset_index" in settings["PV"] else 0

        if not ED_offset_index==0:
            logger.info(f"ED_offset_index is set to {ED_offset_index}")
            
        regirstered_calibrated_volumes = np.roll(regirstered_calibrated_volumes, best_shift+ED_offset_index)
        regirstered_pressures = np.roll(regirstered_pressures, best_shift+ED_offset_index)

        regirstered_calibrated_volumes = np.append(regirstered_calibrated_volumes, regirstered_calibrated_volumes[0])
        regirstered_pressures = np.append(regirstered_pressures, regirstered_pressures[0])
        mri_time = np.append(mri_time,  mri_time[-1]+mri_time[-1]-mri_time[-2])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(regirstered_calibrated_volumes, regirstered_pressures, "k", linewidth=1)
        ax.scatter(regirstered_calibrated_volumes, regirstered_pressures, s=15, c="k")
        ax.scatter(regirstered_calibrated_volumes[0], regirstered_pressures[0], s=15, c="m", label="ED Point")
        ax.scatter(calibrated_edpvr_volumes, edpvr_pressures, s=8, c="r")
        for p,v in zip(edpvr_pressures_all, edpvr_volumes_all):
            v_calibrated = a * np.array(v) + b
            ax.plot(v_calibrated, p, c="k", linewidth=0.05)
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
        ax.plot(calibrated_edpvr_volumes, res.intercept + res.slope*calibrated_edpvr_volumes, 'b', label='EDVPR')
        ax.axhline(y=0, color='gray', linestyle='--')

        # Add a second y-axis for LV Pressure in kPa
        ax2 = ax.twinx()
        mmHg_to_kPa = 0.133322
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * mmHg_to_kPa, ymax * mmHg_to_kPa)
        ax2.set_ylabel("LV Pressure [kPa]")

        fname = output_dir / f"registered_edpvr_with_calibrated_cather_volume.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        # Save the calibrated EDPVR data
        fname = output_dir / "ordered_calibrated_pv_data.csv"
        np.savetxt(fname, np.vstack((mri_time, regirstered_pressures, regirstered_calibrated_volumes)).T, delimiter=",")

        # updating the geometries by adjusting based on best shift
        geo_outdir = output_dir / "Geometries"
        geo_outdir.mkdir(parents=True, exist_ok=True)
        indices = list(np.roll(np.arange(len(mri_time[:-1])), best_shift+ED_offset_index))
        for i, n in enumerate(indices):
            geo_fname = meshes_data_dir / f"time_{n}/Geometry/geometry.h5"
            geo_outname = geo_outdir / f"geometry_{i}.h5"
            if i == 0:
                logger.info(f"Time {n} is considered as ED")
                outname = geo_outdir / f"geometry_{i}_ffun.xdmf"
                geo = pulse.HeartGeometry.from_file(geo_fname.as_posix())
                with dolfin.XDMFFile(outname.as_posix()) as infile:
                    infile.write(geo.ffun)
            if geo_fname.is_file():
                shutil.copy(geo_fname, geo_outname)
            else:
                logger.warning(f"Geometry file {geo_fname} does not exist, skipping.")

if __name__ == "__main__":
    main()
