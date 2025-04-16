import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pulse
import dolfin
import h5py
import json


from structlog import get_logger

logger = get_logger()

#%%
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
    h5_files = list(h5_dir.glob('*.h5'))
    if len(h5_files) > 1:
        logger.error("There are multiple h5 files!")
        return
    
    with h5py.File(h5_files[0], "r") as f:
        CC_duration = f.attrs["cardiac_cycle_duration"]
        
    return CC_duration

def load_pressure_volumes(data_dir, sample_name):
    PV_data_fname = [fname for fname in data_dir.iterdir() if "PV_data" in fname.stem][0]
    PV_data = np.loadtxt(PV_data_fname.as_posix(), delimiter=",")
    time = PV_data[:, 0] * 1000
    pressures = PV_data[:, 1]
    volumes = PV_data[:, 2]
    return time, pressures, volumes

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
#%%
def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--number",
        type=int,
        help="The sample number(s), will process all the sample if not indicated",
    )
    parser.add_argument(
        "-f",
        "--data_folder",
        default='coarse_mesh',
        type=str,
        help="The data folder where the time series mesh are stored",
    )
    parser.add_argument(
        "--settings_dir",
        default="/home/shared/dynacomp/settings",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    parser.add_argument(
        "--settings_tpm_dir",
        default="/home/shared/dynacomp/settings_tpm",
        type=Path,
        help="The settings directory where json files are stored.",
    )
    args = parser.parse_args()

    sample_num = args.number
    folder = args.data_folder
    setting_dir = args.settings_dir
    setting_tpm_dir = args.settings_tpm_dir
    sample_name = get_sample_name(sample_num, setting_tpm_dir)
    settings = load_settings(setting_dir, sample_name)
    settings_tpm = load_settings(setting_tpm_dir, sample_name)
    
    mri_folder = Path(settings_tpm["path"]) / folder
    pv_folder = Path(settings["path"]) / "PV Data" / "PV Data"
    pv_time, pv_pressures, pv_volumes = load_pressure_volumes(pv_folder, sample_name)

    h5_dir = mri_folder.parent
    cc_duration = load_mr_cardiac_cycle_duration(h5_dir)
    mri_time_total = np.mean(cc_duration)*1000
    mri_time_total_std = np.std(cc_duration)*1000
    if mri_time_total_std/mri_time_total>0.05:
        logger.warning(f"The cardiac cyclee duration between stacks have a STD/AVE > 5%, Ave: {mri_time_total}ms and STD: {mri_time_total_std}ms")

    mri_time_series = [file for file in mri_folder.iterdir() if file.is_dir()]
    # Sorting numerically based on the number in 'time_X'
    mri_time_series = sorted(
        mri_time_series,
        key=lambda p: int(p.name.split("_")[-1]),  # Extract and convert the number
    )
    
    mri_volumes = []

    for folder in mri_time_series:
        mesh_fname = folder / "geometry/Geometry.h5"
        geo = pulse.HeartGeometry.from_file(mesh_fname.as_posix())
        mri_volumes.append(geo.cavity_volume())
    mri_time = np.linspace(0, mri_time_total, len(mri_volumes))
    best_shift, _ = find_best_mri_shift(mri_time, mri_volumes, pv_time, pv_volumes, N=5)
    mri_volumes = np.roll(mri_volumes, best_shift)
    if best_shift>0:
        logger.warning(f"MRI data has been shifted by {best_shift} in time")
    
    
    regirstered_pressures = np.interp(mri_time, pv_time, pv_pressures)    
    #Triming the mri_volumes based on EDV
    ind = np.where(mri_volumes[-10:]>mri_volumes[0])[0]
    if ind.shape[0]>0:
        print(ind)
        ind = ind[-1]
        mri_time = mri_time[:-ind]
        mri_volumes = mri_volumes[:-ind]
        regirstered_pressures = regirstered_pressures[:-ind]
        
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.scatter(mri_time, mri_volumes, s=20, label="MRI Volumes", color="b")
    ax1.plot(mri_time, mri_volumes, color="b")
    ax1.set_xlabel("Time [ms]")
    ax1.set_ylabel("MRI Volume [micro Liter]", color="b")
    ax1.tick_params(axis='y', labelcolor="b")
    # Create a second y-axis sharing the same x-axis for PV data
    ax2 = ax1.twinx()
    ax2.scatter(pv_time, pv_volumes, s=20, label="PV Volumes", color="r")
    ax2.plot(pv_time, pv_volumes, color="r")
    ax2.set_ylabel("PV Volume [RVU]", color="r")  
    ax2.tick_params(axis='y', labelcolor="r")
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right')
    plt.tight_layout()
    fname = mri_folder.parent / f"Volumes.png"
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

    fname = mri_folder.parent / f"Registered_PV.png"
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
