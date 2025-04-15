import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pulse
import dolfin
import h5py


from structlog import get_logger

logger = get_logger()


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
    mmHg_to_kPa = 0.133322
    time = PV_data[:, 0] * 1000
    pressures = PV_data[:, 1] * mmHg_to_kPa
    volumes = PV_data[:, 2]
    return time, pressures, volumes


#%%
def main(args=None) -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mri",
        default='/home/shared/dynacomp/00_data/TPMData/AS/6weeks/130/OP129_1/coarse_mesh',
        type=str,
        help="The directory to mri results, where the meshes are generated and stored in folders",
    )

    parser.add_argument(
        "--pv",
        default="/home/shared/dynacomp/00_data/CineData/AS/6weeks/130/OP129_1/PV Data/PV Data",
        type=str,
        help="The directory to pv results",
    )
    
    args = parser.parse_args()

    mri_folder = Path(args.mri)
    sample_name = mri_folder.parent.stem
    pv_folder = Path(args.pv)
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
    mri_shift = 2
    mri_volumes = np.roll(mri_volumes, mri_shift)
    
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

if __name__ == "__main__":
    main()
