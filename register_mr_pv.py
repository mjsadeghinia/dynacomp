import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pulse
import dolfin


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
            center_coord = np.mean(coord,0)
            if center_coord[2]>0:
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
    tissue_volume = dolfin.assemble(dolfin.Constant(1)*dolfin.dx(domain=mesh, subdomain_data=cfun, subdomain_id = (0, 3)))
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
        center_coord = np.mean(coord,0)
        if center_coord[2]>0:
            cfun[c] = 1
    fname = "test_cfun.xdmf"
    with dolfin.XDMFFile(fname) as infile:
        infile.write(cfun)
    return geometry     
        
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
        default='dynacomp/00_data/CineData/AS/6weeks/130/OP129_1/PV Data/PV Data',
        type=str,
        help="The directory to pv results",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        default='/home/shared/dynacomp/00_data/TPMData/AS/6weeks/130/OP129_1/coarse_mesh',
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )
    args = parser.parse_args()

    mri = Path(args.mri)
    pv = Path(args.pv)
    output_folder = Path(args.output_folder)
    
    mri_time_series = [file for file in mri.iterdir() if file.is_dir()]
    # Sorting numerically based on the number in 'time_X'
    mri_time_series = sorted(
        mri_time_series, 
        key=lambda p: int(p.name.split('_')[-1])  # Extract and convert the number
    )
    
    volumes = []
    tissues = []
    for folder in mri_time_series:
        mesh_fname = folder / 'geometry/Geometry.h5'
        geo = pulse.HeartGeometry.from_file(
            mesh_fname.as_posix()
        )
        volume_t = calculate_cavity_volume_sliced(geo)
        volumes.append(volume_t)
        tissue_t = calculate_tissue_volume_sliced(geo)
        tissues.append(tissue_t)
        
        
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.linspace(0,1,len(volumes)), volumes, s=20, label="Data Points")
    ax.plot(np.linspace(0,1,len(volumes)), volumes, color="b")    
    
    plt.xlabel("Normalized time from ES to ED[-]]")
    plt.ylabel("Volume [micro Liter]")
    plt.legend()
    fname = output_folder / f"MRI_Volumes.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.linspace(0,1,len(tissues)), tissues, s=20, label="Data Points")
    ax.plot(np.linspace(0,1,len(tissues)), tissues, color="b")    
    
    plt.xlabel("Normalized time from ES to ED[-]]")
    plt.ylabel("Tissue Volume [micro Liter]")
    plt.legend()
    fname = output_folder / f"MRI_Tissue_Volumes.png"
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
