import subprocess
import numpy as np
from structlog import get_logger
from pathlib import Path
import matplotlib.pyplot as plt

import ldrb
import dolfin
import pulse
import logging


logger = get_logger()
#%%
def grid_triangle(N=30, amin=0.05, amax=5, afmin=0.05, afmax=5):
    """
    Generate approximately N equally distributed points inside a triangle
    with vertices (amin, afmin), (amax, afmin) and (amin, afmax).
    Returns arrays of a, af.
    """
    # Find the smallest n such that (n+1)*(n+2)//2 >= N
    n = 1
    while (n+1)*(n+2)//2 < N:
        n += 1

    V1 = np.array([amin, afmin])
    V2 = np.array([amax, afmin])
    V3 = np.array([amin, afmax])

    a_af_list = []
    # af_list = []

    for i in range(n+1):
        for j in range(n+1 - i):
            u = i / n
            v = j / n
            w = 1 - u - v
            pt = w * V1 + u * V2 + v * V3
            a_af = [round(pt[0],3) , round(pt[1],3)]
            a_af_list.append(a_af)
            # af_list.apped(pt[1])

    return a_af_list

def biased_linspace(start, stop, N, bias_power=2):
    """Return N points from start to stop, biased toward start."""
    t = np.linspace(0, 1, N)
    t_biased = t**bias_power
    return start + (stop - start) * t_biased

def grid_triangle_biased(N, amin=0.05, amax=5, afmin=0.05, afmax=5, bias_power=1.3):
    # Vertical and horizontal edges
    a_edge = biased_linspace(amin, amax, N, bias_power)
    af_edge = np.full(N, afmin)
    af_edge_h = biased_linspace(afmin, afmax, N, bias_power)
    a_edge_h = np.full(N, amin)

    a_af_list = []
    for i in range(N):
        n_div = i + 2  # Number of points along this line
        for j in range(n_div):
            t = j / (n_div - 1) if n_div > 1 else 0
            a_val = a_edge_h[i] + t * (a_edge[i] - a_edge_h[i])
            af_val = af_edge_h[i] + t * (af_edge[i] - af_edge_h[i])
            a_af = [round(a_val,3) , round(af_val,3)]
            a_af_list.append(a_af)

    return a_af_list

def plot_triangle(a_af_lists, colors=None, labels=None):
    """
    Plot the triangle defined by the vertices (amin, afmin), (amax, afmin) and (amin, afmax)
    and the points in a_af_lists (can be a single list or a list of lists).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    a_af_lists = [a_af_lists]
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange']
    if labels is None:
        labels = [f'Grid {i+1}' for i in range(len(a_af_lists))]

    for i, a_af in enumerate(a_af_lists):
        a_af = np.array(a_af)
        ax.scatter(a_af[:, 0], a_af[:, 1], color=colors[i % len(colors)], s=15, label=labels[i])

    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('a')
    plt.ylabel('af')
    plt.title('Triangle Grid Points')
    plt.legend()
    plt.grid()
    return fig, ax

def load_fiber_modeling(sample_ID):
    sample_dir = Path(f"01_results_coarse_mesh/OP{sample_ID}/TPM")
    fiber_modeling_fname = sample_dir / "03_Fiber_Modeling" / "Fiber_results.csv"
    fiber_modeling_data = np.loadtxt(fiber_modeling_fname, delimiter=",", skiprows=1)
    error = fiber_modeling_data[:, -2]
    best_fit_ind = np.where(error == np.min(error))[0][0]
    alpha_endo_lv = fiber_modeling_data[best_fit_ind, 2]
    alpha_epi_lv = fiber_modeling_data[best_fit_ind, 3]
    fiber_angles = {
        "alpha_endo_lv": alpha_endo_lv,
        "alpha_epi_lv": alpha_epi_lv,
        "beta_endo_lv": -15,
        "beta_epi_lv": 15
    }
    return fiber_angles

def update_fiber(sample_ID, fiber_angles, results_folder):
    sample_dir = Path(f"01_results_coarse_mesh/OP{sample_ID}/TPM")
    results_dir  = sample_dir / results_folder
    pv_dir = sample_dir / "01_PVCalibration/"
    geo_dir = pv_dir / "Geometries"
    geo_fname = geo_dir / "geometry_0.h5"
    geo = pulse.HeartGeometry.from_file(geo_fname)

    # This is a string on the form {family}_{degree}
    fiber_space = "DG_0"

    # Convert markers to correct format
    markers = {
        "base": geo.markers["BASE"][0],
        "lv": geo.markers["ENDO"][0],
        "epi": geo.markers["EPI"][0],
    }

    # Compute the microstructure
    logger.info("Computing fiber angles...")
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=geo.mesh,
        fiber_space=fiber_space,
        ffun=geo.ffun,
        markers=markers,
        log_level=30,
        **fiber_angles,
    )

    pulse_logger = logging.getLogger("pulse")
    pulse_logger.setLevel(logging.WARNING)
    geo.microstructure = pulse.Microstructure(f0=fiber, s0=sheet, n0=sheet_normal)
    geo_fname = results_dir / "geometry_0_modified_fiber.h5"
    geo.save(geo_fname.as_posix(), overwrite_file=True)

    fname = results_dir / "ffun_0_modified_fiber.xdmf"
    with dolfin.XDMFFile(fname.as_posix()) as f:
        f.write(geo.mesh)

    fname = results_dir / "fiber_0_modified_fiber.xdmf"
    ldrb.fiber_to_xdmf(geo.f0, fname.as_posix())

    return geo_fname.as_posix()

#%%
sample_nums = [10, 15, 17, 18, 19, 21, 22, 24, 25, 26, 28, 29, 44, 45, 51, 53]
sample_IDs = ["135_2", "136_1", "136_2", "136_3", "138_1", "138_2", "139_1", "139_2",
              "140_2", "141_1", "141_2", "142_2", "163_3", "169_1", "169_3", "172_1", "183_1", "183_2", "185_1",
              "185_2", "186_2", "187_1"]

sample_IDs = ["135_2"]
results_folder = f"02_EDPVR_Modeling_v2"
cpu_num = 8

a_af_list = grid_triangle_biased(N=10, amin=0.05, amax=5, afmin=0.05, afmax=5, bias_power=1.4)
fig, ax = plot_triangle(a_af_list,)
fig.savefig("triangle_grid_points.png", dpi=300)
a_af_list = a_af_list[::-1]  # Reverse the list to start from the largest a and af
bf_list = [0.001]
update_fiber_flag = True
fiber_modeling_flag = True
results_folder = results_folder + "_best_fiber_fit" if update_fiber_flag else results_folder
epi_fibers = [-30, -35, -40, -45, -50, -55, -60]
endo_fibers = [30, 35, 40, 45, 50, 55, 60]

for bf in bf_list:
    for sample_ID in sample_IDs:
        subprocess.run(f"python3 dynacomp/pv_analyzer.py -i {sample_ID} ", shell=True, check=True)
        subprocess.run(f"python3 dynacomp/pv_calibrator.py -i {sample_ID} ", shell=True, check=True)
        subprocess.run(f"python3 dynacomp/pv_shifter.py -i {sample_ID} ", shell=True, check=True)
        if fiber_modeling_flag:
            print("------------------------------")
            print(f"Processing sample {sample_ID}")
            print("------------------------------")
            for epi_fiber in epi_fibers:
                for endo_fiber in endo_fibers:
                    print("------------------------------")
                    print(f"Processing {sample_ID} with epi_fiber={epi_fiber} and endo_fiber={endo_fiber}")
                    print("------------------------------")
                    output_folder_fiber_modeling = f"03_Fiber_Modeling/epi_{epi_fiber}_endo_{endo_fiber}"
                    output_dir_fiber_modeling = Path(f"/home/shared/01_results_coarse_mesh/OP{sample_ID}/TPM") / output_folder_fiber_modeling
                    if output_dir_fiber_modeling.exists():
                        print(f"Output directory {output_dir_fiber_modeling} already exists. Skipping fiber modeling for this configuration.")
                        continue
                    try:
                        subprocess.run(f"mpirun -n 8 python3 dynacomp/processing.py --fiber_modeling_flag -i {sample_ID} -o {output_folder_fiber_modeling} --epi_fiber {epi_fiber} --endo_fiber {endo_fiber}", shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {sample_ID}: {e}")
                    try:
                        subprocess.run(f"python3 dynacomp/validator.py -i {sample_ID} -o {output_folder_fiber_modeling} --logging_flag", shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error validating {sample_ID}: {e}")
                    try:
                        subprocess.run(f"python3 dynacomp/create_fibparam_sweep_contour.py -i {sample_ID} -o 03_Fiber_Modeling", shell=True, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error creating contour for {sample_ID}: {e}")
                
        if update_fiber_flag:
            fiber_angles = load_fiber_modeling(sample_ID)
            geo_fname = update_fiber(sample_ID, fiber_angles, results_folder)
        else:
            geo_fname = None

        for n, (a, af) in enumerate(a_af_list):
            logger.info(f"Running unloading and inflator for a={a}, af={af}, bf={bf}")
            output_folder = f"{results_folder}/a_{a}_af_{af}_bf_{bf}"
            try:
                subprocess.run(
                    f"mpirun -n {cpu_num} python3 dynacomp/unloading.py "
                    # f"python3 dynacomp/unloading.py "
                    f"-i {sample_ID} "
                    f"-o {output_folder} "
                    f"--a_matparam {a} "
                    f"--af_matparam {af} "
                    f"--bf_matparam {bf} "
                    f"--geometry_fname {geo_fname} ",
                    shell=True, check=True
                )

                subprocess.run(
                    f"mpirun -n {cpu_num} python3 dynacomp/inflator.py "
                    f"-i {sample_ID} "
                    f"-o {output_folder} "
                    f"--a_matparam {a} "
                    f"--af_matparam {af} "
                    f"--bf_matparam {bf} "
                    f"-lp",
                    shell=True, check=True
                )
                if n > 2:
                    subprocess.run(
                        f"python3 dynacomp/create_matparam_sweep_contour.py -i {sample_ID} -c 30 --bf_flag -o {results_folder}",
                        shell=True, check=True
                    )

                logger.info("-----------------------------------------")
                logger.info(f"Unloading-Inflating is done for a={a}, af={af}, bf={bf}")
                logger.info("-----------------------------------------")
            except subprocess.CalledProcessError as e:
                logger.error("=======================================")
                logger.error(f"Error for a={a}, af={af}, bf={bf}: {e}. Skipping this combination.")
                logger.error("=======================================")
                continue
