import subprocess
import numpy as np
from structlog import get_logger
from pathlib import Path
import matplotlib.pyplot as plt

import utils

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

#%%
sample_nums = [10, 15, 17, 18, 19, 21, 22, 24, 25, 26, 28, 29, 44, 45, 51, 53]
sample_IDs = ["131_1", "133_1", "136_1", "136_2", "136_3", "138_1", "138_2", "139_1", "139_2", "140_2", "142_2", "142_2", "169_1", "169_3", "183_1", "185_1"]

results_folder = f"02_EDPVR_Modeling"
cpu_num = 8

a_af_list = grid_triangle_biased(N=10, amin=0.05, amax=5, afmin=0.05, afmax=5, bias_power=1.4)
fig, ax = plot_triangle(a_af_list,)
fig.savefig("triangle_grid_points.png", dpi=300)
a_af_list = a_af_list[::-1]  # Reverse the list to start from the largest a and af
bf_list = [0.001]

for bf in bf_list:
    for sample_ID in sample_IDs:
        for n, (a, af) in enumerate(a_af_list):
            logger.info(f"Running unloading and inflator for a={a}, af={af}, bf={bf}")
            output_folder = f"{results_folder}/a_{a}_af_{af}_bf_{bf}"
            try:
                subprocess.run(
                    f"mpirun -n {cpu_num} python3 dynacomp/unloading.py "
                    f"-i {sample_ID} "
                    f"-o {output_folder} "
                    f"--a_matparam {a} "
                    f"--af_matparam {af} "
                    f"--bf_matparam {bf} ",
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
