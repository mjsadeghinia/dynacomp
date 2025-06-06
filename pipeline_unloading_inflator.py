import subprocess
import numpy as np
from structlog import get_logger
import matplotlib.pyplot as plt

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

def plot_triangle(a_af_lists, colors=None, labels=None):
    """
    Plot the triangle defined by the vertices (amin, afmin), (amax, afmin) and (amin, afmax)
    and the points in a_af_lists (can be a single list or a list of lists).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    if isinstance(a_af_lists, np.ndarray):
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
sample_nums = [13]
results_folder = f"02_EDPVR_Modeling"
cpu_num = 8

a_af_list = grid_triangle(N=50, amin=0.25, amax=5, afmin=0.25, afmax=5)
a_af_list_2 = grid_triangle(N=30, amin=0.05, amax=2, afmin=0.05, afmax=2)

fig, ax = plot_triangle([a_af_list, a_af_list_2],)
fig.savefig("triangle_grid_points.png", dpi=300)

a_af_list = a_af_list[::-1]  # Reverse the list to start from the largest a and af
a_af_list_2 = a_af_list_2[::-1]  # Reverse the list to start from the largest a and af
bf_list = [0.001]

for sample_num in sample_nums:
    for bf in bf_list:
        for n, (a, af) in enumerate(a_af_list_2):
            logger.info(f"Running unloading and inflator for a={a}, af={af}, bf={bf}")
            output_folder = f"{results_folder}/a_{a}_af_{af}_bf_{bf}"
            try:
                subprocess.run(
                    f"mpirun -n {cpu_num} python3 dynacomp/unloading.py "
                    f"-n {sample_num} "
                    f"-o {output_folder} "
                    f"--a_matparam {a} "
                    f"--af_matparam {af} "
                    f"--bf_matparam {bf} ",
                    shell=True, check=True
                )

                subprocess.run(
                    f"mpirun -n {cpu_num} python3 dynacomp/inflator.py "
                    f"-n {sample_num} "
                    f"-o {output_folder} "
                    f"--a_matparam {a} "
                    f"--af_matparam {af} "
                    f"--bf_matparam {bf} "
                    f"-lp",
                    shell=True, check=True
                )
                if n > 2:
                    subprocess.run(
                        f"python3 dynacomp/create_matparam_sweep_contour.py -n {sample_num} -c 30 --bf_flag",
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
