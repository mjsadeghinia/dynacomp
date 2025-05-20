import argparse
import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


#%%
def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)

def plot_error_contours(data, outname,
                        levels=10,
                        cmap=None,
                        a_min=None, a_max=None, a_steps=None,
                        af_min=None, af_max=None, af_steps=None):
    """
    Plot error contours from raw (a, af, error) data by reconstructing the grid.
    Optionally specify parameter ranges directly via min/max/steps.

    Parameters:
    -----------
    data : array-like, shape (n, 3)
        Columns correspond to [a, af, error] from a full parameter sweep.
        Must form a complete grid: len(a_vals) * len(af_vals) == n.
    outname : str
        File path (including filename) where the plot will be saved.
    levels : int or sequence, optional
        If int, number of contour levels between min and max error.
        If sequence, explicit contour level values.
    cmap : str or Colormap, optional
        Colormap to use (e.g., 'viridis').
    a_min, a_max, a_steps : float, float, int, optional
        If provided, generate `a_vals = np.linspace(a_min, a_max, a_steps)`.
        Otherwise `a_vals` is inferred from unique values in data[:,0].
    af_min, af_max, af_steps : float, float, int, optional
        If provided, generate `af_vals = np.linspace(af_min, af_max, af_steps)`.
        Otherwise `af_vals` is inferred from unique values in data[:,1].
    """
    # Convert to array
    data = np.asarray(data)
    # Determine a_vals and af_vals
    if a_min is not None and a_max is not None and a_steps is not None:
        a_vals = np.linspace(a_min, a_max, a_steps)
    else:
        a_vals = np.unique(data[:, 0])

    if af_min is not None and af_max is not None and af_steps is not None:
        af_vals = np.linspace(af_min, af_max, af_steps)
    else:
        af_vals = np.unique(data[:, 1])

    n, m = len(a_vals), len(af_vals)
    if data.shape[0] != n * m:
        raise ValueError(f"Data does not form a complete grid: expected {n*m} points, got {data.shape[0]}")

    # Build error grid according to specified ordering
    error_grid = np.empty((m, n))
    for i, a in enumerate(a_vals):
        for j, af in enumerate(af_vals):
            # find matching data row
            mask = (np.isclose(data[:, 0], a) & np.isclose(data[:, 1], af))
            if not np.any(mask):
                raise ValueError(f"Missing error value for a={a}, af={af}")
            error_grid[j, i] = data[mask, 2]

    # Create meshgrid for plotting
    A, AF = np.meshgrid(a_vals, af_vals)

    # Determine error range
    err_min, err_max = np.nanmin(error_grid), np.nanmax(error_grid)

    # Define contour levels
    if isinstance(levels, int):
        levels = np.linspace(err_min, err_max, levels)

    # Plot contours
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(A, AF, error_grid,
                     levels=levels,
                     cmap=cmap,
                     vmin=err_min,
                     vmax=err_max)
    ax.set_xlabel('a')
    ax.set_ylabel('af')
    ax.set_title('Parameter Sweep Error Contours')

    # Add colorbar
    cbar = fig.colorbar(cs, ax=ax, label='Error (%)')

    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    plt.close(fig)

#%%
def main():
    parser = argparse.ArgumentParser(description="2D parameter sweep of (a, a_f) for HeartModelDynaComp")
    parser.add_argument(
        '-n',
        '--number',
        nargs='*',
        type=int,
        default=None,
        help='Sample number(s) to process. If omitted, all samples in settings_dir will be processed.'
    )
    parser.add_argument(
        '--settings_dir',
        type=Path,
        default=Path('/home/shared/dynacomp/settings'),
        help='Directory where JSON settings files are stored.'
    )
    parser.add_argument(
        '-s',
        '--scan_type',
        type=str,
        default='TPM',
        help='Scan type; subdirectories will be named accordingly.'
    )
    parser.add_argument(
        '-r',
        '--results_dir',
        type=Path,
        default=Path('/home/shared/01_results_coarse_mesh'),
        help='Directory where results will be saved.'
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=8, 
        help="Number of points along each axis"
        )
    parser.add_argument(
        '--a_min',
        default=0.25,
        type=float,
        help='Minimum a-value for the sweep grid.'
    )
    parser.add_argument(
        '--a_max',
        type=float,
        default=5,
        help='Maximum a-value for the sweep grid.'
    )
    parser.add_argument(
        '--af_min',
        type=float,
        default=0.5,
        help='Minimum a_f-value for the sweep grid.'
    )
    parser.add_argument(
        '--af_max',
        type=float,
        default=10,
        help='Maximum a_f-value for the sweep grid.'
    )


    args = parser.parse_args()

    settings_dir = args.settings_dir
    scan_type = args.scan_type
    results_dir = args.results_dir
    grid_size = args.grid_size
    a_min, a_max = args.a_min, args.a_max
    af_min, af_max = args.af_min, args.af_max

    if args.number:
        sample_list = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(files) + 1))

    for sample in sample_list:
        settings = load_settings(settings_dir, sample)
        sample_id = settings['id']
        # Prepare directories
        out_dir = results_dir / sample_id / scan_type
        fname = out_dir / f"EDPVR_parameter_sweeps.txt"
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        out_path = results_dir / sample_id / scan_type / "ParameterSweep_contours.png"
        plot_error_contours(data, out_path)
        
        

if __name__ == "__main__":
    main()
