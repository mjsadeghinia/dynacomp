import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)


def plot_error_contours(data, outname,
                        a_min, a_max, a_steps,
                        af_min, af_max, af_steps,
                        levels=10,
                        cmap=None):
    """
    Plot error contours from raw (a, af, error) data by reconstructing the grid
    based on explicitly provided parameter ranges.

    Parameters:
    -----------
    data : array-like, shape (n, 3)
        Columns correspond to [a, af, error] from a full parameter sweep.
        Must form a complete grid: a_steps * af_steps == n.
    outname : str
        File path (including filename) where the plot will be saved.
    a_min : float
        Minimum value of parameter 'a'.
    a_max : float
        Maximum value of parameter 'a'.
    a_steps : int
        Number of points along the 'a' axis.
    af_min : float
        Minimum value of parameter 'af'.
    af_max : float
        Maximum value of parameter 'af'.
    af_steps : int
        Number of points along the 'af' axis.
    levels : int or sequence, optional
        If int, number of contour levels between min and max error.
        If sequence, explicit contour level values.
    cmap : str or Colormap, optional
        Colormap to use (e.g., 'viridis').
    """
    data = np.asarray(data)
    # Generate the parameter grids
    a_vals = np.linspace(a_min, a_max, a_steps)
    af_vals = np.linspace(af_min, af_max, af_steps)

    n, m = len(a_vals), len(af_vals)
    expected = n * m
    if data.shape[0] != expected:
        raise ValueError(f"Data size mismatch: expected {expected} points ({n}x{m} grid), got {data.shape[0]}")

    # Build error grid
    error_grid = np.empty((m, n))
    for i, a in enumerate(a_vals):
        for j, af in enumerate(af_vals):
            mask = (np.isclose(data[:, 0], a) & np.isclose(data[:, 1], af))
            if not np.any(mask):
                raise ValueError(f"Missing error value for a={a}, af={af}")
            error_grid[j, i] = data[mask, 2]

    # Create meshgrid for plotting
    A, AF = np.meshgrid(a_vals, af_vals)
    breakpoint()
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
    fig.colorbar(cs, ax=ax, label='Error (%)')

    fig.tight_layout()
    fig.savefig(outname, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="2D parameter sweep of (a, a_f) for HeartModelDynaComp"
    )
    parser.add_argument(
        '-n', '--number',
        nargs='*',
        type=int,
        default=None,
        help='Sample number(s) to process. If omitted, all samples will be processed.'
    )
    parser.add_argument(
        '--settings_dir',
        type=Path,
        default=Path('/home/shared/dynacomp/settings'),
        help='Directory where JSON settings files are stored.'
    )
    parser.add_argument(
        '-s', '--scan_type',
        type=str,
        default='TPM',
        help='Scan type; subdirectories will be named accordingly.'
    )
    parser.add_argument(
        '-r', '--results_dir',
        type=Path,
        default=Path('/home/shared/01_results_coarse_mesh'),
        help='Directory where results will be saved.'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=8,
        help='Number of points along each axis'
    )
    parser.add_argument(
        '--a_min',
        type=float,
        default=0.25,
        help='Minimum a-value for the sweep grid.'
    )
    parser.add_argument(
        '--a_max',
        type=float,
        default=5.5,
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
        default=10.5,
        help='Maximum a_f-value for the sweep grid.'
    )

    args = parser.parse_args()

    # Determine samples to process
    if args.number:
        sample_list = args.number
    else:
        files = sorted([f for f in args.settings_dir.iterdir() if f.suffix == ".json"])
        sample_list = list(range(1, len(files) + 1))

    for sample in sample_list:
        settings = load_settings(args.settings_dir, sample)
        sample_id = settings['id']

        # Prepare directories and file paths
        out_dir = args.results_dir / sample_id / args.scan_type
        data_file = out_dir / f"EDPVR_parameter_sweeps.txt"
        out_path = out_dir / "ParameterSweep_contours.png"

        # Load sweep data
        data = np.loadtxt(data_file, skiprows=1, delimiter=',')

        # Plot and save contours
        plot_error_contours(
            data, out_path,
            a_min=args.a_min, a_max=args.a_max, a_steps=args.grid_size,
            af_min=args.af_min, af_max=args.af_max, af_steps=args.grid_size,
            levels=42, cmap='viridis'
        )

if __name__ == "__main__":
    main()
