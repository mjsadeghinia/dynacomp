import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)


def plot_error_contour(data, output_dir, filename='error_contour.png', contour_levels=20, interpolation='cubic'):
    """
    Creates an interpolated contour plot of error over (a_f, a) from the provided data,
    marks the minimum-error point with a red cross, and saves the figure in output_dir/filename.

    Parameters:
    - data: numpy array with columns [a, a_f, b, b_f, error].
    - output_dir: directory path (string) where the plot will be saved.
    - filename: name of the output file (string), default 'error_contour.png'.

    Returns:
    - output_path: full path to the saved file.
    """
    a = data[:, 0]
    a_f = data[:, 1]
    error = data[:, 4]

    # Triangulate and create linear interpolator
    triang = mtri.Triangulation(a_f, a)
    if interpolation == 'cubic':
        # Use cubic interpolation if specified
        interp = mtri.CubicTriInterpolator(triang, error)
    else:
        # Default to linear interpolation
        interp = mtri.LinearTriInterpolator(triang, error)

    # Build regular grid for interpolation
    a_f_lin = np.linspace(np.min(a_f), np.max(a_f), 200)
    a_lin = np.linspace(np.min(a), np.max(a), 200)
    a_f_grid, a_grid = np.meshgrid(a_f_lin, a_lin)
    error_grid = interp(a_f_grid, a_grid)

    min_idx = np.argmin(error)
    best_a = a[min_idx]
    best_a_f = a_f[min_idx]
    best_error = error[min_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(np.nanmin(error), np.nanmax(error), contour_levels)

    cs = ax.contour(a_f_grid, a_grid, error_grid, levels=levels, colors='black', linewidths=0.25)
    # Label only the first 5 (lowest) contour lines
    lowest_levels = cs.levels[:5]
    ax.clabel(cs, levels=lowest_levels, fmt="%.2f", fontsize=8)
    # Draw filled contours
    cf = ax.contourf(a_f_grid, a_grid, error_grid, levels=levels, cmap='viridis', alpha=0.7)

    ax.scatter(a_f, a, c='white', edgecolor='black', s=40, label='Data points')
    ax.scatter(best_a_f, best_a, c='red', edgecolor='black', s=40, label='Best Fit')  

    # Labels, title, grid, legend
    ax.set_xlabel('a_f')
    ax.set_ylabel('a')
    ax.set_xlim(0,np.max(a_f) + 1)
    ax.set_ylim(0,np.max(a) + 1)
    ax.legend(loc='upper right')

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('RMS Error (kPa)')

    # Save and close
    fname = output_dir / filename
    fig.savefig(fname, dpi=300)
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
        '-c', '--contour_levels',
        type=int,
        default=20,
        help='The number of contour lines.'
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
        data_dir = out_dir / "02_EDPVR_Modeling"
        fname = data_dir / "inflation_results.txt"
        # Load sweep data
        data = np.loadtxt(fname, skiprows=1, delimiter=',')
        plot_error_contour(data, data_dir, filename=f'error_contour_sample_{sample_id}.png', contour_levels=args.contour_levels)
        

if __name__ == "__main__":
    main()
