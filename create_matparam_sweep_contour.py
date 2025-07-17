import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import plotly.graph_objects as go
from scipy.interpolate import griddata


def load_settings(settings_dir: Path, sample_num: int) -> dict:
    """
    Load the JSON settings file for a given sample index (1-based).
    """
    files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
    with open(files[sample_num - 1], 'r') as f:
        return json.load(f)

def _plot_contour_slice(a, a_f, error, output_path, contour_levels, interpolation):
    """
    Helper to plot and save a single (a_f, a) vs. error contour slice.
    """
    # Build triangulation and interpolator
    triang = mtri.Triangulation(a_f, a)
    if interpolation == 'cubic':
        interp = mtri.CubicTriInterpolator(triang, error)
    else:
        interp = mtri.LinearTriInterpolator(triang, error)

    # Create regular grid
    a_f_lin = np.linspace(np.min(a_f), np.max(a_f), 200)
    a_lin = np.linspace(np.min(a), np.max(a), 200)
    a_f_grid, a_grid = np.meshgrid(a_f_lin, a_lin)
    error_grid = interp(a_f_grid, a_grid)

    # Find minimum‐error point in this slice
    min_idx = np.argmin(error)
    best_a = a[min_idx]
    best_a_f = a_f[min_idx]

    # Start plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(0, np.nanmax(error), contour_levels)

    # Contour lines (only lowest 5 labeled)
    cs = ax.contour(a_f_grid, a_grid, error_grid,
                    levels=levels, colors='black', linewidths=0.25)
    lowest_levels = cs.levels[:5]
    ax.clabel(cs, levels=lowest_levels, fmt="%.2f", fontsize=8)

    # Filled contour
    cf = ax.contourf(a_f_grid, a_grid, error_grid,
                     levels=levels, cmap='viridis', alpha=0.7)

    ax.scatter(a_f, a, c='white', edgecolor='black', s=10, linewidth=0.5, label='Data points')
    # Mark best fit
    ax.scatter(best_a_f, best_a, c='red', edgecolor='black', s=10, linewidth=0.5, label='Best Fit')

    # Labels and limits
    ax.set_xlabel('a_f')
    ax.set_ylabel('a')
    ax.set_xlim(0, np.max(a_f) + 1)
    ax.set_ylim(0, np.max(a) + 1)
    ax.legend(loc='upper right')

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label('RMS Error (kPa)')

    # Save and close
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_contour(data, output_dir: Path, filename='error_contour.png',
                       contour_levels=20, interpolation='cubic', bf_flag=False):
    """
    Creates interpolated contour plot(s) of error over (a_f, a).
    If bf_flag is False (default), produces a single plot using all data.
    If bf_flag is True, creates one plot per unique b_f value.
    Saves files in output_dir with names based on filename.
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Columns in data: [a, a_f, b, b_f, error]
    a     = data[:, 0]
    a_f   = data[:, 1]
    b_f   = data[:, 3]
    error = data[:, 4]

    if bf_flag:
        # Generate one plot for each unique b_f
        for bf in np.unique(b_f):
            mask = np.isclose(b_f, bf)
            a_slice     = a[mask]
            a_f_slice   = a_f[mask]
            error_slice = error[mask]

            # Build a filename that includes the b_f value
            stem, ext = Path(filename).stem, Path(filename).suffix
            safe_bf = str(bf).replace('.', '_')
            out_name = f"{stem}_bf_{safe_bf}{ext}"
            out_path = output_dir / out_name

            _plot_contour_slice(
                a_slice,
                a_f_slice,
                error_slice,
                out_path,
                contour_levels,
                interpolation
            )
    else:
        # Single plot using all data
        out_path = output_dir / filename
        _plot_contour_slice(
            a,
            a_f,
            error,
            out_path,
            contour_levels,
            interpolation
        )


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
        '-o',
        "--output_folder",
        default="02_EDPVR_Modeling",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )
    parser.add_argument(
        '-c', '--contour_levels',
        type=int,
        default=20,
        help='The number of contour lines.'
    )
    parser.add_argument(
        '--interpolation',
        type=str,
        choices=['linear', 'cubic'],
        default='cubic',
        help='Interpolation method for contours.'
    )
    parser.add_argument(
        '--bf_flag',
        action='store_true',
        help='If set, create and save one plot per unique b_f value.'
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
        out_dir  = args.results_dir / sample_id / args.scan_type / args.output_folder
        data_dir = out_dir
        fname    = data_dir / "inflation_results.txt"

        # Load sweep data
        data = np.loadtxt(fname, skiprows=1, delimiter=',')

        # Plot and save contours (single or small multiples)
        plot_error_contour(
            data,
            data_dir,
            filename=f'error_contour.png',
            contour_levels=args.contour_levels,
            interpolation=args.interpolation,
            bf_flag=args.bf_flag
        )

if __name__ == "__main__":
    main()
