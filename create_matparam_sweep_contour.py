import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import plotly.graph_objects as go
from scipy.interpolate import griddata
from structlog import get_logger

logger = get_logger()
import utils

def fit_quadratic_surface(x, y, z):
    """
    Fit z = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 via least squares.
    Returns (coeffs, zhat(xv,yv), analytic_xy or None).
    """
    X = np.column_stack([np.ones_like(x), x, y, x**2, x*y, y**2])
    coeffs, *_ = np.linalg.lstsq(X, z, rcond=None)
    c0, c1, c2, c3, c4, c5 = coeffs

    def zhat(xv, yv):
        return c0 + c1*xv + c2*yv + c3*xv**2 + c4*xv*yv + c5*yv**2

    A = np.array([[2*c3, c4],
                  [c4,   2*c5]], dtype=float)
    b = -np.array([c1, c2], dtype=float)
    analytic_xy = None
    try:
        analytic_xy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pass
    return coeffs, zhat, analytic_xy

def tri_interp_grid(x, y, z, method, nx=200, ny=200):
    """
    Triangulation-based interpolation ('linear' or 'cubic').
    Returns Xi, Yi, Zi.
    """
    xi = np.linspace(np.min(x), np.max(x), nx)
    yi = np.linspace(np.min(y), np.max(y), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # Clean: remove non-finite and duplicate (x,y)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[m]; y = y[m]; z = z[m]
    if x.size < 3:
        Zi = np.full_like(Xi, np.nan, dtype=float)
        return Xi, Yi, Zi
    xy = np.column_stack([x, y])
    _, idx = np.unique(xy, axis=0, return_index=True)
    x = x[idx]; y = y[idx]; z = z[idx]

    triang = mtri.Triangulation(x, y)
    # Mask skinny/flat triangles that can break the trifinder
    mask = mtri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
    triang.set_mask(mask)

    try:
        if method == "cubic":
            interp = mtri.CubicTriInterpolator(triang, z)
        else:
            interp = mtri.LinearTriInterpolator(triang, z)
        Zi = interp(Xi, Yi)
    except RuntimeError:
        # Fallback if triangulation is still invalid on this Matplotlib build
        Zi = griddata(np.column_stack((x, y)), z, (Xi, Yi),
                      method="cubic" if method == "cubic" else "linear")
    return Xi, Yi, Zi


def plot_contours(
    x, y, z, Xi, Yi, Zi, output_path: Path, contour_levels=25,
    xlabel="a_f", ylabel="a", zlabel="RMS Error (kPa)",
    mark_analytic=None, manual_bestfit=None, clim=None
):
    # Levels from data range (avoid NaNs)
    if clim is None:
        vmin, vmax = np.nanmin(Zi), np.nanmax(Zi)
    else:
        vmin, vmax = clim
    levels = np.linspace(vmin, vmax, contour_levels)
    Zplot = np.array(Zi, copy=True)
    Zplot = np.clip(Zplot, vmin, vmax)

    # Plot base
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(Xi, Yi, Zplot, levels=levels, colors='black', linewidths=0)
    if hasattr(cs, "levels") and len(cs.levels) > 0:
        ax.clabel(cs, levels=cs.levels[:5], fmt="%.2f", fontsize=8)

    cf = ax.contourf(Xi, Yi, Zplot, levels=levels, cmap='viridis_r', alpha=0.75)
    # Data points (keep original size/color) + min(data)
    ax.scatter(x, y, c='white', edgecolor='black', s=20, linewidth=0.7, label='Data points')
    if np.isfinite(z).any():
        sort_idx = np.argsort(z)
        min_idx = sort_idx[0]
        ax.scatter(x[min_idx], y[min_idx], c='red', edgecolor='black', s=20, linewidth=0.7, label='Best Fit')
        if manual_bestfit is not None:
            sel_idx = sort_idx[int(manual_bestfit)-1]
            ax.scatter(x[sel_idx], y[sel_idx], c='yellow', edgecolor='black', s=20, linewidth=0.7, label='Selected Best Fit')

    # Optional analytic minimum marker (x) if provided & in bounds
    if mark_analytic is not None:
        xa, ya = mark_analytic
        if (Xi.min() <= xa <= Xi.max()) and (Yi.min() <= ya <= Yi.max()):
            ax.scatter(xa, ya, s=30, color='red', marker='x', linewidths=1.0, label='Analytic min (quad)')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label(zlabel + " — contour")
    ax.grid(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def process_one_csv(
    csv_path: Path,
    output_dir: Path,
    filename: str,
    method: str,
    contour_levels: int,
    grid_nx: int = 200,
    grid_ny: int = 200,
    manual_bestfit=None,
    bf_value=None,
    clim=None
):
    """
    Read inflation_results.txt (CSV-style), build grid by chosen method,
    and plot contours using script-2-like flow but with original markers.
    Columns in data: [a, a_f, b, b_f, error]
    """
    data = np.loadtxt(csv_path, skiprows=1, delimiter=',')
    if bf_value is not None:
        mask = np.isclose(data[:, 3], bf_value)
        data = data[mask]
    a = data[:, 0]
    a_f = data[:, 1]
    err = data[:, 4]

    # x = a_f, y = a, z = error
    x = a_f
    y = a
    z = err

    if method == "quadratic":
        _, zhat, analytic_xy = fit_quadratic_surface(x, y, z)
        xi = np.linspace(np.min(x), np.max(x), grid_nx)
        yi = np.linspace(np.min(y), np.max(y), grid_ny)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = zhat(Xi, Yi)
        mark_xy = analytic_xy
    else:
        Xi, Yi, Zi = tri_interp_grid(x, y, z, method=method, nx=grid_nx, ny=grid_ny)
        mark_xy = None

    out_path = output_dir / filename
    plot_contours(
        x, y, z, Xi, Yi, Zi, out_path,
        contour_levels=contour_levels,
        xlabel='a_f', ylabel='a', zlabel='RMS Error (kPa)',
        mark_analytic=mark_xy, manual_bestfit=manual_bestfit, clim=clim
    )
#%% Main function
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
        "-i",
        "--sample_ID",
        nargs="+",
        type=str,
        help="The sample ID to be processd, if passed in the sample number will be ignored.",
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
        '--method',
        type=str,
        choices=['linear', 'cubic', 'quadratic'],
        default='cubic',
        help='Interpolation method for contours.'
    )
    parser.add_argument(
        '--bf_flag',
        action='store_true',
        help='If set, create and save one plot per unique b_f value.'
    )
    parser.add_argument(
        '--clim',
        nargs=2,
        type=float,
        default=None,
        help='Color limits for contour plot (vmin vmax).'
    )

    args = parser.parse_args()

    # Determine samples to process
    if args.sample_ID is not None:
        sample_nums = []
        for id in args.sample_ID:
            id_num = utils.get_num_from_id(id, args.settings_dir)
            sample_nums.append(id_num)
    elif args.number:
        sample_nums = args.number
    else:
        files = sorted([f for f in args.settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(files) + 1))

    for sample in sample_nums:
        settings = utils.load_settings(args.settings_dir, sample)
        sample_id = settings['id']

        # Prepare directories and file paths
        out_dir  = args.results_dir / sample_id / args.scan_type / args.output_folder
        data_dir = out_dir
        fname    = data_dir / "inflation_results.txt"
        if not fname.exists():
            logger.warning(f"File does not exist, skipping sample {sample_id}.")
            continue
        logger.info(f"Processing sample {sample_id}")

        # New: process_one_csv flow (quadratic/linear/cubic), plotting like script 2 (markers unchanged)
        if args.bf_flag:
            data_all = np.loadtxt(fname, skiprows=1, delimiter=',')
            unique_bf = np.unique(data_all[:, 3])
            for bf in unique_bf:
                safe_bf = str(bf).replace('.', '_')
                process_one_csv(
                    csv_path=fname,
                    output_dir=data_dir,
                    filename=f'error_contour_bf_{safe_bf}.png',
                    method=args.method,
                    contour_levels=args.contour_levels,
                    grid_nx=50,
                    grid_ny=50,
                    manual_bestfit=settings["PV"].get('EDPVR_modeling_manual_bestfit', None),
                    bf_value=bf,
                    clim=args.clim
                )
        else:
            process_one_csv(
                csv_path=fname,
                output_dir=data_dir,
                filename='error_contour.png',
                method=args.method,
                contour_levels=args.contour_levels,
                grid_nx=200,
                grid_ny=200,
                manual_bestfit=settings["PV"].get('EDPVR_modeling_manual_bestfit', None),
                clim=args.clim
            )

if __name__ == "__main__":
    main()
