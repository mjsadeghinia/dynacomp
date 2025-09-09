import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import utils

#%%
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

    # Analytic minimum: solve grad = 0
    A = np.array([[2*c3, c4],
                  [c4,   2*c5]], dtype=float)
    b = -np.array([c1, c2], dtype=float)
    analytic_xy = None
    try:
        analytic_xy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pass

    return coeffs, zhat, analytic_xy


def tri_interp_grid(x, y, z, method, nx=300, ny=300):
    """
    Triangulation-based interpolation ('linear' or 'cubic').
    Returns Xi, Yi, Zi.
    """
    xi = np.linspace(np.min(x), np.max(x), nx)
    yi = np.linspace(np.min(y), np.max(y), ny)
    Xi, Yi = np.meshgrid(xi, yi)
    triang = mtri.Triangulation(x, y)
    if method == "cubic":
        interp = mtri.CubicTriInterpolator(triang, z)
    else:
        interp = mtri.LinearTriInterpolator(triang, z)
    Zi = interp(Xi, Yi)
    return Xi, Yi, Zi

def plot_contours(
    x, y, z, Xi, Yi, Zi, output_path: Path, contour_levels=25,
    xlabel="epi_fib", ylabel="endo_fib", zlabel="total distance (mean)",
    mark_analytic=None, clim=None
):
    # Choose levels (avoid NaN min/max)
    data_min, data_max = np.nanmin(Zi), np.nanmax(Zi)
    if clim is None:
        vmin, vmax = data_min, data_max
    else:
        vmin, vmax = clim
    levels = np.linspace(vmin, vmax, contour_levels)

    # Best fit (grid min)
    min_idx = np.unravel_index(np.nanargmin(Zi), Zi.shape)
    x_best, y_best = Xi[min_idx], Yi[min_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(Xi, Yi, Zi, levels=levels, colors="black", linewidths=.5)
    if hasattr(cs, "levels") and len(cs.levels) > 0:
        ax.clabel(cs, levels=cs.levels[:5], fmt="%.2f", fontsize=7)

    cf = ax.contourf(Xi, Yi, Zi, levels=levels, alpha=0.75, cmap='viridis_r')

    # Data points (white face, black outline) + minimum (data) in red circles
    ax.scatter(x, y, s=20, facecolors="white", edgecolors="black",
               linewidths=0.7, marker="o", label="Data points")
    min_mask = np.isfinite(z) & np.isclose(z, np.nanmin(z))
    ax.scatter(x[min_mask], y[min_mask], s=20, facecolors="red", edgecolors="black",
               linewidths=0.7, marker="o", label="Minimum (data)")

    # Optional analytic minimum marker (if inside bounds)
    if mark_analytic is not None:
        xa, ya = mark_analytic
        if (Xi.min() <= xa <= Xi.max()) and (Yi.min() <= ya <= Yi.max()):
            ax.scatter(xa, ya, s=70, color="red",
                       linewidths=1.2, marker="x", label="Analytic min (quad)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
    grid_nx: int,
    grid_ny: int,
    show_analytic_min: bool,
    xname="epi_fib",
    yname="endo_fib",
    zname="total_distance (mean)",
    clim=None
):
    data = np.loadtxt(csv_path, skiprows=1, delimiter=',')
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 9]

    if method == "quadratic":
        _, zhat, analytic_xy = fit_quadratic_surface(x, y, z)
        xi = np.linspace(np.min(x), np.max(x), grid_nx)
        yi = np.linspace(np.min(y), np.max(y), grid_ny)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = zhat(Xi, Yi)
        mark_xy = analytic_xy if show_analytic_min else None
    else:
        Xi, Yi, Zi = tri_interp_grid(x, y, z, method=method, nx=grid_nx, ny=grid_ny)
        mark_xy = None

    out_path = output_dir / filename
    plot_contours(
        x, y, z, Xi, Yi, Zi, out_path,
        contour_levels=contour_levels,
        xlabel=xname, ylabel=yname, zlabel=zname,
        mark_analytic=mark_xy, clim=clim
    )

#%%
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
        default="03_Fiber_Modeling",
        type=str,
        help="The result folder name tha would be created in the directory of the sample.",
    )
    parser.add_argument(
        "--csv_name",
        default="Fiber_results.csv",
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
        "--method", 
        type=str, 
        choices=["quadratic", "linear", "cubic"],
        default="quadratic", 
        help="Interpolation method (default: quadratic second-order fit)."
    )

    parser.add_argument(
        "--nx", 
        type=int, 
        default=300, 
        help="Grid points in X."
        )
    
    parser.add_argument(
        "--ny", 
        type=int,
        default=300, 
        help="Grid points in Y."
        )
    
    parser.add_argument(
        "--show_analytic_min", 
        action="store_true",
        help="Mark analytic minimum for quadratic (if inside bounds)."
    )

    parser.add_argument(
        "--xcol", 
        type=str, 
        default="epi_fib", 
        help="X column name."
    )
    parser.add_argument(
        "--ycol", 
        type=str, 
        default="endo_fib", 
        help="Y column name."
    )
    parser.add_argument(
        "--zcol", 
        type=str, 
        default="total_distance (mean)", 
        help="Z column name."
    )

    parser.add_argument(
        "--clim", 
        nargs=2, 
        type=float, 
        default=None,
        help="Color limits for contour plot (vmin vmax)."
    )

    args = parser.parse_args()   

    settings_dir = args.settings_dir
    scan_type = args.scan_type
    output_folder = args.output_folder
    results_dir = args.results_dir
    csv_name = args.csv_name
    contour_levels = args.contour_levels
    method = args.method
    nx = args.nx
    ny = args.ny
    show_analytic_min = args.show_analytic_min
    xname = args.xcol
    yname = args.ycol
    zname = args.zcol
    clim = args.clim

    # Determine samples to process
    if args.sample_ID is not None:
        sample_nums = []
        for id in args.sample_ID:
            id_num = utils.get_num_from_id(id, settings_dir)
            sample_nums.append(id_num)
    elif args.number:
        sample_nums = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(files) + 1))

    for sample in sample_nums:
        settings = utils.load_settings(settings_dir, sample)
        sample_id = settings['id']

        out_dir = results_dir / sample_id / scan_type / output_folder
        csv_path = out_dir / csv_name

        process_one_csv(
            csv_path=csv_path,
            output_dir=out_dir,
            filename="Fiber_contour.png",
            method=method,
            contour_levels=contour_levels,
            grid_nx=nx,
            grid_ny=ny,
            show_analytic_min=show_analytic_min,
            xname=xname, yname=yname, zname=zname, clim=clim
        )


if __name__ == "__main__":
    main()
