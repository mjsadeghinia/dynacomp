import os
import sys
import shlex
import subprocess
import numpy as np
from structlog import get_logger
from pathlib import Path
import argparse

import ldrb
import dolfin
import pulse
import logging

import utils

logger = get_logger()

# -----------------------------------------------------------------------------
# Slurm-aware helpers
# -----------------------------------------------------------------------------

def slurm_ntasks(default: int = 8) -> int:
    try:
        return int(os.environ.get("SLURM_NTASKS", default))
    except (TypeError, ValueError):
        return default


def _launch_srun(n: int, script_path: Path, argv: list[str]):
    cmd = ["srun", "-n", str(n), sys.executable, "-u", str(script_path)] + argv
    logger.info("launch_srun", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_py(script_path: Path, argv: list[str]):
    cmd = [sys.executable, "-u", str(script_path)] + argv
    logger.info("run_py", cmd=" ".join(cmd))
    subprocess.run(cmd, check=True)

# -----------------------------------------------------------------------------
# Geometry / grids
# -----------------------------------------------------------------------------

def biased_linspace(start, stop, N, bias_power=2):
    t = np.linspace(0, 1, N)
    t_biased = t ** bias_power
    return start + (stop - start) * t_biased


def grid_triangle_biased(N, amin=0.05, amax=5, afmin=0.05, afmax=5, bias_power=1.3):
    a_edge = biased_linspace(amin, amax, N, bias_power)
    af_edge = np.full(N, afmin)
    af_edge_h = biased_linspace(afmin, afmax, N, bias_power)
    a_edge_h = np.full(N, amin)

    a_af_list = []
    for i in range(N):
        n_div = i + 2
        for j in range(n_div):
            t = j / (n_div - 1) if n_div > 1 else 0
            a_val = a_edge_h[i] + t * (a_edge[i] - a_edge_h[i])
            af_val = af_edge_h[i] + t * (af_edge[i] - af_edge_h[i])
            a_af_list.append([round(a_val, 3), round(af_val, 3)])

    return a_af_list

# -----------------------------------------------------------------------------
# Repo paths
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[0]
DYNACOMP = REPO_ROOT

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

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
        "beta_epi_lv": 15,
    }
    return fiber_angles


def update_fiber(sample_ID, fiber_angles, results_folder):
    sample_dir = Path(f"01_results_coarse_mesh/OP{sample_ID}/TPM")
    results_dir = sample_dir / results_folder
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
    geo_out = results_dir / "geometry_0_modified_fiber.h5"
    geo.save(geo_out.as_posix(), overwrite_file=True)

    fname = results_dir / "ffun_0_modified_fiber.xdmf"
    with dolfin.XDMFFile(fname.as_posix()) as f:
        f.write(geo.mesh)

    fname = results_dir / "fiber_0_modified_fiber.xdmf"
    ldrb.fiber_to_xdmf(geo.f0, fname.as_posix())

    return geo_out.as_posix()


# -----------------------------------------------------------------------------
# Main orchestration (MPI via srun)
# -----------------------------------------------------------------------------

def run_EDPVR(sample_ID, a_af_list, bf, results_folder, settings_dir, geo_fname=None, cpu_num=8):
    unloading_py = DYNACOMP / "unloading.py"
    inflator_py = DYNACOMP / "inflator.py"
    contour_py = DYNACOMP / "create_matparam_sweep_contour.py"
    results_dir = DYNACOMP.parent / "01_results_coarse_mesh"

    for n, (a, af) in enumerate(a_af_list):
        logger.info("unloading_inflator_start", a=a, af=af, bf=bf)
        output_folder = f"{results_folder}/a_{a}_af_{af}_bf_{bf}"

        try:
            unload_args = [
                "-i", str(sample_ID),
                "-o", output_folder,
                "--a_matparam", str(a),
                "--af_matparam", str(af),
                "--bf_matparam", str(bf),
                "--settings_dir", str(settings_dir),
                "--results_dir", str(results_dir)
            ]
            if geo_fname is not None:
                unload_args += ["--geometry_fname", geo_fname]
            _launch_srun(cpu_num, unloading_py, unload_args)

            infl_args = [
                "-i", str(sample_ID),
                "-o", output_folder,
                "--a_matparam", str(a),
                "--af_matparam", str(af),
                "--bf_matparam", str(bf),
                "-lp",
                "--settings_dir", str(settings_dir),
                "--results_dir", str(results_dir)
            ]
            _launch_srun(cpu_num, inflator_py, infl_args)

            if n > 2:
                _run_py(contour_py, [
                    "-i", str(sample_ID),
                    "-c", "30",
                    "--bf_flag",
                    "-o", results_folder,
                    "--settings_dir", str(settings_dir),
                    "--results_dir", str(results_dir)
                ])

            logger.info("unloading_inflator_done", a=a, af=af, bf=bf)
        except subprocess.CalledProcessError as e:
            logger.error("unloading_inflator_error", a=a, af=af, bf=bf, error=str(e))
            continue


def run_fiber_modeling(sample_ID, epi_fibers, endo_fibers, edpvr_folder, settings_dir, cpu_num=8):
    processing_py = DYNACOMP / "processing.py"
    validator_py = DYNACOMP / "validator.py"
    contour_py = DYNACOMP / "create_fibparam_sweep_contour.py"
    results_dir = DYNACOMP.parent / "01_results_coarse_mesh"


    for epi_fiber in epi_fibers:
        for endo_fiber in endo_fibers:
            logger.info("fiber_modeling_start", sample_ID=sample_ID, epi_fiber=epi_fiber, endo_fiber=endo_fiber)
            output_folder = f"03_Fiber_Modeling/epi_{epi_fiber}_endo_{endo_fiber}"
            output_dir = Path(f"/home/shared/01_results_coarse_mesh/OP{sample_ID}/TPM") / output_folder
            if output_dir.exists():
                logger.warning("fiber_modeling_skip_existing", path=str(output_dir))
                continue
            try:
                _launch_srun(cpu_num, processing_py, [
                    "--fiber_modeling_flag",
                    "-i", str(sample_ID),
                    "-o", output_folder,
                    "--epi_fiber", str(epi_fiber),
                    "--endo_fiber", str(endo_fiber),
                    "--edpvr_folder", edpvr_folder,
                    "--settings_dir", str(settings_dir),
                    "--results_dir", str(results_dir)
                ])
            except subprocess.CalledProcessError as e:
                logger.error("fiber_processing_error", sample_ID=sample_ID, error=str(e))

            try:
                _run_py(validator_py, [
                    "-i", str(sample_ID),
                    "-o", output_folder,
                    "-f", edpvr_folder,
                    "--epi_fiber", str(epi_fiber),
                    "--endo_fiber", str(endo_fiber),
                    "--logging_flag",
                    "--settings_dir", str(settings_dir),
                    "--results_dir", str(results_dir)
                ])
            except subprocess.CalledProcessError as e:
                logger.error("fiber_validation_error", sample_ID=sample_ID, error=str(e))

            try:
                _run_py(contour_py, [
                    "-i", str(sample_ID),
                    "-o", "03_Fiber_Modeling",
                    "--settings_dir", str(settings_dir),
                    "--results_dir", str(results_dir)
                ])
            except subprocess.CalledProcessError as e:
                logger.error("fiber_contour_error", sample_ID=sample_ID, error=str(e))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D parameter sweep of (a, a_f) for HeartModelDynaComp",
    )
    parser.add_argument("-n", "--number", nargs="*", type=int, default=None)
    parser.add_argument("-i", "--sample_ID", nargs="+", type=str)
    parser.add_argument("--settings_dir", type=Path, default=Path("settings"))
    parser.add_argument("-s", "--scan_type", type=str, default="TPM")
    parser.add_argument("-r", "--results_folder", type=str, default="02_EDPVR_Modeling_v2")
    parser.add_argument("--cpu_num", type=int, default=slurm_ntasks(8))

    args = parser.parse_args()

    settings_dir: Path = args.settings_dir
    cpu_num: int = args.cpu_num
    results_folder: str = args.results_folder

    a_af_list = np.flipud(grid_triangle_biased(N=10, amin=0.05, amax=5, afmin=0.05, afmax=5, bias_power=1.4))
    bf_list = [0.001]
    epi_fibers = [-30, -35]
    endo_fibers = [30, 35]

    if args.sample_ID:
        sample_nums = []
        for sid in args.sample_ID:
            id_num = utils.get_num_from_id(sid, settings_dir)
            sample_nums.append(id_num)
    elif args.number:
        sample_nums = args.number
    else:
        files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(files) + 1))

    for sample in sample_nums:
        settings = utils.load_settings(settings_dir, sample)
        sample_ID = settings["id"][2:] if settings["id"].startswith("OP") else settings["id"]

        for bf in bf_list:
            print("------------------------------")
            print(f"Processing sample {sample_ID}")
            print("------------------------------")
            results_folder_local = "02_EDPVR_Modeling"
            run_EDPVR(sample_ID, a_af_list, bf, results_folder_local, settings_dir, cpu_num=cpu_num)
            run_fiber_modeling(sample_ID, epi_fibers, endo_fibers, results_folder_local, settings_dir, cpu_num=cpu_num)
            fiber_angles = load_fiber_modeling(sample_ID)
            geo_fname = update_fiber(sample_ID, fiber_angles, results_folder_local)
            run_EDPVR(sample_ID, a_af_list, bf, results_folder_local, settings_dir, geo_fname=geo_fname, cpu_num=cpu_num)

if __name__ == "__main__":
    main()
