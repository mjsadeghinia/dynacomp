#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
import utils  # your existing helper module

"""
Upload/download results to Google Drive using rclone.
Intended to be run locally (not on Docker/ex3).

Prereqs:
  - rclone installed
  - rclone remote configured (default name: gdrive)
"""

def shlex_join(parts):
    # basic, avoids importing shlex for readability in prints
    return " ".join(str(p) for p in parts)

def require_rclone():
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone not found. Install it (e.g., 'brew install rclone') "
            "and run 'rclone config' to create a Google Drive remote."
        )

def run(cmd):
    print("Running:", shlex_join(cmd))
    subprocess.run(cmd, check=True)

def rclone_copy(src, dst, use_sync=False, dry_run=False, extra=None):
    """
    src, dst: paths (src can be local or remote, dst can be local or remote)
    use_sync=False -> rclone copy (no deletions)
    use_sync=True  -> rclone sync (mirror; may delete at destination)
    """
    base = ["rclone", "sync" if use_sync else "copy", str(src), str(dst), "-P"]
    if extra:
        base.extend(extra)
    if dry_run:
        base.append("--dry-run")
    run(base)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--number",
        nargs="*", type=int, default=None,
        help="Sample number(s) to process. If omitted, all settings files are used.",
    )
    parser.add_argument(
        "-i", "--ID",
        nargs="+", type=str,
        help="Sample ID(s). If provided, --number is ignored.",
    )
    parser.add_argument(
        "--settings_dir", type=Path,
        default=Path("/Users/javad/Docker/dynacomp/dynacomp/settings"),
        help="Directory of JSON settings files.",
    )
    parser.add_argument(
        "-s", "--scan_type", type=str, default="TPM",
        help="Scan type; subdirectories are named accordingly.",
    )
    parser.add_argument(
        "--local_results_dir", type=Path,
        default=Path("/Users/javad/Docker/dynacomp/01_results_coarse_mesh"),
        help="Local results root directory.",
    )

    # --- Google Drive specific options ---
    parser.add_argument(
        "--drive_remote_name", type=str, default="gdrive",
        help="rclone remote name for Google Drive (as set in `rclone config`).",
    )
    parser.add_argument(
        "--drive_root_dir", type=str, default="03_Dynacomp/01_results_coarse_mesh",
        help="Root folder in Drive under the remote (created if needed).",
    )

    parser.add_argument(
        "-f", "--folders",
        nargs="+", type=str, required=True,
        help="Extra folders to import/export (e.g., 01_PVCalibration 02_Unloading ...).",
    )

    parser.add_argument(
        "--use_sync", action="store_true",
        help="Use `rclone sync` instead of `rclone copy` (mirrors deletions).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Append `--dry-run` to rclone to preview actions.",
    )

    args = parser.parse_args()
    require_rclone()

    settings_dir         = args.settings_dir
    scan_type            = args.scan_type
    local_results_dir    = Path(args.local_results_dir)
    drive_remote_name    = args.drive_remote_name.rstrip(":")
    drive_root_dir       = args.drive_root_dir.strip("/")

    # Determine sample numbers from IDs or file count
    if args.ID is not None:
        sample_nums = [utils.get_num_from_id(_id, settings_dir) for _id in args.ID]
    elif args.number:
        sample_nums = args.number
    else:
        settings_files = sorted([f for f in settings_dir.iterdir() if f.suffix == ".json"])
        sample_nums = list(range(1, len(settings_files) + 1))

    # Remote root in rclone syntax: "<remote>:<path>"
    # Example: "gdrive:01_results_coarse_mesh"
    DRIVE_ROOT = f"{drive_remote_name}:{drive_root_dir}"

    for sample_num in sample_nums:
        settings = utils.load_settings(settings_dir, sample_num)
        sample_name = settings["id"]
        sample_dir = local_results_dir / sample_name / scan_type

        # Destination base in Drive:
        # gdrive:01_results_coarse_mesh/<sample>/<scan_type>/
        drive_sample_base = f"{DRIVE_ROOT}/{sample_name}/{scan_type}"
        for folder in args.folders:
            folder_path = sample_dir / folder
            if not folder_path.exists():
                print(f"Skipping missing folder: {folder_path}")
                continue

            rclone_copy(
                src=str(folder_path),
                dst=f"{drive_sample_base}/{folder}",
                use_sync=args.use_sync,
                dry_run=args.dry_run,
                extra=["--no-traverse","--fast-list","--transfers=16","--checkers=32"]
            )

if __name__ == "__main__":
    main()
