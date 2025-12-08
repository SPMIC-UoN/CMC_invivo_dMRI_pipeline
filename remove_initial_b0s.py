#!/usr/bin/env python3
"""
Remove the first volume (initial b0) from all runs in AP/PA/LR/RL (+ *_ph)
directories under <input_root>.

Logic:

  • If <TAG>_orig already exists (e.g., AP_orig/), then that TAG is skipped.
  • Otherwise:
        - Move <TAG>/ → <TAG>_orig/
        - Recreate empty <TAG>/ directory
        - Process each NIfTI inside <TAG>_orig/ and write processed output
          with the ORIGINAL base name into <TAG>/

Results example:

    AP_orig/AP_1.nii.gz
    AP_orig/AP_1.bval
    AP_orig/AP_1.bvec

    AP/AP_1.nii.gz      (b0 removed)
    AP/AP_1.bval
    AP/AP_1.bvec
"""

import argparse
import os
import sys
import subprocess
import re
import glob
import numpy as np
import shutil


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s\n\n" % message)
        self.print_help()
        sys.exit(2)


def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def stem_only(path: str) -> str:
    b = os.path.basename(path)
    if b.endswith(".nii.gz"):
        return b[:-7]
    if b.endswith(".nii"):
        return b[:-4]
    return b


def get_ext(path: str) -> str:
    if path.endswith(".nii.gz"):
        return ".nii.gz"
    if path.endswith(".nii"):
        return ".nii"
    return ""


def sort_key_numeric(path: str):
    s = stem_only(path)
    toks = [t for t in s.split("_") if t.isdigit()]
    if toks:
        return (0, int(toks[0]), s)
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, s.lower())


def remove_initial_b0(nii_file: str, outdir: str, b0range: float):
    """Create b0-removed version of NIfTI+sidecars."""
    base = strip_ext(nii_file)
    ext = get_ext(nii_file)

    # sidecar paths from the original (orig) directory
    bval_orig = nii_file.replace(ext, ".bval")
    bvec_orig = nii_file.replace(ext, ".bvec")

    if not os.path.exists(bval_orig) or not os.path.exists(bvec_orig):
        raise FileNotFoundError(f"Missing .bval or .bvec for {nii_file}")

    print(f"  Processing: {os.path.basename(nii_file)}")

    # load bvals from original
    bvals = np.loadtxt(bval_orig)
    if bvals.ndim > 1:
        bvals = bvals.flatten()

    if bvals.size < 2:
        sys.exit(f"Error: <2 bvals for {nii_file}. Cannot remove first volume.")

    if not (bvals[0] <= b0range and bvals[1] <= b0range):
        sys.exit(
            f"Error: First two bvals of {nii_file} must be <= {b0range}. "
            f"Found {bvals[0]}, {bvals[1]}"
        )

    out_nii = os.path.join(outdir, strip_ext(nii_file) + ext)
    out_bval = out_nii.replace(ext, ".bval")
    out_bvec = out_nii.replace(ext, ".bvec")

    # remove first volume using fslroi
    print("    fslroi (removing first volume)")
    subprocess.run(["fslroi", nii_file, out_nii, "1", "-1"], check=True)

    # update bvals/bvecs
    print("    updating bval/bvec")

    bvecs = np.loadtxt(bvec_orig)
    if bvecs.ndim == 1:
        bvecs = bvecs.reshape(1, -1)

    bvals_new = bvals[1:]
    bvecs_new = bvecs[:, 1:]

    fmt_bval = "%d" if np.all(np.mod(bvals_new, 1) == 0) else "%.8f"
    if fmt_bval == "%d":
        bvals_new = bvals_new.astype(int)

    np.savetxt(out_bval, bvals_new[np.newaxis], fmt=fmt_bval)
    np.savetxt(out_bvec, bvecs_new, fmt="%.8f")

    print(f"    wrote: {os.path.basename(out_nii)}, bval, bvec")


def main():
    parser = MyParser(prog="remove_initial_b0")
    required = parser.add_argument_group("Required arguments")
    required.add_argument("input_root", help="Root directory containing AP/, PA/, LR/, RL/ (+ *_ph)")

    optional = parser.add_argument_group("Optional arguments")
    optional.add_argument("--b0range", type=float, default=60.0,
                          help="Max b-value to consider b0 (default=60)")

    args = parser.parse_args()
    root = os.path.abspath(args.input_root)
    b0range = float(args.b0range)

    print(f"\n=== remove_initial_b0 (directory-based) ===")
    print(f"Input root : {root}")
    print(f"b0 range   : {b0range}\n")

    dirs = ["AP", "PA", "LR", "RL", "AP_ph", "PA_ph", "LR_ph", "RL_ph"]

    for d in dirs:
        in_dir = os.path.join(root, d)
        orig_dir = os.path.join(root, f"{d}_orig")

        if not os.path.isdir(in_dir):
            continue

        print(f"\n--- {d}/ ---")

        if os.path.isdir(orig_dir):
            print(f"  SKIP: {orig_dir} already exists. {d} assumed previously processed.")
            continue

        # Move directory to *_orig
        print(f"  Renaming {d}/ → {d}_orig/")
        shutil.move(in_dir, orig_dir)

        # Recreate empty <d>/ directory
        os.makedirs(in_dir, exist_ok=True)

        # Find NIfTIs in *_orig
        if d.endswith("_ph"):
            prefix = d.replace("_ph", "")
            pattern = os.path.join(orig_dir, f"{prefix}_*_ph.nii*")
        else:
            pattern = os.path.join(orig_dir, f"{d}_*.nii*")

        nii_files = sorted(glob.glob(pattern), key=sort_key_numeric)

        if not nii_files:
            print("  No NIfTI files found.")
            continue

        # Process each file
        for nii in nii_files:
            remove_initial_b0(nii, in_dir, b0range)

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
