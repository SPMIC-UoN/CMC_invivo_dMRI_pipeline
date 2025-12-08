#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import re
import numpy as np


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help()
        sys.exit(2)


def strip_ext(p: str) -> str:
    """Return basename without .nii/.nii.gz."""
    b = os.path.basename(p)
    if b.endswith(".nii.gz"):
        return b[:-7]
    if b.endswith(".nii"):
        return b[:-4]
    return b


def sidecar(nii_path: str, ext: str) -> str:
    """
    Resolve the sidecar path next to the NIfTI, not the CWD.
    Works even if input is absolute.
    """
    d = os.path.dirname(nii_path)
    return os.path.join(d, strip_ext(nii_path) + ext)


def num_key(p: str):
    """Sort by the first numeric token in the *stem*; fallback to lexicographic."""
    s = strip_ext(p)
    toks = [t for t in s.split("_") if t.isdigit()]
    if toks:
        return (0, int(toks[0]), s.lower())
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s.lower())
    return (1, s.lower())


parser = MyParser(prog="combine_series")
req = parser.add_argument_group("Required arguments")
req.add_argument(
    "-indat",
    metavar="<list>",
    nargs="+",
    required=True,
    help="Space-separated list of input .nii.gz files",
)
req.add_argument(
    "-outprefix",
    metavar="<str>",
    required=True,
    help="Output prefix for merged NIfTI and bval/bvec files",
)
args = parser.parse_args()

print("\n--- Combine dMRI runs --- ")
nii_files = [os.path.abspath(f) for f in args.indat]
nii_files.sort(key=num_key)

outprefix = os.path.abspath(args.outprefix)
outdir = os.path.dirname(outprefix) or "."
os.makedirs(outdir, exist_ok=True)

# ---------- bvals/bvecs ----------
print("Combining bvals and bvecs...")

bval_list = []
bvec_list = []

for nii in nii_files:
    bval = sidecar(nii, ".bval")
    bvec = sidecar(nii, ".bvec")

    if not (os.path.exists(bval) and os.path.exists(bvec)):
        raise FileNotFoundError(
            f"Missing bval or bvec for {nii}\n"
            f"  expected: {bval}\n"
            f"            {bvec}"
        )

    bv = np.loadtxt(bval)
    bx = np.loadtxt(bvec)

    if bv.ndim > 1:
        bv = bv.flatten()

    if bx.ndim == 1:
        bx = bx.reshape(1, -1)

    if bx.shape[0] != 3 and bx.shape[1] == 3:
        bx = bx.T

    if bx.shape[0] != 3:
        raise RuntimeError(f"bvecs for {nii} are not 3xN or Nx3 (got {bx.shape})")

    bval_list.append(bv)
    bvec_list.append(bx)

combined_bval = np.concatenate(bval_list)
combined_bvec = np.hstack(bvec_list)

out_bval = outprefix + ".bval"
out_bvec = outprefix + ".bvec"
out_nii = outprefix + ".nii.gz"

fmt_bval = "%d" if np.all(np.equal(np.mod(combined_bval, 1), 0)) else "%.8f"
if fmt_bval == "%d":
    combined_bval = combined_bval.astype(int)

np.savetxt(out_bval, combined_bval[np.newaxis], fmt=fmt_bval)
np.savetxt(out_bvec, combined_bvec, fmt="%.8f")

# ---------- images ----------
print("Combining NIfTI files...")
subprocess.run(["fslmerge", "-t", out_nii] + nii_files, check=True)

print(f"Saved: {outprefix}.nii.gz, .bval, .bvec")
print("\n--- Done! --- ")
