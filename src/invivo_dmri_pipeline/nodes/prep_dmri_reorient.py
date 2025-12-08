#!/usr/bin/env python3
"""
prep_dmri_reorient.py

Reorient dMRI NIfTI volumes with `fslswapdim`, update BVECs,
and optionally fix sform/qform headers.

Includes a hard header fix option that diagonalises voxel scales
and negates translations (to mimic manual fslorient -set* fixes).
"""

import argparse
import os
import sys
import subprocess
import numpy as np
import nibabel as nib


# Map axis tokens to indices
AX2IDX = {'x': 0, 'y': 1, 'z': 2}


# ------------------------
# Utility functions
# ------------------------

def check_tools():
    """Check required FSL tools are on PATH."""
    for t in ['fslswapdim']:
        if subprocess.call(['which', t],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) != 0:
            sys.stderr.write(f"ERROR: {t} not found (is FSL on PATH?)\n")
            sys.exit(2)


def make_tokens(v_tokens, r_signs):
    """Combine axis tokens and signs into fslswapdim args."""
    if len(v_tokens) != 3 or len(r_signs) != 3:
        raise ValueError("'-v' and '-r' must have 3 entries each")

    out = []
    for ax, sgn in zip(v_tokens, r_signs):
        base = ax.lower().strip()
        if base not in AX2IDX:
            raise ValueError(f"Invalid axis token: {ax} (must be x,y,z)")
        if sgn not in [-1, 1]:
            raise ValueError(f"Invalid sign {sgn}; must be 1 or -1")
        out.append(('-' if sgn == -1 else '') + base)
    return out


def make_P(tokens):
    """
    Construct permutation matrix P for affine transforms:
    v_old = P @ v_new
    """
    P = np.zeros((4, 4), dtype=float)
    P[3, 3] = 1.0
    used = set()
    for new_axis, t in enumerate(tokens):
        sign = -1.0 if t.startswith('-') else 1.0
        base = t.lstrip('+-')
        old_axis = AX2IDX[base]
        if old_axis in used:
            raise ValueError("Each of x,y,z must appear exactly once.")
        used.add(old_axis)
        P[old_axis, new_axis] = sign
    return P


def make_B(tokens):
    """
    Construct permutation/flip matrix for bvecs:
    bvec_new = B @ bvec_old
    """
    B = np.zeros((3, 3), dtype=float)
    for new_axis, t in enumerate(tokens):
        sign = -1.0 if t.startswith('-') else 1.0
        base = t.lstrip('+-')
        old_axis = AX2IDX[base]
        B[new_axis, old_axis] = sign
    return B


def read_bvec(path):
    """Read a .bvec file into 3xN numpy array."""
    M = np.loadtxt(path)
    if M.ndim == 1:
        M = M.reshape(1, -1)
    if M.shape[0] == 3:
        return M
    if M.shape[1] == 3:
        return M.T
    raise ValueError(f"Unexpected bvec shape {M.shape} for {path}")


def write_bvec(path, M):
    """Write 3xN bvec array back to disk."""
    np.savetxt(path, M, fmt="%.10f")


def fix_sforms(out_nii, in_sform, in_scode, in_qform, in_qcode, P):
    """Update sform/qform of a NIfTI to match new data orientation."""
    img = nib.load(out_nii)
    Snew = in_sform @ P if in_sform is not None else img.affine
    Qnew = in_qform @ P if in_qform is not None else img.affine
    img.set_sform(Snew, int(in_scode) if in_scode else 1)
    img.set_qform(Qnew, int(in_qcode) if in_qcode else 1)
    nib.save(img, out_nii)


def hard_header_fix(path):
    """
    Apply hard fix:
    - Force voxel scales onto diagonal
    - Negate translations
    - Set both sform and qform
    """
    img = nib.load(path)

    # Extract voxel sizes from column norms
    S = img.get_sform()
    if S is None:
        S = img.affine
    vx = np.linalg.norm(S[0:3, 0])
    vy = np.linalg.norm(S[0:3, 1])
    vz = np.linalg.norm(S[0:3, 2])
    T = S[0:3, 3]

    # Construct fixed matrix
    Sfix = np.zeros((4, 4), dtype=float)
    Sfix[0, 0] = abs(vx)
    Sfix[1, 1] = abs(vy)
    Sfix[2, 2] = abs(vz)
    Sfix[3, 3] = 1.0
    Sfix[0:3, 3] = -T

    img.set_sform(Sfix, 1)
    img.set_qform(Sfix, 1)
    nib.save(img, path)


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Swap/flip NIfTI with fslswapdim, update BVECs, fix s/qforms; "
                    "optionally apply a hard header fix."
    )
    ap.add_argument("-i", required=True, help="Input NIfTI (.nii.gz)")
    ap.add_argument("-o", required=True, help="Output NIfTI (.nii.gz)")
    ap.add_argument("-v", metavar="AXIS", nargs=3, required=False,
                    help="Axes for fslswapdim (e.g., x y z)")
    ap.add_argument("-r", metavar="SIGN", nargs=3, required=False, type=int,
                    help="Signs for axes (1 or -1), e.g., 1 1 -1")

    ap.add_argument("--header-only", action="store_true",
                    help="Do not run fslswapdim; just copy input to output")
    ap.add_argument("--hard-fix-header", action="store_true",
                    help="Force s/qform to diagonal voxel scales and negated translations")

    args = ap.parse_args()

    in_nii = os.path.abspath(args.i)
    out_nii = os.path.abspath(args.o)
    os.makedirs(os.path.dirname(out_nii), exist_ok=True)

    if args.header_only:
        # Just copy NIfTI and sidecars
        subprocess.run(['cp', '-f', in_nii, out_nii], check=True)

        stem_in = in_nii[:-7] if in_nii.endswith('.nii.gz') else os.path.splitext(in_nii)[0]
        stem_out = out_nii[:-7] if out_nii.endswith('.nii.gz') else os.path.splitext(out_nii)[0]

        if os.path.isfile(stem_in + '.bval'):
            subprocess.run(['cp', '-f', stem_in + '.bval', stem_out + '.bval'], check=True)
        if os.path.isfile(stem_in + '.bvec'):
            subprocess.run(['cp', '-f', stem_in + '.bvec', stem_out + '.bvec'], check=True)

    else:
        if args.v is None or args.r is None:
            raise SystemExit("When not using --header-only, you must provide -v and -r")

        check_tools()
        tokens = make_tokens(args.v, args.r)
        P = make_P(tokens)
        B = make_B(tokens)

        # Save original header info
        src_img = nib.load(in_nii)
        sform, scode = src_img.get_sform(coded=True)
        qform, qcode = src_img.get_qform(coded=True)
        if sform is None:
            sform, scode = src_img.affine, 1
        if qform is None:
            qform, qcode = src_img.affine, 1

        # 1) Reorient the data
        cmd = ['fslswapdim', in_nii] + tokens + [out_nii]
        subprocess.run(cmd, check=True)

        # 2) Handle sidecars
        stem_in = in_nii[:-7] if in_nii.endswith('.nii.gz') else os.path.splitext(in_nii)[0]
        stem_out = out_nii[:-7] if out_nii.endswith('.nii.gz') else os.path.splitext(out_nii)[0]

        if os.path.isfile(stem_in + '.bval'):
            subprocess.run(['cp', '-f', stem_in + '.bval', stem_out + '.bval'], check=True)

        if os.path.isfile(stem_in + '.bvec'):
            BV = read_bvec(stem_in + '.bvec')
            BV_new = B @ BV
            write_bvec(stem_out + '.bvec', BV_new)

        # 3) Update header to match new orientation
        P44 = np.eye(4)
        P44[:3, :3] = P[:3, :3]
        fix_sforms(out_nii, sform, scode, qform, qcode, P44)

    if args.hard_fix_header:
        hard_header_fix(out_nii)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
