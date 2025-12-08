# src/invivo_dmri_pipeline/nodes/skullstrip.py

import os
import shutil
import subprocess
from typing import Tuple

import numpy as np
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, traits, isdefined
)

from ..utils import FSL


class SkullstripInputSpec(BaseInterfaceInputSpec):
    data_4d   = File(exists=True, mandatory=True, desc="4D DWI after eddy")
    bvals     = File(exists=True, mandatory=True, desc="bvals for data_4d")
    outdir    = Directory(mandatory=True, desc="data/ dir where final mask must be copied")
    lower_b   = traits.Int(mandatory=True, desc="shell b-value to average for masking (e.g., 100)")
    b0max     = traits.Int(usedefault=True, value=60, desc="tolerance around 0 (unused here)")
    t1        = File(exists=True, desc="(optional) T1 used to make mask via registration")
    brain_mask = File(exists=True, desc="(optional) precomputed brain mask, either in dMRI or T1 space")
    bet4animal_bin = traits.Str(usedefault=True, value="bet4animal")
    bet4animal_z   = traits.Int(2, usedefault=True, desc="-z flag for bet4animal (2=nhp, 0=hum)")


class SkullstripOutputSpec(TraitedSpec):
    mask_file   = File(desc="final nodif_brain_mask.nii.gz (in data/)")
    mean_bshell = File(desc="mean image of selected shell (in skullstripping/)")


class Skullstrip(BaseInterface):
    """
    If brain_mask is provided:
      - If already in dMRI space: binarise & copy to data/.
      - Else if it matches T1 space (and T1 provided): warp to dMRI.
      - Else: error.

    Else if T1 provided: skullstrip T1 with bet4animal, then warp mask to dMRI.

    Else: skullstrip the mean of LOWER_B shell with bet4animal.

    bet4animal is always used, with -z = bet4animal_z (0 HUM, 2 NHP).
    """

    input_spec  = SkullstripInputSpec
    output_spec = SkullstripOutputSpec

    # ---------- helpers ----------
    def _fsl(self, exe: str) -> str:
        return os.path.join(FSL, "bin", exe)

    def _run_cmd(self, args: list[str], check: bool = True) -> None:
        subprocess.run(args, check=check)

    def _fslval(self, img: str, key: str) -> str:
        out = subprocess.check_output([self._fsl("fslval"), img, key], text=True).strip()
        return out.split()[0]

    def _dims_pix(self, img: str) -> Tuple[int, int, int, float, float, float]:
        d1 = int(self._fslval(img, "dim1"))
        d2 = int(self._fslval(img, "dim2"))
        d3 = int(self._fslval(img, "dim3"))
        p1 = float(self._fslval(img, "pixdim1"))
        p2 = float(self._fslval(img, "pixdim2"))
        p3 = float(self._fslval(img, "pixdim3"))
        return d1, d2, d3, p1, p2, p3

    def _same_space(self, img_a: str, img_b: str, tol: float = 1e-4) -> bool:
        a = self._dims_pix(img_a); b = self._dims_pix(img_b)
        same_dims = a[:3] == b[:3]
        same_pix  = all(abs(a[i] - b[i]) <= tol for i in range(3, 6))
        return bool(same_dims and same_pix)

    def _prepare_trunc_for_select(self, data_4d: str, bvals: str, out_prefix: str, n_keep: int):
        trunc_4d = out_prefix + "_trunc.nii.gz"
        trunc_bv = out_prefix + "_trunc.bvals"
        # Truncate images
        self._run_cmd([self._fsl("fslroi"), data_4d, trunc_4d, "0", str(n_keep)])
        # Truncate bvals in lockstep
        B = np.loadtxt(bvals).reshape(-1)
        np.savetxt(trunc_bv, B[:n_keep][None, :].astype(int), fmt="%d")
        return trunc_4d, trunc_bv

    def _select_mean_shell(self, data_4d: str, bvals: str, skull_dir: str, shell: int) -> str:
        """
        Use select_dwi_vols properly:
          select_dwi_vols <data_4D> <bvals> <output_prefix> <approx_bval> -m
        The mean will be written to <output_prefix>.nii.gz
        """
        if shutil.which("select_dwi_vols") is None:
            raise RuntimeError("select_dwi_vols not found on PATH")

        out_prefix = os.path.join(skull_dir, f"b{shell}_mean")
        out_mean   = out_prefix + ".nii.gz"

        nvol = int(self._fslval(data_4d, "dim4"))
        if nvol > 400:
            n_keep = min(380, nvol)
            trunc_4d, trunc_bvals = self._prepare_trunc_for_select(data_4d, bvals, out_prefix, n_keep)
            self._run_cmd(["select_dwi_vols", trunc_4d, trunc_bvals, out_prefix, str(shell), "-m"])
        else:
            self._run_cmd(["select_dwi_vols", data_4d, bvals, out_prefix, str(shell), "-m"])

        if not os.path.isfile(out_mean) or os.path.getsize(out_mean) == 0:
            raise RuntimeError(
                f"select_dwi_vols did not produce {out_mean}. "
                f"Ensure LOWER_B={shell} exists exactly in your bvals."
            )
        return out_mean

    # ---------- run ----------
    def _run_interface(self, runtime):
        data_dir  = os.path.abspath(self.inputs.outdir)
        root_dir  = os.path.dirname(data_dir)
        skull_dir = os.path.join(root_dir, "skullstripping")
        os.makedirs(skull_dir, exist_ok=True)

        data_4d  = os.path.abspath(self.inputs.data_4d)
        bvals    = os.path.abspath(self.inputs.bvals)
        shell    = int(self.inputs.lower_b)
        bet4     = self.inputs.bet4animal_bin or "bet4animal"
        zflag    = str(int(self.inputs.bet4animal_z))

        # Build mean image strictly from the LOWER_B shell
        mean_img = self._select_mean_shell(data_4d, bvals, skull_dir, shell)

        mask_out_in_data = os.path.join(data_dir, "nodif_brain_mask.nii.gz")

        # Shortcuts to FSL binaries
        flirt    = self._fsl("flirt")
        cx       = self._fsl("convert_xfm")
        fslmaths = self._fsl("fslmaths")

        # If user provided a mask
        user_mask = self.inputs.brain_mask if isdefined(self.inputs.brain_mask) else ""
        has_user_mask = bool(user_mask)
        t1_src = os.path.abspath(self.inputs.t1) if (isdefined(self.inputs.t1) and self.inputs.t1) else ""

        if has_user_mask:
            mask_src = os.path.abspath(user_mask)

            # Case A: mask already in dMRI space
            if self._same_space(mask_src, data_4d):
                tmp_bin = os.path.join(skull_dir, "user_mask_dmri_bin.nii.gz")
                self._run_cmd([fslmaths, mask_src, "-bin", tmp_bin])
                shutil.copy2(tmp_bin, mask_out_in_data)
                self._mask = mask_out_in_data
                self._mean = mean_img
                return runtime

            # Case B: mask matches T1 space and T1 provided
            if t1_src and self._same_space(mask_src, t1_src):
                t1_copy   = os.path.join(skull_dir, "struct.nii.gz")
                t1_brain  = os.path.join(skull_dir, "struct_brain.nii.gz")
                tmp_bin   = os.path.join(skull_dir, "user_mask_t1_bin.nii.gz")
                if not os.path.isfile(t1_copy):
                    shutil.copy2(t1_src, t1_copy)

                self._run_cmd([fslmaths, mask_src, "-bin", tmp_bin])
                self._run_cmd([fslmaths, t1_copy, "-mul", tmp_bin, t1_brain])

                d2s_mat = os.path.join(skull_dir, "diff_to_struct.mat")
                d2s_img = os.path.join(skull_dir, "diff_to_struct.nii.gz")
                self._run_cmd([flirt, "-in", mean_img, "-ref", t1_brain, "-omat", d2s_mat, "-out", d2s_img])

                s2d_mat = os.path.join(skull_dir, "struct_to_diff.mat")
                self._run_cmd([cx, "-inverse", "-omat", s2d_mat, d2s_mat])

                mask_in_diff = os.path.join(skull_dir, "nodif_brain_mask.nii.gz")
                self._run_cmd([
                    flirt,
                    "-in", tmp_bin,
                    "-ref", mean_img,
                    "-applyxfm", "-init", s2d_mat,
                    "-out", mask_in_diff,
                    "-interp", "nearestneighbour"
                ])
                shutil.copy2(mask_in_diff, mask_out_in_data)
                self._mask = mask_out_in_data
                self._mean = mean_img
                return runtime

            # Case C: T1 not provided — expect mask in dMRI space; if not, error
            if not t1_src:
                if not self._same_space(mask_src, data_4d):
                    dm = self._dims_pix(data_4d)
                    mm = self._dims_pix(mask_src)
                    raise RuntimeError(
                        "Provided brain_mask does not match dMRI space and no T1 was provided.\n"
                        f"dMRI dims/pix: {dm[:3]}/{dm[3:]}, mask dims/pix: {mm[:3]}/{mm[3:]}"
                    )
                tmp_bin = os.path.join(skull_dir, "user_mask_dmri_bin.nii.gz")
                self._run_cmd([fslmaths, mask_src, "-bin", tmp_bin])
                shutil.copy2(tmp_bin, mask_out_in_data)
                self._mask = mask_out_in_data
                self._mean = mean_img
                return runtime

            # Otherwise: mask did not match either space
            dm  = self._dims_pix(data_4d)
            mm  = self._dims_pix(mask_src)
            t1m = self._dims_pix(t1_src) if t1_src else None
            raise RuntimeError(
                "Provided brain_mask does not match dMRI or T1 dimensions/voxel sizes.\n"
                f"dMRI dims/pix: {dm[:3]}/{dm[3:]}\n"
                f"mask dims/pix: {mm[:3]}/{mm[3:]}\n"
                + (f"T1   dims/pix: {t1m[:3]}/{t1m[3:]}\n" if t1m else "")
            )

        # No user mask — create mask
        if isdefined(self.inputs.t1) and self.inputs.t1:
            if shutil.which(bet4) is None:
                raise RuntimeError("bet4animal not found on PATH")

            t1_copy = os.path.join(skull_dir, "struct.nii.gz")
            shutil.copy2(t1_src, t1_copy)

            t1_brain     = os.path.join(skull_dir, "struct_brain.nii.gz")
            t1_brainmask = os.path.join(skull_dir, "struct_brain_mask.nii.gz")
            self._run_cmd([bet4, t1_copy, t1_brain, "-z", zflag, "-m", "-R", "-f", "0.3"])

            if not os.path.isfile(t1_brainmask):
                raise RuntimeError("bet4animal did not produce struct_brain_mask.nii.gz")

            d2s_mat = os.path.join(skull_dir, "diff_to_struct.mat")
            d2s_img = os.path.join(skull_dir, "diff_to_struct.nii.gz")
            self._run_cmd([flirt, "-in", mean_img, "-ref", t1_brain, "-omat", d2s_mat, "-out", d2s_img])

            s2d_mat = os.path.join(skull_dir, "struct_to_diff.mat")
            self._run_cmd([self._fsl("convert_xfm"), "-inverse", "-omat", s2d_mat, d2s_mat])

            mask_in_diff = os.path.join(skull_dir, "nodif_brain_mask.nii.gz")
            self._run_cmd([
                flirt,
                "-in", t1_brainmask,
                "-ref", mean_img,
                "-applyxfm", "-init", s2d_mat,
                "-out", mask_in_diff,
                "-interp", "nearestneighbour"
            ])
            if not os.path.isfile(mask_in_diff) or os.path.getsize(mask_in_diff) == 0:
                raise RuntimeError("Failed to generate diffusion-space mask from T1.")

            shutil.copy2(mask_in_diff, mask_out_in_data)

        else:
            # No T1 → skullstrip LOWER_B mean directly
            if shutil.which(bet4) is None:
                raise RuntimeError("bet4animal not found on PATH")

            nodif_brain  = os.path.join(skull_dir, "nodif_brain.nii.gz")
            mask_in_diff = os.path.join(skull_dir, "nodif_brain_mask.nii.gz")
            self._run_cmd([bet4, mean_img, nodif_brain, "-z", zflag, "-m", "-R", "-B", "-f", "0.3"])

            if not os.path.isfile(mask_in_diff):
                raise RuntimeError("bet4animal did not produce nodif_brain_mask.nii.gz")

            shutil.copy2(mask_in_diff, mask_out_in_data)

        self._mask = mask_out_in_data
        self._mean = mean_img
        return runtime

    def _list_outputs(self):
        return {"mask_file": getattr(self, "_mask", ""),
                "mean_bshell": getattr(self, "_mean", "")}
