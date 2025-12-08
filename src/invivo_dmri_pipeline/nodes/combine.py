# src/invivo_dmri_pipeline/nodes/b0_and_combine.py

# Standard lib
import os
import re
import glob
import shutil
import subprocess

# Third party
import numpy as np
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, InputMultiPath, traits,
    CommandLine, CommandLineInputSpec, isdefined
)

# Package-local helpers
from ..utils import img_stem

# -------------------------
# remove initial b0 from runs
# -------------------------
class RemoveB0InputSpec(CommandLineInputSpec):
    script   = File(exists=True, position=0, argstr="%s", mandatory=True)
    indat    = File(exists=True, argstr="-indat %s",  mandatory=True)
    outdir   = Directory(mandatory=True, argstr="-outdir %s")
    b0range  = traits.Float(argstr="--b0range %s", usedefault=True, value=60.0)
    outname  = traits.Str(argstr="--outname %s")
    deps     = InputMultiPath(File(), desc="dummy deps to enforce scheduling")


class RemoveB0OutputSpec(TraitedSpec):
    out_file = File()


class RemoveB0(CommandLine):
    _cmd        = "python3"
    input_spec  = RemoveB0InputSpec
    output_spec = RemoveB0OutputSpec

    def _expected_out(self):
        if isdefined(self.inputs.outname) and self.inputs.outname:
            base = self.inputs.outname
        else:
            base = os.path.basename(img_stem(self.inputs.indat))
        return os.path.abspath(os.path.join(self.inputs.outdir, f"{base}.nii.gz"))

    def _list_outputs(self):
        return {"out_file": self._expected_out()}

    def _run_interface(self, runtime):
        if os.path.exists(self._expected_out()):
            runtime.returncode = 0
            return runtime
        return super()._run_interface(runtime)


# -------------------------
# combine in vivo runs
# -------------------------
class CombineInputSpec(CommandLineInputSpec):
    script          = File(exists=True, position=0, argstr="%s", mandatory=True)
    indat           = traits.List(File(exists=True), argstr="-indat %s", mandatory=True, sep=" ")
    outprefix       = traits.Str(argstr="-outprefix %s", mandatory=True)
    b0range         = traits.Float(usedefault=True, value=60.0)
    bvals_round_py  = File(exists=True)
    deps            = InputMultiPath(File(), desc="dummy deps to enforce scheduling")


class CombineOutputSpec(TraitedSpec):
    out_file = File()
    out_bval = File()
    out_bvec = File()


class Combine(CommandLine):
    _cmd        = "python3"
    input_spec  = CombineInputSpec
    output_spec = CombineOutputSpec

    def _pref(self):
        return os.path.abspath(self.inputs.outprefix)

    def _list_outputs(self):
        pref = self._pref()
        return {
            "out_file": pref + ".nii.gz",
            "out_bval": pref + ".bval",
            "out_bvec": pref + ".bvec",
        }

    def _round_bvals_inplace(self, bval_path: str):
        """Round shells and force b<=B0RANGE → 0, in-place.

        Preferred: use helper script (bvals_round.py).
        Fallback: do a simple in-Python rounding if the script isn't provided.
        """
        if not os.path.exists(bval_path):
            return

        b0thr = int(round(float(self.inputs.b0range)))
        vals  = np.loadtxt(bval_path).reshape(-1)

        if isdefined(self.inputs.bvals_round_py) and self.inputs.bvals_round_py:
            # Derive candidate shells from current values (nearest 100) + ensure 0 present
            def _cand_shells(x):
                x = np.asarray(x).reshape(-1)
                return np.unique((np.rint(x / 100.0) * 100).astype(int))

            cand = _cand_shells(vals)
            if 0 not in cand:
                cand = np.unique(np.concatenate([cand, np.array([0], dtype=int)]))
            blist = ",".join(str(x) for x in cand.tolist())

            subprocess.run([
                "python3", os.path.abspath(self.inputs.bvals_round_py),
                "-in",  bval_path,
                "-out", bval_path,
                "-blist", blist,
                "-tol", str(b0thr),
            ], check=True)
        else:
            # Simple fallback: <=B0RANGE → 0, otherwise round to nearest 100
            vals_rounded = np.where(vals <= b0thr, 0, (np.rint(vals / 100.0) * 100.0))
            np.savetxt(bval_path, vals_rounded[np.newaxis, :], fmt="%.0f")

    def _run_interface(self, runtime):
        indats = list(self.inputs.indat or [])
        pref   = self._pref()
        out_nii  = pref + ".nii.gz"
        out_bval = pref + ".bval"
        out_bvec = pref + ".bvec"

        os.makedirs(os.path.dirname(pref), exist_ok=True)

        # 1) Ensure combined outputs exist (copy-through for single input)
        if len(indats) == 0:
            raise RuntimeError("Combine: no inputs were provided.")

        if len(indats) == 1:
            # copy single run
            src  = os.path.abspath(indats[0])
            stem = os.path.basename(img_stem(src))
            src_bval = os.path.join(os.path.dirname(src), stem + ".bval")
            src_bvec = os.path.join(os.path.dirname(src), stem + ".bvec")

            if not os.path.exists(out_nii):
                shutil.copy2(src, out_nii)
            if os.path.exists(src_bval):
                shutil.copy2(src_bval, out_bval)
            if os.path.exists(src_bvec):
                shutil.copy2(src_bvec, out_bvec)
        else:
            # run external combine script (concatenate NIfTI + cat bvals/bvecs)
            if not (os.path.exists(out_nii) and os.path.exists(out_bval) and os.path.exists(out_bvec)):
                runtime = super()._run_interface(runtime)

        # 2) Always (re-)round the combined bvals in place
        if os.path.exists(out_bval):
            self._round_bvals_inplace(out_bval)

        runtime.returncode = 0
        return runtime