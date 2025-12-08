# src/invivo_dmri_pipeline/nodes/denoise.py

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
    CommandLine, CommandLineInputSpec
)

# Package-local helpers
from ..utils import img_stem, FSL, fslval

# -------------------------
# Helpers
# -------------------------
def _num_suffix(path: str, prefix: str) -> int:
    """
    Extract numeric suffix from files like:
      TAG_3.nii.gz      (prefix='TAG')
      TAG_ph_3.nii.gz   (prefix='TAG_ph')
    """
    base = os.path.basename(path)
    m = re.search(rf"{re.escape(prefix)}_(\d+)\.nii(\.gz)?$", base)
    if not m:
        # also allow TAG_3 (no extension yet)
        m = re.search(rf"{re.escape(prefix)}_(\d+)$", base)
    return int(m.group(1)) if m else 0

# -------------------------
# Pair discovery + arg building + denoise call
# -------------------------
def plan_pairs(root: str):
    """
    Detect scheme (AP/PA or LR/RL) and return (ap_mag, ap_ph, pa_mag, pa_ph).
    """
    def _sorted(globpat, tag):  # sort by numeric suffix
        return sorted(glob.glob(globpat), key=lambda p: _num_suffix(p, tag))

    # AP/PA layout?
    ap_mag = _sorted(os.path.join(root, "AP",    "AP_*.nii*"),    "AP")
    pa_mag = _sorted(os.path.join(root, "PA",    "PA_*.nii*"),    "PA")
    ap_ph  = _sorted(os.path.join(root, "AP_ph", "AP_ph_*.nii*"), "AP_ph")
    pa_ph  = _sorted(os.path.join(root, "PA_ph", "PA_ph_*.nii*"), "PA_ph")

    ap_pa_present = (len(ap_mag) + len(pa_mag) + len(ap_ph) + len(pa_ph)) > 0

    # LR/RL layout?
    lr_mag = _sorted(os.path.join(root, "LR",    "LR_*.nii*"),    "LR")
    rl_mag = _sorted(os.path.join(root, "RL",    "RL_*.nii*"),    "RL")
    lr_ph  = _sorted(os.path.join(root, "LR_ph", "LR_ph_*.nii*"), "LR_ph")
    rl_ph  = _sorted(os.path.join(root, "RL_ph", "RL_ph_*.nii*"), "RL_ph")

    lr_rl_present = (len(lr_mag) + len(rl_mag) + len(lr_ph) + len(rl_ph)) > 0

    if ap_pa_present and lr_rl_present:
        raise RuntimeError("Mixed AP/PA and LR/RL layouts detected. Use only one scheme.")
    if ap_pa_present:
        return ap_mag, ap_ph, pa_mag, pa_ph
    if lr_rl_present:
        # Map LR→'ap', RL→'pa'
        return lr_mag, lr_ph, rl_mag, rl_ph

    raise RuntimeError("No AP/PA or LR/RL folders found under data_prep.")

class BuildPairsInputSpec(BaseInterfaceInputSpec):
    root     = Directory(exists=True, mandatory=True,
                         desc="data_prep root containing AP/, PA/, AP_ph/, PA_ph/ OR LR/RL")
    deps_mag = InputMultiPath(File(), desc="dependencies on MAG finalisation")
    deps_ph  = InputMultiPath(File(), desc="dependencies on PHASE finalisation")

class BuildPairsOutputSpec(TraitedSpec):
    ap_mag = traits.List(File(), usedefault=True)
    ap_ph  = traits.List(File(), usedefault=True)
    pa_mag = traits.List(File(), usedefault=True)
    pa_ph  = traits.List(File(), usedefault=True)

class BuildPairs(BaseInterface):
    input_spec  = BuildPairsInputSpec
    output_spec = BuildPairsOutputSpec

    def _run_interface(self, runtime):
        ap_mag, ap_ph, pa_mag, pa_ph = plan_pairs(self.inputs.root)
        self._ap_mag = ap_mag or []
        self._ap_ph  = ap_ph  or []
        self._pa_mag = pa_mag or []
        self._pa_ph  = pa_ph  or []
        return runtime

    def _list_outputs(self):
        return {
            "ap_mag": getattr(self, "_ap_mag", []),
            "ap_ph":  getattr(self, "_ap_ph",  []),
            "pa_mag": getattr(self, "_pa_mag", []),
            "pa_ph":  getattr(self, "_pa_ph",  []),
        }

class MakeDenoiseArgsInputSpec(BaseInterfaceInputSpec):
    mags      = InputMultiPath(File(exists=True), mandatory=True,
                               desc="List of MAG NIfTIs for this branch")
    phases    = InputMultiPath(File(), desc="Optional matching PHASE NIfTIs")
    root      = Directory(exists=True, mandatory=True,
                          desc="data_prep root; output subdir(s) will be created here")
    direction = traits.Enum("AP", "PA", usedefault=True)

class MakeDenoiseArgsOutputSpec(TraitedSpec):
    magn    = InputMultiPath(File())
    phase   = InputMultiPath(File())
    domains = traits.List(traits.Str)
    names   = traits.List(traits.Str)
    outdir  = Directory()
    outdirs = traits.List(Directory())

class MakeDenoiseArgs(BaseInterface):
    input_spec  = MakeDenoiseArgsInputSpec
    output_spec = MakeDenoiseArgsOutputSpec

    def _run_interface(self, runtime):
        mags   = list(self.inputs.mags or [])
        phases = list(self.inputs.phases or [])
        root   = self.inputs.root

        def _tag_of(path: str) -> str:
            b = os.path.basename(path)
            m = re.match(r"^(AP|PA|LR|RL)", b)
            if not m:
                raise ValueError(f"Cannot extract direction tag from '{b}'.")
            return m.group(1)

        # Group by tag (preserve AP/PA or LR/RL labels)
        by_tag = {}
        for m in mags:
            tag = _tag_of(m)
            by_tag.setdefault(tag, []).append(m)

        # Sort each tag by numeric suffix
        def _sorted_by_num(paths, tag):
            return sorted(paths, key=lambda p: _num_suffix(p, tag))
        for tag in list(by_tag.keys()):
            by_tag[tag] = _sorted_by_num(by_tag[tag], tag)

        # Prepare phases grouped & sorted by tag too
        phases_by_tag = {}
        for p in phases:
            t = _tag_of(p)
            phases_by_tag.setdefault(t, []).append(p)
        for tag in list(phases_by_tag.keys()):
            phases_by_tag[tag] = _sorted_by_num(phases_by_tag[tag], f"{tag}_ph")

        names, magn, phase, domains, outdirs = [], [], [], [], []

        # Build outputs PER TAG; write to <root>/<TAG>_denoised/
        for tag, group in by_tag.items():
            outdir_tag = os.path.join(root, f"{tag}_denoised")
            os.makedirs(outdir_tag, exist_ok=True)

            tag_phases = phases_by_tag.get(tag, [])

            for i, mpath in enumerate(group, start=1):
                names.append(f"{tag}_{i}")
                magn.append(mpath)

                if i <= len(tag_phases) and os.path.exists(tag_phases[i-1]):
                    phase.append(tag_phases[i-1]); domains.append("complex")
                else:
                    phase.append(mpath);           domains.append("mag")

                outdirs.append(outdir_tag)

        # For compatibility, outdir = root; per-series dirs in outdirs
        self._names, self._magn, self._phase = names, magn, phase
        self._domains, self._outdir, self._outdirs = domains, root, outdirs
        return runtime

    def _list_outputs(self):
        return {
            "magn":    getattr(self, "_magn", []),
            "phase":   getattr(self, "_phase", []),
            "domains": getattr(self, "_domains", []),
            "names":   getattr(self, "_names", []),
            "outdir":  getattr(self, "_outdir", ""),
            "outdirs": getattr(self, "_outdirs", []),
        }

class DenoiseInputSpec(CommandLineInputSpec):
    denoise_sh = File(exists=True, position=0, argstr="%s", mandatory=True,
                      desc="Path to denoise wrapper")
    meth    = traits.Str(argstr="-meth %s",   usedefault=True, default_value="NORDIC")
    domain  = traits.Str(argstr="-domain %s", usedefault=True, default_value="complex")
    magn    = File(exists=True, mandatory=True, argstr="-magn %s")
    phase   = File(exists=True, mandatory=True, argstr="-phase %s")
    name    = traits.Str(mandatory=True, argstr="-name %s")
    oPath   = Directory(mandatory=True, argstr="-oPath %s")

class DenoiseOutputSpec(TraitedSpec):
    out_nii  = File()
    out_bvec = File()
    out_bval = File()

class Denoise(CommandLine):
    _cmd = "bash"
    input_spec  = DenoiseInputSpec
    output_spec = DenoiseOutputSpec

    def _out_stem(self):
        return os.path.join(self.inputs.oPath, self.inputs.name)

    def _list_outputs(self):
        stem = self._out_stem()
        outs = self._outputs().get()
        outs["out_nii"]  = stem + ".nii.gz"
        outs["out_bvec"] = stem + ".bvec"
        outs["out_bval"] = stem + ".bval"
        return outs

    def _run_interface(self, runtime):
        if os.path.exists(self._out_stem() + ".nii.gz"):
            runtime.returncode = 0
            return runtime
        return super()._run_interface(runtime)

class CopySidecarsInputSpec(BaseInterfaceInputSpec):
    magn   = InputMultiPath(File(exists=True), mandatory=True)
    names  = traits.List(traits.Str, mandatory=True)
    outdir = Directory(exists=True, mandatory=True)

class CopySidecarsOutputSpec(TraitedSpec):
    out_bvecs = InputMultiPath(File())
    out_bvals = InputMultiPath(File())

class CopySidecars(BaseInterface):
    input_spec  = CopySidecarsInputSpec
    output_spec = CopySidecarsOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(self.inputs.outdir, exist_ok=True)
        bvecs_out, bvals_out = [], []
        for magn, name in zip(self.inputs.magn, self.inputs.names):
            stem = img_stem(magn)
            src_bvec = stem + ".bvec"
            src_bval = stem + ".bval"
            dst_bvec = os.path.join(self.inputs.outdir, name + ".bvec")
            dst_bval = os.path.join(self.inputs.outdir, name + ".bval")
            if os.path.exists(src_bvec): shutil.copy2(src_bvec, dst_bvec)
            if os.path.exists(src_bval): shutil.copy2(src_bval, dst_bval)
            bvecs_out.append(dst_bvec if os.path.exists(dst_bvec) else "")
            bvals_out.append(dst_bval if os.path.exists(dst_bval) else "")
        self._bvecs, self._bvals = bvecs_out, bvals_out
        return runtime

    def _list_outputs(self):
        return {"out_bvecs": getattr(self, "_bvecs", []),
                "out_bvals": getattr(self, "_bvals", [])}
