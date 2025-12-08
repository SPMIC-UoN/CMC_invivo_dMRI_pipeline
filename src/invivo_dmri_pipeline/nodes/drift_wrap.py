# src/dmri_pipeline/nodes/drift_wrap.py

# Standard lib
import os
import shutil

# Third party
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, InputMultiPath, traits,
    CommandLine, CommandLineInputSpec
)

# -------------------------
# Drift correction (wrapper around drift.py)
# -------------------------
class DriftInputSpec(CommandLineInputSpec):
    script = File(exists=True, position=0, argstr="%s", mandatory=True)
    input  = traits.List(File(exists=True), argstr="-i %s",     mandatory=True, sep=" ")
    bvals  = traits.List(File(exists=True), argstr="-bvals %s", mandatory=True, sep=" ")
    # Default is the first positional value ("single")
    model  = traits.Enum("single", "linear", "quadratic", usedefault=True, argstr="-model %s")
    outdir = traits.Str(argstr="-outdir %s", usedefault=True, default_value="drift_analysis")
    output = traits.List(traits.Str, argstr="--output %s", sep=" ")

class DriftOutputSpec(TraitedSpec):
    out_files = traits.List(File())

class Drift(CommandLine):
    _cmd = "python3"
    input_spec  = DriftInputSpec
    output_spec = DriftOutputSpec

    def _list_outputs(self):
        outs = self._outputs().get()
        outdir = os.path.abspath(self.inputs.outdir)
        outs["out_files"] = [os.path.join(outdir, p) for p in (self.inputs.output or [])]
        return outs

# -------------------------
# Copy sidecars
# -------------------------
class CopyPairInputSpec(BaseInterfaceInputSpec):
    src_bval   = File(exists=True, mandatory=True)
    src_bvec   = File(exists=True, mandatory=True)
    dst_prefix = traits.Str(mandatory=True)
    deps       = InputMultiPath(File())

class CopyPairOutputSpec(TraitedSpec):
    out_bval = File()
    out_bvec = File()

class CopyPair(BaseInterface):
    input_spec  = CopyPairInputSpec
    output_spec = CopyPairOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(os.path.dirname(self.inputs.dst_prefix), exist_ok=True)
        out_bval = self.inputs.dst_prefix + ".bval"
        out_bvec = self.inputs.dst_prefix + ".bvec"
        shutil.copy2(self.inputs.src_bval, out_bval)
        shutil.copy2(self.inputs.src_bvec, out_bvec)
        self._out_bval = os.path.abspath(out_bval)
        self._out_bvec = os.path.abspath(out_bvec)
        return runtime

    def _list_outputs(self):
        return {"out_bval": self._out_bval, "out_bvec": self._out_bvec}
