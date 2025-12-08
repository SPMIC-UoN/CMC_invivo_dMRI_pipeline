# src/invivo_dmri_pipeline/nodes/bedpostx_xtract.py

# Standard lib
import os
import glob
import subprocess

# Third party
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, InputMultiPath, traits,
    CommandLine, CommandLineInputSpec, isdefined
)

# Package-local helpers
from ..utils import FSL


# -------------------------
# bedpostx
# -------------------------
class BedpostxInputSpec(CommandLineInputSpec):
    datadir = Directory(position=0, argstr="%s", mandatory=True)
    use_gpu = traits.Bool(True, usedefault=True)
    deps = InputMultiPath(File(), desc="unused; enforce dependency")


class BedpostxOutputSpec(TraitedSpec):
    bpx_dir = Directory()
    dyads1 = File()
    mean_f1 = File()


class Bedpostx(CommandLine):
    _cmd = "bash -lc true"
    input_spec = BedpostxInputSpec
    output_spec = BedpostxOutputSpec

    def _bpx_dir(self):
        return os.path.abspath(self.inputs.datadir.rstrip(os.sep) + ".bedpostX")

    def _is_done(self):
        return os.path.isfile(os.path.join(self._bpx_dir(), "dyads1.nii.gz"))

    def _run_interface(self, runtime):
        os.makedirs(os.path.abspath(self.inputs.datadir), exist_ok=True)
        if self._is_done():
            print(f"[bedpostx] Found existing outputs in {self._bpx_dir()} — skipping.")
            runtime.returncode = 0
            return runtime
        exe = os.path.join(FSL, "bin", "bedpostx_gpu" if self.inputs.use_gpu else "bedpostx")
        subprocess.run([exe, os.path.abspath(self.inputs.datadir)], check=True)
        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        bpx = self._bpx_dir()
        outs = self._outputs().get()
        outs["bpx_dir"] = bpx
        outs["dyads1"] = os.path.join(bpx, "dyads1.nii.gz")
        outs["mean_f1"] = os.path.join(bpx, "mean_f1samples.nii.gz")
        return outs


# -------------------------
# xtract + streamlines per tract + viewer
# -------------------------
class XtractInputSpec(CommandLineInputSpec):
    bpx_dir   = Directory(exists=True, mandatory=True)
    outdir    = Directory(mandatory=True)
    stdref    = File(exists=True, mandatory=True)
    warp_std2anat = File(exists=True, mandatory=True)
    warp_anat2std = File(exists=True, mandatory=True)
    ptx_steplength = traits.Float(1.0, usedefault=True)
    species  = traits.Str("CUSTOM", usedefault=True)
    no_gpu   = traits.Bool(False, usedefault=True)
    profiles_dir = Directory(exists=True, desc="XTRACT profiles dir (optional when using -species)")

    # --- fsl-streamlines controls ---
    do_streamlines = traits.Bool(True, usedefault=True)
    streamlines_format = traits.Enum("trk", "vtk", usedefault=True)
    streamlines_density_threshold = traits.Float(1e-3, usedefault=True)
    streamlines_ptx2_prefix = traits.Str("densityNorm", usedefault=True,
                                         desc="PTX2/XTRACT file prefix (e.g., fdt_paths or densityNorm)")
    streamlines_num_jobs = traits.Int(1, usedefault=True)

    # --- QC viewer ---
    viewer_script = File(exists=True, desc="Path to tract_viewer.py (optional)")
    do_viewer = traits.Bool(True, usedefault=True, desc="Run tract_viewer after streamlines")


class XtractOutputSpec(TraitedSpec):
    outdir = Directory()
    streamlines_files = InputMultiPath(File(), desc="per-tract streamline files")


class Xtract(CommandLine):
    _cmd = "bash -lc true"
    input_spec = XtractInputSpec
    output_spec = XtractOutputSpec

    def _run_interface(self, runtime):
        outdir = os.path.abspath(self.inputs.outdir)
        os.makedirs(outdir, exist_ok=True)

        if os.path.isdir(outdir) and any(os.scandir(outdir)):
            print(f"[xtract] Found existing outputs in {outdir} — skipping XTRACT run.")
        else:
            ptx_opts = os.path.join(outdir, "ptx_opts.txt")
            with open(ptx_opts, "w") as f:
                f.write(f"--steplength={self.inputs.ptx_steplength} --savepaths --opathdir")

            cmd = [
                os.path.join(FSL, "bin", "xtract"),
                "-bpx", os.path.abspath(self.inputs.bpx_dir),
                "-out", outdir,
                "-species", str(self.inputs.species),
                "-stdref", os.path.abspath(self.inputs.stdref),
                "-stdwarp", os.path.abspath(self.inputs.warp_std2anat), os.path.abspath(self.inputs.warp_anat2std),
                "-ptx_options", ptx_opts,
            ]
            if isdefined(self.inputs.profiles_dir) and self.inputs.profiles_dir:
                pdir = os.path.abspath(self.inputs.profiles_dir)
                cmd += ["-p", pdir, "-str", os.path.join(pdir, "structureList")]
            if not self.inputs.no_gpu:
                cmd.append("-gpu")
            subprocess.run(cmd, check=True)

            if not self.inputs.no_gpu:
                cmd.append("-gpu")

            subprocess.run(cmd, check=True)

        # --- fsl_streamlines: run once per tract directory ---
        self._stream_files = []
        if self.inputs.do_streamlines:
            tracts_root = os.path.join(outdir, "tracts")
            tract_dirs = sorted([d for d in glob.glob(os.path.join(tracts_root, "*")) if os.path.isdir(d)])
            if not tract_dirs:
                raise RuntimeError(f"[fsl_streamlines] No tract folders found at {tracts_root}")

            fmt = self.inputs.streamlines_format  # trk|vtk
            ext = ".trk" if fmt == "trk" else ".vtk"

            for tract_dir in tract_dirs:
                tract_name = os.path.basename(tract_dir.rstrip(os.sep))
                out_prefix = os.path.join(tract_dir, tract_name)

                sl_cmd = [
                    "fsl_streamlines",
                    tract_dir,
                    "-o", out_prefix,
                    "-f", fmt,
                    "-p", str(self.inputs.streamlines_ptx2_prefix),
                    "-t", str(self.inputs.streamlines_density_threshold),
                    "-nj", str(int(self.inputs.streamlines_num_jobs)),
                ]
                subprocess.run(sl_cmd, check=True)
                self._stream_files.append(out_prefix + ext)

        # --- tract viewer (QC PNGs) ---
        if self.inputs.do_viewer and isdefined(self.inputs.viewer_script) and self.inputs.viewer_script:
            tracts_root = os.path.join(outdir, "tracts")
            viewer_cmd = [
                os.path.abspath(self.inputs.viewer_script),
                "--xtract", tracts_root,
                "--brain", os.path.abspath(self.inputs.stdref),
                "--outfile", outdir
            ]
            subprocess.run(viewer_cmd, check=True)

        runtime.returncode = 0
        return runtime

    def _list_outputs(self):
        outs = self._outputs().get()
        outs["outdir"] = os.path.abspath(self.inputs.outdir)
        outs["streamlines_files"] = getattr(self, "_stream_files", [])
        return outs
