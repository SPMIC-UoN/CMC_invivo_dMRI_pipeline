# src/dmri_pipeline/nodes/Gibbs_N4.py

# Standard lib
import os
import sys
import glob
import shutil
import subprocess
from typing import List, Tuple

from nipype.interfaces.base import (
    BaseInterface,
    BaseInterfaceInputSpec,
    TraitedSpec,
    File,
    Directory,
    InputMultiPath,
    traits,
    isdefined,
)


def _stem(p: str) -> str:
    b = os.path.basename(p)
    if b.endswith(".nii.gz"):
        return b[:-7]
    if b.endswith(".nii"):
        return b[:-4]
    return b


def _pair_next_to(nii_path: str, ext: str) -> str:
    """Return sidecar path next to a NIfTI path."""
    d = os.path.dirname(nii_path)
    return os.path.join(d, _stem(nii_path) + ext)


def _pair_from_combined(combined_dir: str, nii_path: str, ext: str) -> str:
    """Return sidecar path from the combined_dir for the given NIfTI basename."""
    return os.path.join(combined_dir, _stem(nii_path) + ext)


def _copy2(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


class GibbsN4InputSpec(BaseInterfaceInputSpec):
    combined_dir = Directory(exists=True, mandatory=True, desc="dir with combined .nii.gz and .bval/.bvec")
    outdir_gibbs = Directory(mandatory=True, desc="output dir for mrdegibbs")
    outdir_n4    = Directory(mandatory=True, desc="output dir for N4 results")

    run_gibbs    = traits.Bool(True,  usedefault=True, desc="run mrdegibbs (2D)")
    run_n4       = traits.Bool(True,  usedefault=True, desc="run dwibiascorrect -ants")

    use_docker        = traits.Bool(True,  usedefault=True, desc="use container runtime")
    container_runtime = traits.Str("docker", usedefault=True, desc="docker or podman")
    interactive_tty   = traits.Bool(True,  usedefault=True, desc="add -it to docker/podman run")

    # Pin to MRtrix 3.0.7
    docker_image = traits.Str("docker.io/mrtrix3/mrtrix3:3.0.7", usedefault=True)
    # gibbs_image = traits.Str("docker.io/mrtrix3/mrtrix3:3.0.7", usedefault=True)
    # n4_image    = traits.Str("docker.io/mrtrix3/mrtrix3:3.0.7", usedefault=True)

    mask = File(exists=True, desc="optional nodif_brain_mask for N4")

    deps = InputMultiPath(File, desc="dummy deps to enforce ordering")


class GibbsN4OutputSpec(TraitedSpec):
    gibbs_files = InputMultiPath(File, desc="mrdegibbs outputs")
    n4_files    = InputMultiPath(File, desc="N4 corrected outputs")
    bias_fields = InputMultiPath(File, desc="N4 bias fields")


class GibbsN4(BaseInterface):
    input_spec  = GibbsN4InputSpec
    output_spec = GibbsN4OutputSpec

    # --------------- helpers ---------------
    def _list_inputs(self) -> Tuple[List[str], List[str], List[str]]:
        d = self.inputs.combined_dir
        imgs  = sorted([f for f in glob.glob(os.path.join(d, "*.nii.gz")) if os.path.isfile(f)])
        bvals = [_pair_from_combined(d, f, ".bval") for f in imgs]
        bvecs = [_pair_from_combined(d, f, ".bvec") for f in imgs]
        return imgs, bvals, bvecs

    def _run_logged(self, cmd: list) -> int:
        """Run a command, stream stdout/stderr into Nipype logs, return exit code."""
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stderr.write(err)
        return proc.returncode

    def _mounts(self, in_dir: str, out_dir: str) -> List[str]:
        return ["-v", f"{in_dir}:/data:ro,z", "-v", f"{out_dir}:/out:z"]

    def _rt_prefix(self) -> List[str]:
        rt = [self.inputs.container_runtime, "run"]
        if self.inputs.interactive_tty:
            rt += ["-it"]
        rt += ["--rm"]
        return rt

    # --------------- main run ---------------

    def _run_interface(self, runtime):
        imgs, bvals, bvecs = self._list_inputs()
        if not imgs:
            raise RuntimeError(f"No .nii.gz in {self.inputs.combined_dir}")

        os.makedirs(self.inputs.outdir_gibbs, exist_ok=True)
        os.makedirs(self.inputs.outdir_n4,    exist_ok=True)

        # --- Gibbs (2D) ---
        gibbs: List[str] = []
        if self.inputs.run_gibbs:
            for i, img in enumerate(imgs):
                stem      = _stem(img)
                out_img   = os.path.join(self.inputs.outdir_gibbs, stem + ".nii.gz")
                src_bval  = bvals[i]
                src_bvec  = bvecs[i]
                if not (os.path.exists(src_bval) and os.path.exists(src_bvec)):
                    raise RuntimeError(f"Missing sidecars for {img} — expected {src_bval} and {src_bvec}")

                # run mrdegibbs
                if self.inputs.use_docker:
                    cmd = self._rt_prefix() + self._mounts(self.inputs.combined_dir, self.inputs.outdir_gibbs) + [
                        self.inputs.docker_image,
                        "mrdegibbs",
                        f"/data/{os.path.basename(img)}",
                        f"/out/{os.path.basename(out_img)}",
                    ]
                else:
                    cmd = ["mrdegibbs", img, out_img]

                rc = self._run_logged(cmd)
                if rc != 0:
                    raise RuntimeError("mrdegibbs failed.")
                if not os.path.exists(out_img):
                    raise RuntimeError(f"mrdegibbs produced no output: {out_img}")

                # copy sidecars into outdir_gibbs with same basename (AP_combined.bval/.bvec)
                dst_bval = os.path.join(self.inputs.outdir_gibbs, stem + ".bval")
                dst_bvec = os.path.join(self.inputs.outdir_gibbs, stem + ".bvec")
                _copy2(src_bval, dst_bval)
                _copy2(src_bvec, dst_bvec)

                gibbs.append(out_img)
        else:
            # reuse existing outputs; also ensure sidecars are present in outdir_gibbs
            for i, img in enumerate(imgs):
                stem     = _stem(img)
                out_img  = os.path.join(self.inputs.outdir_gibbs, stem + ".nii.gz")
                if os.path.exists(out_img):
                    # ensure sidecars exist in outdir_gibbs (copy if missing)
                    src_bval = bvals[i]
                    src_bvec = bvecs[i]
                    dst_bval = os.path.join(self.inputs.outdir_gibbs, stem + ".bval")
                    dst_bvec = os.path.join(self.inputs.outdir_gibbs, stem + ".bvec")
                    if os.path.exists(src_bval) and not os.path.exists(dst_bval):
                        _copy2(src_bval, dst_bval)
                    if os.path.exists(src_bvec) and not os.path.exists(dst_bvec):
                        _copy2(src_bvec, dst_bvec)
                    gibbs.append(out_img)

        # --- N4 (dwibiascorrect ants) ---
        n4:   List[str] = []
        bias: List[str] = []
        if self.inputs.run_n4:
            srcs = gibbs if self.inputs.run_gibbs else imgs
            for img in srcs:
                stem      = _stem(img if self.inputs.run_gibbs else os.path.join(self.inputs.combined_dir, os.path.basename(img)))
                in_dir    = os.path.dirname(img) if self.inputs.run_gibbs else self.inputs.combined_dir
                grad_bval = os.path.join(self.inputs.outdir_gibbs if self.inputs.run_gibbs else in_dir, stem + ".bval")
                grad_bvec = os.path.join(self.inputs.outdir_gibbs if self.inputs.run_gibbs else in_dir, stem + ".bvec")

                if not (os.path.exists(grad_bval) and os.path.exists(grad_bvec)):
                    raise RuntimeError(f"N4: missing sidecars for {img} — expected {grad_bval} and {grad_bvec}")

                oimg   = os.path.join(self.inputs.outdir_n4, stem + "_N4.nii.gz")
                ofield = os.path.join(self.inputs.outdir_n4, stem + "_N4_field.nii.gz")

                if self.inputs.use_docker:
                    mounts = self._mounts(in_dir, self.inputs.outdir_n4)
                    mask_opt = []
                    if isdefined(self.inputs.mask) and self.inputs.mask:
                        mounts += ["-v", f"{os.path.dirname(self.inputs.mask)}:/mask:ro,z"]
                        mask_opt = ["-mask", f"/mask/{os.path.basename(self.inputs.mask)}"]

                    cmd = self._rt_prefix() + mounts + [
                        self.inputs.docker_image,
                        "dwibiascorrect", "ants",
                        f"/data/{os.path.basename(img)}",
                        f"/out/{os.path.basename(oimg)}",
                        "-fslgrad",
                        f"/data/{os.path.basename(grad_bvec)}",
                        f"/data/{os.path.basename(grad_bval)}",
                    ] + mask_opt + ["-bias", f"/out/{os.path.basename(ofield)}"]

                else:
                    mask_opt = ["-mask", self.inputs.mask] if (isdefined(self.inputs.mask) and self.inputs.mask) else []
                    cmd = [
                        "dwibiascorrect", "ants",
                        img, oimg,
                        "-fslgrad", grad_bvec, grad_bval,
                    ] + mask_opt + ["-bias", ofield]

                rc = self._run_logged(cmd)
                if rc != 0:
                    raise RuntimeError("dwibiascorrect failed.")
                if not os.path.exists(oimg):
                    raise RuntimeError(f"N4 produced no output: {oimg}")

                # copy sidecars into outdir_n4 with _N4 suffix
                n4_bval = os.path.join(self.inputs.outdir_n4, stem + "_N4.bval")
                n4_bvec = os.path.join(self.inputs.outdir_n4, stem + "_N4.bvec")
                _copy2(grad_bval, n4_bval)
                _copy2(grad_bvec, n4_bvec)

                n4.append(oimg)
                bias.append(ofield)

        self._gibbs = gibbs
        self._n4    = n4
        self._bias  = bias
        return runtime

    def _list_outputs(self):
        return {
            "gibbs_files": getattr(self, "_gibbs", []),
            "n4_files":    getattr(self, "_n4", []),
            "bias_fields": getattr(self, "_bias", []),
        }