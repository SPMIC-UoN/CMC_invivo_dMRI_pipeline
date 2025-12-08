# src/dmri_pipeline/nodes/dti_and_reg.py

# Standard lib
import os
import shutil
import subprocess
import tempfile
import re

# Third party
import numpy as np
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, traits, CommandLine, CommandLineInputSpec, isdefined
)

# Package-local
from ..utils import FSL, fslval


# -------------------------
# Select lower shell (buffer-overflow safe)
# -------------------------
class SelectLowerShellInputSpec(CommandLineInputSpec):
    data = File(exists=True, position=0, argstr="%s", mandatory=True)
    bvals = File(exists=True, position=1, argstr="%s", mandatory=True)
    outprefix = traits.Str(position=2, argstr="%s", mandatory=True)
    approx_b = traits.Int(position=3, argstr="%d", usedefault=True, value=1000)
    b0 = traits.Int(argstr="-b %d", usedefault=True, value=0)
    db = traits.Int(argstr="-db %d", usedefault=True, value=60)
    bvecs = File(exists=True, argstr="-obv %s", mandatory=True)


class SelectLowerShellOutputSpec(TraitedSpec):
    out_data = File()
    out_bvals = File()
    out_bvecs = File()


class SelectLowerShell(CommandLine):
    """
    If timepoints > 400:
      - chunk with fslroi (<=380 per chunk)
      - slice sidecars; run select_dwi_vols for each chunk
      - merge outputs (fslmerge) and cat sidecars
    Else: call select_dwi_vols directly.
    """
    _cmd = os.path.join(FSL, "bin", "select_dwi_vols")
    input_spec = SelectLowerShellInputSpec
    output_spec = SelectLowerShellOutputSpec

    def _list_outputs(self):
        op = os.path.abspath(self.inputs.outprefix)
        return {"out_data": op + ".nii.gz",
                "out_bvals": op + ".bval",
                "out_bvecs": op + ".bvec"}

    def _nonempty(self, p: str) -> bool:
        try:
            return os.path.isfile(p) and os.path.getsize(p) > 0
        except Exception:
            return False

    def _call(self, args):
        subprocess.run(args, check=True)

    def _chunk_select(self, runtime):
        data = os.path.abspath(self.inputs.data)
        bvals_in = os.path.abspath(self.inputs.bvals)
        bvecs_in = os.path.abspath(self.inputs.bvecs)
        outpref = os.path.abspath(self.inputs.outprefix)
        approx_b = int(self.inputs.approx_b)
        b0 = int(self.inputs.b0)
        db = int(self.inputs.db)

        os.makedirs(os.path.dirname(outpref), exist_ok=True)

        nvol = int(fslval(data, "dim4"))
        max_chunk = 380
        starts = list(range(0, nvol, max_chunk))

        Bvals = np.loadtxt(bvals_in).reshape(-1)
        Bvecs = np.loadtxt(bvecs_in)
        if Bvecs.ndim == 1:
            Bvecs = Bvecs.reshape(3, -1)

        tmpdir = tempfile.mkdtemp(prefix="select_ls_chunks_",
                                  dir=os.path.dirname(outpref))

        chunk_out_niis = []
        chunk_bvals_cat = []
        chunk_bvecs_cat = []

        try:
            for s in starts:
                count = min(max_chunk, nvol - s)
                span = slice(s, s + count)

                c_dat = os.path.join(tmpdir, f"chunk_{s:06d}.nii.gz")
                c_bval = os.path.join(tmpdir, f"chunk_{s:06d}.bval")
                c_bvec = os.path.join(tmpdir, f"chunk_{s:06d}.bvec")
                c_out = os.path.join(tmpdir, f"chunk_{s:06d}_sel")

                # 4D chunk
                self._call([os.path.join(FSL, "bin", "fslroi"), data, c_dat, str(s), str(count)])

                # sidecars
                np.savetxt(c_bval, Bvals[span][None, :].astype(int), fmt="%d")
                np.savetxt(c_bvec, Bvecs[:, span], fmt="%.8f")

                # select within chunk
                self._call([self._cmd,
                            c_dat, c_bval, c_out, str(approx_b),
                            "-b", str(b0),
                            "-obv", c_bvec,
                            "-db", str(db)])

                c_sel_nii = c_out + ".nii.gz"
                c_sel_bval = c_out + ".bval"
                c_sel_bvec = c_out + ".bvec"

                if self._nonempty(c_sel_nii):
                    chunk_out_niis.append(c_sel_nii)
                    if self._nonempty(c_sel_bval):
                        chunk_bvals_cat.append(np.loadtxt(c_sel_bval).reshape(-1))
                    if self._nonempty(c_sel_bvec):
                        C = np.loadtxt(c_sel_bvec)
                        if C.ndim == 1:
                            C = C.reshape(3, 1)
                        chunk_bvecs_cat.append(C)

            outs = self._list_outputs()
            if chunk_out_niis:
                self._call([os.path.join(FSL, "bin", "fslmerge"), "-t", outs["out_data"]] + chunk_out_niis)
            else:
                raise RuntimeError("No volumes selected in any chunk.")

            if chunk_bvals_cat:
                bcat = np.concatenate([bv.ravel() for bv in chunk_bvals_cat])[None, :]
                np.savetxt(outs["out_bvals"], bcat.astype(int), fmt="%d")
            else:
                open(outs["out_bvals"], "w").close()

            if chunk_bvecs_cat:
                vcat = np.concatenate(chunk_bvecs_cat, axis=1)
                np.savetxt(outs["out_bvecs"], vcat, fmt="%.8f")
            else:
                open(outs["out_bvecs"], "w").close()

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        runtime.returncode = 0
        return runtime

    def _run_interface(self, runtime):
        nvol = int(fslval(self.inputs.data, "dim4"))
        if nvol > 400:
            print(f"[select_lower_shell] Using chunked selection (nvol={nvol}).")
            return self._chunk_select(runtime)

        print(f"[select_lower_shell] Using direct select_dwi_vols (nvol={nvol}).")
        runtime = super()._run_interface(runtime)

        # Regenerate sidecars if the helper didn't write them
        outs = self._list_outputs()
        if (not self._nonempty(outs["out_bvals"])) or (not self._nonempty(outs["out_bvecs"])):
            print("[select_lower_shell] Fallback: regenerating .bval/.bvec from indices.")
            bvals = np.loadtxt(self.inputs.bvals).reshape(-1)
            tol = int(self.inputs.db)
            bsel = int(self.inputs.approx_b)
            idx_shell = [i for i, b in enumerate(bvals) if abs(b - bsel) < tol]
            idx_b0 = [i for i, b in enumerate(bvals) if abs(b - 0) < tol]
            idx = idx_shell + idx_b0
            np.savetxt(outs["out_bvals"], bvals[idx][None, :].astype(int), fmt="%d")
            B = np.loadtxt(self.inputs.bvecs)
            if B.ndim == 1:
                B = B.reshape(3, -1)
            np.savetxt(outs["out_bvecs"], B[:, idx], fmt="%.8f")
        return runtime


# -------------------------
# DTIFIT
# -------------------------
class DTIFITInputSpec(CommandLineInputSpec):
    k = File(exists=True, argstr="-k %s", mandatory=True)
    m = File(exists=True, argstr="-m %s", mandatory=True)
    o = traits.Str(argstr="-o %s", mandatory=True)
    r = File(exists=True, argstr="-r %s", mandatory=True)
    b = File(exists=True, argstr="-b %s", mandatory=True)
    save_tensor = traits.Bool(True, usedefault=True, argstr="--save_tensor",
                              desc="save the diffusion tensor")


class DTIFITOutputSpec(TraitedSpec):
    fa = File()
    md = File()
    v1 = File()
    tensor = File()


class DTIFIT(CommandLine):
    _cmd = os.path.join(FSL, "bin", "dtifit")
    input_spec = DTIFITInputSpec
    output_spec = DTIFITOutputSpec

    def _run_interface(self, runtime):
        os.makedirs(os.path.dirname(self.inputs.o), exist_ok=True)
        return super()._run_interface(runtime)

    def _list_outputs(self):
        pref = os.path.abspath(self.inputs.o)
        return {
            "fa": pref + "_FA.nii.gz",
            "md": pref + "_MD.nii.gz",
            "v1": pref + "_V1.nii.gz",
            "tensor": pref + "_tensor.nii.gz",
        }


# -------------------------
# Registration to standard: MMORF (default) or FNIRT
# -------------------------
class Reg2StdInputSpec(BaseInterfaceInputSpec):
    # Method
    method = traits.Enum("mmorf", "fnirt", usedefault=True,
                         desc="Registration: 'mmorf' (default) or 'fnirt'")

    # Subject inputs
    fa = File(exists=True, mandatory=True, desc="Subject FA (from DTIFIT)")
    tensor = File(exists=True, mandatory=True, desc="Subject tensor (from DTIFIT --save_tensor)")
    outdir = Directory(mandatory=True, desc="Output dir (STDreg/)")

    # Templates
    atl_fa = File(exists=True, mandatory=True, desc="Template FA")
    atl_tensor = File(exists=True, mandatory=True, desc="Template tensor")

    # MMORF config template (only used if method=mmorf)
    mmorf_config_template = File(exists=True, desc="files/mmorf_config_template.ini")

    # Mode & overrides
    mode = traits.Enum("nhp", "hum", usedefault=True,
                       desc="Controls MMORF parameterisation")
    warp_res_init_arg = traits.Int(desc="Override warp_res_init. If not set, nhp=16, hum=64.")
    # Optional override for all fwhm_* lists
    fwhm_override = traits.Str(desc="Optional override for all fwhm_* lists (space-separated)")


class Reg2StdOutputSpec(TraitedSpec):
    outdir = Directory()
    warp_anat2std = File()
    warp_std2anat = File()
    fa_in_std = File()


class Reg2Std(BaseInterface):
    input_spec = Reg2StdInputSpec
    output_spec = Reg2StdOutputSpec

    def _fsl(self, exe: str) -> str:
        return os.path.join(FSL, "bin", exe)

    def _run_cmd(self, args: list[str]) -> None:
        subprocess.run(args, check=True)

    def _list_outputs(self):
        od = os.path.abspath(self.inputs.outdir)
        return {
            "outdir": od,
            "warp_anat2std": os.path.join(od, "stdreg_anat_to_std_warp.nii.gz"),
            "warp_std2anat": os.path.join(od, "stdreg_std_to_anat_warp.nii.gz"),
            "fa_in_std": os.path.join(od, "FA_in_STD.nii.gz"),
        }

    def _flirt_affine(self, fa: str, ref_fa: str, outdir: str) -> str:
        """Create initial affine using FLIRT (FA->template FA)."""
        flirt = self._fsl("flirt")
        aff = os.path.join(outdir, "stdreg_anat_to_std.mat")
        fa_lin = os.path.join(outdir, "stdreg_anat_to_std")
        self._run_cmd([flirt, "-in", fa, "-ref", ref_fa, "-omat", aff, "-out", fa_lin])
        return aff

    def _assert_mmorf_ini_filled(self, text: str, src_path: str):
        """Raise if any of the literal placeholders are still present."""
        placeholders = [
            "FSLDIR",
            "std_ref_fa_file",
            "std_ref_tensor_file",
            "subject_fa_file",
            "subject_tensor_file",
            "subject_aff_mat",
        ]
        leftovers = [p for p in placeholders if p in text]
        if leftovers:
            raise RuntimeError(
                "MMORF config still contains unresolved placeholders: "
                + ", ".join(leftovers) + f"\nFrom template: {src_path}"
            )

    def _mmorf_mode_params(self):
        """Return warp_res_init and FWHM list according to mode, unless overridden."""
        # warp_res_init
        if isdefined(self.inputs.warp_res_init_arg) and self.inputs.warp_res_init_arg:
            warp_res_init = int(self.inputs.warp_res_init_arg)
        else:
            warp_res_init = 16 if self.inputs.mode == "nhp" else 64

        # FWHM schedule
        if isdefined(self.inputs.fwhm_override) and self.inputs.fwhm_override:
            fset = [float(x) for x in self.inputs.fwhm_override.split()]
        else:
            if self.inputs.mode == "nhp":
                fset = [4.0, 4.0, 2.0, 1.0, 0.5, 0.25, 0.125]
            else:
                fset = [16.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5]

        return warp_res_init, fset

    def _run_mmorf(self, fa: str, tensor: str, ref_fa: str, ref_tensor: str, aff_mat: str, outdir: str):
        """MMORF with scalar (FA) + tensor, INI templated then mode-overridden."""
        outs = self._list_outputs()
        cfg_in = os.path.abspath(self.inputs.mmorf_config_template)
        cfg_out = os.path.join(outdir, "mmorf_config.ini")

        # Start from template
        with open(cfg_in, "r") as f:
            txt = f.read()

        # Basic placeholder replacement
        fsl_root = os.environ.get("FSLDIR", os.path.dirname(os.path.dirname(self._fsl("flirt"))))
        repl = {
            "FSLDIR": fsl_root,
            "std_ref_fa_file": os.path.abspath(ref_fa),
            "std_ref_tensor_file": os.path.abspath(ref_tensor),
            "subject_fa_file": os.path.abspath(fa),
            "subject_tensor_file": os.path.abspath(tensor),
            "subject_aff_mat": os.path.abspath(aff_mat),
        }
        for k, v in repl.items():
            txt = txt.replace(k, v)

        # Mode-dependent overrides for warp_res_init and all FWHMs
        warp_res_init, fset = self._mmorf_mode_params()
        fset_str = " ".join(f"{v:g}" for v in fset)

        # These rely on the exact placeholder forms in your template:
        #   warp_res_init           = warp_res_init_arg
        #   fwhm_ref_scalar     = fwhm_set
        #   fwhm_mov_scalar     = fwhm_set
        #   fwhm_ref_tensor     = fwhm_set
        #   fwhm_mov_tensor     = fwhm_set
        txt = txt.replace(
            "warp_res_init           = warp_res_init_arg",
            f"warp_res_init           = {warp_res_init}"
        )
        txt = txt.replace(
            "fwhm_ref_scalar     = fwhm_set",
            f"fwhm_ref_scalar     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_mov_scalar     = fwhm_set",
            f"fwhm_mov_scalar     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_ref_tensor     = fwhm_set",
            f"fwhm_ref_tensor     = {fset_str}"
        )
        txt = txt.replace(
            "fwhm_mov_tensor     = fwhm_set",
            f"fwhm_mov_tensor     = {fset_str}"
        )

        # Sanity: make sure required keys are there and no core placeholders remain
        if "warp_res_init" not in txt:
            raise RuntimeError("MMORF INI missing 'warp_res_init' (check your template).")
        self._assert_mmorf_ini_filled(txt, cfg_in)

        with open(cfg_out, "w") as f:
            f.write(txt)

        # Nonlinear (subject->std)
        nl_warp = os.path.join(outdir, "mmorf_nonlin_warp.nii.gz")
        self._run_cmd(["mmorf", "--config", cfg_out, "--warp_out", nl_warp])

        # Combine with affine to template
        self._run_cmd([
            self._fsl("convertwarp"),
            "-m", aff_mat,
            "-w", nl_warp,
            "-r", ref_fa,
            "-o", outs["warp_anat2std"],
        ])

        # Apply to FA
        self._run_cmd([
            self._fsl("applywarp"),
            "-i", fa,
            "-r", ref_fa,
            "-w", outs["warp_anat2std"],
            "-o", outs["fa_in_std"],
        ])

        # Inverse (std->anat)
        self._run_cmd([
            self._fsl("invwarp"),
            "-w", outs["warp_anat2std"],
            "-o", outs["warp_std2anat"],
            "-r", fa,
        ])

    def _run_fnirt(self, fa: str, ref_fa: str, aff_mat: str, outdir: str):
        """Classic FLIRT+FNIRT using FA contrast."""
        outs = self._list_outputs()

        self._run_cmd([
            self._fsl("fnirt"),
            f"--in={fa}",
            f"--ref={ref_fa}",
            f"--aff={aff_mat}",
            f"--cout={outs['warp_anat2std']}",
            f"--iout={outs['fa_in_std']}"
        ])

        self._run_cmd([
            self._fsl("invwarp"),
            "-w", outs["warp_anat2std"],
            "-o", outs["warp_std2anat"],
            "-r", fa
        ])

    def _run_interface(self, runtime):
        fa = os.path.abspath(self.inputs.fa)
        tensor = os.path.abspath(self.inputs.tensor)
        ref_fa = os.path.abspath(self.inputs.atl_fa)
        ref_ten = os.path.abspath(self.inputs.atl_tensor)
        outdir = os.path.abspath(self.inputs.outdir)
        os.makedirs(outdir, exist_ok=True)

        # 1) Initial affine
        aff_mat = self._flirt_affine(fa, ref_fa, outdir)

        # 2) Nonlinear
        if self.inputs.method == "mmorf":
            if not isdefined(self.inputs.mmorf_config_template) or not self.inputs.mmorf_config_template:
                raise RuntimeError("mmorf_config_template must be set when method='mmorf'.")
            self._run_mmorf(fa, tensor, ref_fa, ref_ten, aff_mat, outdir)
        else:
            self._run_fnirt(fa, ref_fa, aff_mat, outdir)

        return runtime
