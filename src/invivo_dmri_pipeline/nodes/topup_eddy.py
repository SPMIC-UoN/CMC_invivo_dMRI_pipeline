#!/usr/bin/env python

import os
import shutil
import subprocess
from typing import List, Tuple
# Third party
from nipype.interfaces.base import (
    BaseInterface, BaseInterfaceInputSpec, TraitedSpec,
    File, Directory, InputMultiPath, traits,
    CommandLine, CommandLineInputSpec, isdefined
)

# Package-local helpers
from ..utils import FSL


def _run(cmd: List[str]) -> None:
    """Run a shell command with error checking."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

# --- live streaming helper ---
def _run_and_stream(cmd, env=None):
    """
    Run a subprocess while streaming stdout/stderr to the terminal in real time.
    Raises RuntimeError on non-zero exit.
    """
    print("[cmd]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")

def _read_bvals(path: str) -> List[float]:
    with open(path, "r") as f:
        txt = f.read().strip().split()
    return [float(x) for x in txt]


def _read_bvecs(path: str) -> Tuple[List[float], List[float], List[float]]:
    with open(path, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]
    if len(lines) != 3:
        raise RuntimeError(f"Expected 3 lines in bvecs file, found {len(lines)} in {path}")
    return (
        [float(x) for x in lines[0]],
        [float(x) for x in lines[1]],
        [float(x) for x in lines[2]],
    )


def _write_bvals(path: str, vals: List[float]) -> None:
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")


def _write_bvecs(path: str, vx: List[float], vy: List[float], vz: List[float]) -> None:
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.8g}" for v in vx) + "\n")
        f.write(" ".join(f"{v:.8g}" for v in vy) + "\n")
        f.write(" ".join(f"{v:.8g}" for v in vz) + "\n")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ======================================================================
# 1. PrepareTopupEddy
# ======================================================================

class PrepareTopupEddyInputSpec(BaseInterfaceInputSpec):
    # Drift corrected concatenated AP / PA
    ap_file = File(exists=True, mandatory=True, desc="AP drift corrected 4D")
    pa_file = File(exists=True, mandatory=True, desc="PA drift corrected 4D")

    # Sidecars for drift corrected AP / PA
    ap_bval = File(exists=True, mandatory=True)
    ap_bvec = File(exists=True, mandatory=True)
    pa_bval = File(exists=True, mandatory=True)
    pa_bvec = File(exists=True, mandatory=True)

    # Per-run dim4 lengths from the data fed into Combine
    ap_run_lengths = traits.List(
        traits.Int,
        mandatory=True,
        desc="number of volumes in each AP run",
    )
    pa_run_lengths = traits.List(
        traits.Int,
        mandatory=True,
        desc="number of volumes in each PA run",
    )

    out_root = Directory(mandatory=True, desc="root processing directory")
    pedir_axis = traits.Enum("auto", "i", "j", "k", usedefault=True, desc="phase encode axis")
    echo_ms = traits.Float(mandatory=True, desc="echo spacing (ms) or similar")
    pifactor = traits.Int(mandatory=True, desc="parallel imaging factor")
    b0max = traits.Float(usedefault=True, default_value=60.0, desc="b threshold for b0")


class PrepareTopupEddyOutputSpec(TraitedSpec):
    # For TOPUP
    pos_neg_b0 = File(desc="AP/PA b0s for TOPUP")
    acqparams = File(desc="acqparams file for TOPUP (shared values with EDDY)")
    topup_config = File(desc="config file for TOPUP")

    # For EDDY
    pos_neg = File(desc="AP/PA drift corrected 4D for EDDY")
    acqparams_eddy = File(desc="acqparams file for EDDY")
    idx_txt = File(desc="index file for EDDY")
    bvals_all = File(desc="combined bvals for EDDY")
    bvecs_all = File(desc="combined bvecs for EDDY")
    series_idx = File(desc="per volume series file for EDDY")


class PrepareTopupEddy(BaseInterface):
    input_spec = PrepareTopupEddyInputSpec
    output_spec = PrepareTopupEddyOutputSpec

    def _vector_for_direction(self, ap_like: bool) -> Tuple[float, float, float]:
        """
        Return a 3 component phase encode direction vector.
        For AP / PA, j axis (0, ±1, 0).
        For LR / RL, i axis (±1, 0, 0).
        """
        pedir = self.inputs.pedir_axis
        if pedir == "auto":
            base_ap = os.path.basename(self.inputs.ap_file).upper()
            if "AP" in base_ap or "PA" in base_ap:
                pedir = "j"
            elif "LR" in base_ap or "RL" in base_ap:
                pedir = "i"
            else:
                pedir = "j"

        if pedir == "i":
            return (-1.0, 0.0, 0.0) if ap_like else (1.0, 0.0, 0.0)
        if pedir == "j":
            return (0.0, -1.0, 0.0) if ap_like else (0.0, 1.0, 0.0)
        if pedir == "k":
            return (0.0, 0.0, -1.0) if ap_like else (0.0, 0.0, 1.0)

        # Fallback
        return (0.0, -1.0, 0.0) if ap_like else (0.0, 1.0, 0.0)

    def _total_readout(self) -> float:
        """
        Compute total readout time (s)

            nPEsteps  = dimPhaseEncode - 1
            ro_time_s = (echo_ms / pifactor) * nPEsteps / 1000

        where:
          - echo_ms   is echo spacing in ms
          - pifactor  is the GRAPPA / parallel imaging factor
        """
        ap_file = os.path.abspath(self.inputs.ap_file)
        pedir = self.inputs.pedir_axis
        echo_ms = float(self.inputs.echo_ms)
        pifactor = int(self.inputs.pifactor)

        if pifactor <= 0:
            raise RuntimeError("pifactor must be > 0")
        
        if pedir == "auto":
            base_ap = os.path.basename(ap_file).upper()
            if "AP" in base_ap or "PA" in base_ap:
                pedir = "j"
            elif "LR" in base_ap or "RL" in base_ap:
                pedir = "i"
            else:
                pedir = "j"

        # Map pedir → phase-encode dimension
        if pedir == "i":
            dim = "dim1"
        elif pedir == "j":
            dim = "dim2"
        elif pedir == "k":
            dim = "dim3"
        else:
            dim = "dim2"  # fallback
        
        fslval = os.path.join(FSL, "bin", "fslval")
        dimP_str = subprocess.check_output(
            [fslval, ap_file, dim],
            text=True
        ).strip().split()[0]
        dimP = int(dimP_str)

        nPEsteps = dimP - 1
        ro_time_ms = (echo_ms / pifactor) * nPEsteps
        ro_time_s = ro_time_ms / 1000.0

        return float(f"{ro_time_s:.6f}")

    def _pick_first_b0_indices(
        self,
        bvals: List[float],
        run_lengths: List[int],
        b0max: float,
    ) -> List[int]:
        idxs: List[int] = []
        offset = 0
        for L in run_lengths:
            found = None
            for rel in range(L):
                if bvals[offset + rel] <= b0max:
                    found = offset + rel
                    break
            if found is not None:
                idxs.append(found)
            offset += L
        return idxs

    def _run_interface(self, runtime):
        ap_file = os.path.abspath(self.inputs.ap_file)
        pa_file = os.path.abspath(self.inputs.pa_file)
        ap_bval = os.path.abspath(self.inputs.ap_bval)
        ap_bvec = os.path.abspath(self.inputs.ap_bvec)
        pa_bval = os.path.abspath(self.inputs.pa_bval)
        pa_bvec = os.path.abspath(self.inputs.pa_bvec)

        ap_run_lengths = list(self.inputs.ap_run_lengths)
        pa_run_lengths = list(self.inputs.pa_run_lengths)

        out_root = os.path.abspath(self.inputs.out_root)
        topup_dir = os.path.join(out_root, "topup")
        eddy_dir = os.path.join(out_root, "eddy")
        _ensure_dir(topup_dir)
        _ensure_dir(eddy_dir)

        # ------------------------------------------------------------------
        # 1) Build Pos_Neg (AP + PA) and combined sidecars for EDDY
        # ------------------------------------------------------------------
        pos_neg = os.path.join(eddy_dir, "Pos_Neg.nii.gz")
        _run(["fslmerge", "-t", pos_neg, ap_file, pa_file])

        ap_bvals = _read_bvals(ap_bval)
        pa_bvals = _read_bvals(pa_bval)
        vx_ap, vy_ap, vz_ap = _read_bvecs(ap_bvec)
        vx_pa, vy_pa, vz_pa = _read_bvecs(pa_bvec)

        if len(ap_bvals) != len(vx_ap) or len(pa_bvals) != len(vx_pa):
            raise RuntimeError("Mismatch between bvals and bvecs length in AP or PA.")

        bvals_all = ap_bvals + pa_bvals
        vx_all = vx_ap + vx_pa
        vy_all = vy_ap + vy_pa
        vz_all = vz_ap + vz_pa

        bvals_all_path = os.path.join(eddy_dir, "bvals")
        bvecs_all_path = os.path.join(eddy_dir, "bvecs")
        _write_bvals(bvals_all_path, bvals_all)
        _write_bvecs(bvecs_all_path, vx_all, vy_all, vz_all)

        # ------------------------------------------------------------------
        # 2) Build EDDY acqparams (1 row per run) and index (1 entry per vol)
        # ------------------------------------------------------------------
        n_ap_runs = len(ap_run_lengths)
        n_pa_runs = len(pa_run_lengths)

        if sum(ap_run_lengths) != len(ap_bvals):
            raise RuntimeError(
                f"Sum(ap_run_lengths) = {sum(ap_run_lengths)} "
                f"but AP bvals length = {len(ap_bvals)}"
            )
        if sum(pa_run_lengths) != len(pa_bvals):
            raise RuntimeError(
                f"Sum(pa_run_lengths) = {sum(pa_run_lengths)} "
                f"but PA bvals length = {len(pa_bvals)}"
            )
        
        gx_ap, gy_ap, gz_ap = self._vector_for_direction(ap_like=True)
        gx_pa, gy_pa, gz_pa = self._vector_for_direction(ap_like=False)

        # ------------------------------------------------------------------
        # Build acqparams.txt: integers for direction, float for readout time
        # ------------------------------------------------------------------
        ro = self._total_readout()

        def _int_vec(x):
            return int(round(x))

        gx_ap_i = _int_vec(gx_ap)
        gy_ap_i = _int_vec(gy_ap)
        gz_ap_i = _int_vec(gz_ap)

        gx_pa_i = _int_vec(gx_pa)
        gy_pa_i = _int_vec(gy_pa)
        gz_pa_i = _int_vec(gz_pa)

        acqp_lines: List[str] = []
        for _ in range(n_ap_runs):
            acqp_lines.append(f"{gx_ap_i:d} {gy_ap_i:d} {gz_ap_i:d} {ro:.6f}")
        for _ in range(n_pa_runs):
            acqp_lines.append(f"{gx_pa_i:d} {gy_pa_i:d} {gz_pa_i:d} {ro:.6f}")

        acqparams_eddy = os.path.join(eddy_dir, "acqparams.txt")
        with open(acqparams_eddy, "w") as f:
            f.write("\n".join(acqp_lines) + "\n")

        # Build index.txt: for each volume in AP then PA, assign run ID 1..(n_ap + n_pa)
        index_vals: List[int] = []
        series_vals: List[int] = []

        run_id = 1
        for L in ap_run_lengths:
            index_vals.extend([run_id] * L)
            series_vals.extend([run_id] * L)
            run_id += 1

        for L in pa_run_lengths:
            index_vals.extend([run_id] * L)
            series_vals.extend([run_id] * L)
            run_id += 1

        if len(index_vals) != len(bvals_all):
            raise RuntimeError(
                f"index length {len(index_vals)} != number of volumes {len(bvals_all)}"
            )

        idx_txt = os.path.join(eddy_dir, "index.txt")
        with open(idx_txt, "w") as f:
            f.write(" ".join(str(i) for i in index_vals) + "\n")

        series_idx = os.path.join(eddy_dir, "series.txt")
        with open(series_idx, "w") as f:
            f.write(" ".join(str(i) for i in series_vals) + "\n")

        # ------------------------------------------------------------------
        # 3) Build TOPUP b0 stack: 1 b0 per run, AP runs then PA runs
        # ------------------------------------------------------------------
        b0max = float(self.inputs.b0max)

        ap_b0_idxs = self._pick_first_b0_indices(ap_bvals, ap_run_lengths, b0max)
        pa_b0_idxs = self._pick_first_b0_indices(pa_bvals, pa_run_lengths, b0max)

        if len(ap_b0_idxs) != n_ap_runs or len(pa_b0_idxs) != n_pa_runs:
            raise RuntimeError(
                "Expected exactly one b0 per run for TOPUP.\n"
                f"AP runs: {n_ap_runs}, AP b0s found: {len(ap_b0_idxs)}; "
                f"PA runs: {n_pa_runs}, PA b0s found: {len(pa_b0_idxs)}."
            )

        # Extract one b0 per run into temporary files, then merge.
        tmp_b0_files: List[str] = []
        for idx in ap_b0_idxs:
            out = os.path.join(topup_dir, f"AP_b0_{idx}.nii.gz")
            _run(["fslroi", ap_file, out, "0", "-1", "0", "-1", "0", "-1", str(idx), "1"])
            tmp_b0_files.append(out)

        for idx in pa_b0_idxs:
            out = os.path.join(topup_dir, f"PA_b0_{idx}.nii.gz")
            _run(["fslroi", pa_file, out, "0", "-1", "0", "-1", "0", "-1", str(idx), "1"])
            tmp_b0_files.append(out)

        pos_neg_b0 = os.path.join(topup_dir, "Pos_Neg_b0.nii.gz")
        _run(["fslmerge", "-t", pos_neg_b0] + tmp_b0_files)

        # TOPUP acqparams: identical values to EDDY (but in topup/ directory)
        acqparams_topup = os.path.join(topup_dir, "acqparams.txt")
        shutil.copy2(acqparams_eddy, acqparams_topup)
        
        # Choose b02b0 config by divisibility of image dims
        fsldir = FSL
        if not fsldir:
            raise RuntimeError("FSL (FSLDIR) path is not set in utils.FSL.")

        fslval = os.path.join(fsldir, "bin", "fslval")
        d1 = int(subprocess.check_output([fslval, ap_file, "dim1"], text=True).split()[0])
        d2 = int(subprocess.check_output([fslval, ap_file, "dim2"], text=True).split()[0])
        d3 = int(subprocess.check_output([fslval, ap_file, "dim3"], text=True).split()[0])

        cfg_dir = os.path.join(fsldir, "etc", "flirtsch")
        if (d1 % 4 == 0) and (d2 % 4 == 0) and (d3 % 4 == 0):
            topup_config = os.path.join(cfg_dir, "b02b0_4.cnf")
        elif (d1 % 2 == 0) and (d2 % 2 == 0) and (d3 % 2 == 0):
            topup_config = os.path.join(cfg_dir, "b02b0_2.cnf")
        else:
            topup_config = os.path.join(cfg_dir, "b02b0_1.cnf")

        if not os.path.isfile(topup_config):
            raise RuntimeError(f"TOPUP config not found at {topup_config}")
        
        self._pos_neg = pos_neg
        self._pos_neg_b0 = pos_neg_b0
        self._acqparams_eddy = acqparams_eddy
        self._acqparams_topup = acqparams_topup
        self._idx_txt = idx_txt
        self._bvals_all = bvals_all_path
        self._bvecs_all = bvecs_all_path
        self._series_idx = series_idx
        self._topup_config = topup_config

        return runtime

    def _list_outputs(self):
        return {
            "pos_neg_b0": getattr(self, "_pos_neg_b0", None),
            "acqparams": getattr(self, "_acqparams_topup", None),
            "topup_config": getattr(self, "_topup_config", None),
            "pos_neg": getattr(self, "_pos_neg", None),
            "acqparams_eddy": getattr(self, "_acqparams_eddy", None),
            "idx_txt": getattr(self, "_idx_txt", None),
            "bvals_all": getattr(self, "_bvals_all", None),
            "bvecs_all": getattr(self, "_bvecs_all", None),
            "series_idx": getattr(self, "_series_idx", None),
        }


# ======================================================================
# 2. RunTopup
# ======================================================================

class RunTopupInputSpec(BaseInterfaceInputSpec):
    imain = File(exists=True, mandatory=True, desc="Pos_Neg_b0 4D for TOPUP")
    acqparams = File(exists=True, mandatory=True, desc="acqparams.txt for TOPUP")
    config = File(exists=True, mandatory=True, desc="TOPUP config file")

    outdir = Directory(mandatory=True, desc="output directory")
    out_base = traits.Str(mandatory=True, desc="base name for TOPUP outputs")

    bet4animal_z = traits.Int(usedefault=True, default_value=0, desc="if >0, use -Z in BET")


class RunTopupOutputSpec(TraitedSpec):
    fieldcoef = File(desc="TOPUP field coefficients")
    iout = File(desc="TOPUP corrected b0s")
    mask_file = File(desc="brain mask derived from TOPUP output")
    hifi_b0 = File(desc="HiFi b0 from applytopup (Pos_b01/Neg_b01)")


class RunTopup(BaseInterface):
    input_spec = RunTopupInputSpec
    output_spec = RunTopupOutputSpec

    def _run_interface(self, runtime):
        imain = os.path.abspath(self.inputs.imain)
        acqp = os.path.abspath(self.inputs.acqparams)
        config = os.path.abspath(self.inputs.config)
        outdir = os.path.abspath(self.inputs.outdir)
        _ensure_dir(outdir)

        out_base = os.path.abspath(self.inputs.out_base)

        # TOPUP
        fout = out_base + "_fout.nii.gz"
        iout = out_base + "_iout.nii.gz"

        _run_and_stream([
            "topup",
            "--imain=" + imain,
            "--datain=" + acqp,
            "--config=" + config,
            "--out=" + out_base,
            "--fout=" + fout,
            "--iout=" + iout,
        ])

        # Build mask from TOPUP corrected b0s
        nodif = os.path.join(outdir, "topup_nodif.nii.gz")
        _run(["fslmaths", iout, "-Tmean", nodif])

        mask_base = os.path.join(outdir, "topup_nodif_brain")
        bet_cmd = ["bet", nodif, mask_base, "-m", "-f", "0.3"]
        if int(self.inputs.bet4animal_z) > 0:
            bet_cmd.append("-Z")
        _run(bet_cmd)

        mask_file = mask_base + "_mask.nii.gz"

        # Save main TOPUP outputs
        self._fieldcoef = out_base + "_fieldcoef.nii.gz"
        self._iout = iout
        self._mask_file = mask_file

        # ------------------------------------------------------------------
        # HiFi b0 via applytopup
        # ------------------------------------------------------------------
        def _find_opposite_pair(acqp_path: str):
            rows = []
            with open(acqp_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    gx, gy, gz = map(int, parts[:3])
                    ro = float(parts[3])
                    rows.append((gx, gy, gz, ro))

            n = len(rows)
            for i in range(n):
                gx_i, gy_i, gz_i, ro_i = rows[i]
                for j in range(i + 1, n):
                    gx_j, gy_j, gz_j, ro_j = rows[j]
                    if (
                        gx_j == -gx_i
                        and gy_j == -gy_i
                        and gz_j == -gz_i
                        and abs(ro_j - ro_i) < 1e-6
                    ):
                        # return 1-based indices
                        return i + 1, j + 1

            if n >= 2:
                print(
                    "[RunTopup] Warning: could not find opposite PE pair in "
                    f"{acqp_path}; falling back to inindex=1,2"
                )
                return 1, 2
            else:
                raise RuntimeError(
                    f"acqparams file {acqp_path} has fewer than 2 valid rows."
                )

        pos_b01 = os.path.join(outdir, "Pos_b01.nii.gz")
        neg_b01 = os.path.join(outdir, "Neg_b01.nii.gz")
        hifi = os.path.join(outdir, "hifib0.nii.gz")

        fslroi = os.path.join(FSL, "bin", "fslroi")
        applytopup = os.path.join(FSL, "bin", "applytopup")

        # imain here is Pos_Neg_b0.nii.gz
        in_i, in_j = _find_opposite_pair(acqp)

        # acqparams rows are 1-based, volumes are 0-based
        _run([fslroi, imain, pos_b01, str(in_i - 1), "1"])
        _run([fslroi, imain, neg_b01, str(in_j - 1), "1"])

        _run([
            applytopup,
            f"--imain={pos_b01},{neg_b01}",
            f"--topup={out_base}",
            f"--datain={acqp}",
            f"--inindex={in_i},{in_j}",
            f"--out={hifi}",
        ])

        self._hifi_b0 = hifi
        return runtime

    def _list_outputs(self):
        return {
            "fieldcoef": getattr(self, "_fieldcoef", None),
            "iout": getattr(self, "_iout", None),
            "mask_file": getattr(self, "_mask_file", None),
            "hifi_b0": getattr(self, "_hifi_b0", None),
        }


# ======================================================================
# 3. RunEddy
# ======================================================================

class RunEddyInputSpec(BaseInterfaceInputSpec):
    imain = File(exists=True, mandatory=True, desc="Pos_Neg 4D image")
    mask = File(exists=True, mandatory=True, desc="brain mask")
    index = File(exists=True, mandatory=True, desc="index.txt")
    acqp = File(exists=True, mandatory=True, desc="acqparams.txt")
    bvecs = File(exists=True, mandatory=True, desc="bvecs")
    bvals = File(exists=True, mandatory=True, desc="bvals")
    topup_base = traits.Str(mandatory=True, desc="TOPUP base (same as out_base in RunTopup)")

    out = traits.Str(mandatory=True, desc="output base for eddy")
    extra_args = traits.Either(
        traits.List(traits.Str),
        traits.Str,
        usedefault=True,
        default_value=[],
        desc="extra eddy arguments",
    )
    session_file = File(exists=True, desc="per-volume series/session file", mandatory=False)


class RunEddyOutputSpec(TraitedSpec):
    out_file = File(desc="eddy corrected 4D image")


class RunEddy(BaseInterface):
    input_spec = RunEddyInputSpec
    output_spec = RunEddyOutputSpec

    def _run_interface(self, runtime):
        imain = os.path.abspath(self.inputs.imain)
        mask = os.path.abspath(self.inputs.mask)
        index = os.path.abspath(self.inputs.index)
        acqp = os.path.abspath(self.inputs.acqp)
        bvecs = os.path.abspath(self.inputs.bvecs)
        bvals = os.path.abspath(self.inputs.bvals)
        topup_base = os.path.abspath(self.inputs.topup_base)
        out = os.path.abspath(self.inputs.out)

        extra = self.inputs.extra_args
        if isinstance(extra, str):
            extra = extra.strip().split() if extra.strip() else []

        cmd = [
            "eddy",
            f"--imain={imain}",
            f"--mask={mask}",
            f"--index={index}",
            f"--acqp={acqp}",
            f"--bvecs={bvecs}",
            f"--bvals={bvals}",
            f"--topup={topup_base}",
            f"--out={out}",
            f"--cnr_maps",
            f"--niter=5",
            f"--fwhm=10,5,0,0,0",
        ] + list(extra)

        _run_and_stream(cmd)

        self._out_file = out + ".nii.gz"
        return runtime

    def _list_outputs(self):
        return {
            "out_file": getattr(self, "_out_file", None),
        }


# -------------------------
# 4. Post-EDDY combine
# -------------------------
class PostEddyCombineInputSpec(CommandLineInputSpec):
    eddy_dir = Directory(mandatory=True, desc="Eddy working directory")
    out_dir = Directory(mandatory=True, desc="Final data directory")
    pos_bval = File(exists=True, mandatory=True)
    pos_bvec = File(exists=True, mandatory=True)
    neg_bval = File(exists=True, mandatory=True)
    neg_bvec = File(exists=True, mandatory=True)
    combine_matched_flag = traits.Int(0, usedefault=True)
    eddy_out = File(exists=True, desc="Dependency on eddy output")


class PostEddyCombineOutputSpec(TraitedSpec):
    data_file = File()
    bval_file = File()
    bvec_file = File()
    mask_file = File()


class PostEddyCombine(CommandLine):
    _cmd = "bash -lc true"
    input_spec = PostEddyCombineInputSpec
    output_spec = PostEddyCombineOutputSpec

    def _list_outputs(self):
        od = os.path.abspath(self.inputs.out_dir)
        outs = self._outputs().get()
        outs["data_file"] = os.path.join(od, "data.nii.gz")
        outs["bval_file"] = os.path.join(od, "bvals")
        outs["bvec_file"] = os.path.join(od, "bvecs")
        outs["mask_file"] = os.path.join(od, "nodif_brain_mask.nii.gz")
        return outs

    def _run_interface(self, runtime):
        fslbin = os.path.join(FSL, "bin")
        eddy_dir = os.path.abspath(self.inputs.eddy_dir)
        out_dir = os.path.abspath(self.inputs.out_dir)
        os.makedirs(out_dir, exist_ok=True)

        data_out = os.path.join(out_dir, "data.nii.gz")
        bvals_out = os.path.join(out_dir, "bvals")
        bvecs_out = os.path.join(out_dir, "bvecs")

        def _count_bvals(path):
            with open(path, "r") as f:
                return len(f.read().split())

        eddy_unwarped = os.path.join(eddy_dir, "eddy_unwarped_images.nii.gz")
        posneg_bvals = os.path.join(eddy_dir, "Pos_Neg.bvals")
        posneg_bvecs = os.path.join(eddy_dir, "Pos_Neg.bvecs")

        eddy_combine_bin = os.path.join(fslbin, "eddy_combine")
        onlymatched = 1 if int(self.inputs.combine_matched_flag) == 1 else 0

        if onlymatched == 0:
            subprocess.run([os.path.join(fslbin, "imcp"), eddy_unwarped, data_out], check=True)
            shutil.copy2(posneg_bvals, bvals_out)
            shutil.copy2(posneg_bvecs, bvecs_out)
        else:
            pos_vols = _count_bvals(self.inputs.pos_bval)
            neg_vols = _count_bvals(self.inputs.neg_bval)
            pos_nii = os.path.join(eddy_dir, "eddy_unwarped_Pos.nii.gz")
            neg_nii = os.path.join(eddy_dir, "eddy_unwarped_Neg.nii.gz")
            subprocess.run([os.path.join(fslbin, "fslroi"), eddy_unwarped, pos_nii, "0", str(pos_vols)], check=True)
            subprocess.run([os.path.join(fslbin, "fslroi"), eddy_unwarped, neg_nii, str(pos_vols), str(neg_vols)], check=True)
            corr = str(min(pos_vols, neg_vols))
            pos_series = os.path.join(eddy_dir, "Pos_SeriesVolNum.txt")
            neg_series = os.path.join(eddy_dir, "Neg_SeriesVolNum.txt")
            with open(pos_series, "w") as f:
                f.write(f"{corr} {pos_vols}\n")
            with open(neg_series, "w") as f:
                f.write(f"{corr} {neg_vols}\n")
            subprocess.run([
                eddy_combine_bin,
                pos_nii, self.inputs.pos_bval, self.inputs.pos_bvec, pos_series,
                neg_nii, self.inputs.neg_bval, self.inputs.neg_bvec, neg_series,
                out_dir, str(onlymatched)
            ], check=True)
            subprocess.run([os.path.join(fslbin, "imrm"), pos_nii], check=True)
            subprocess.run([os.path.join(fslbin, "imrm"), neg_nii], check=True)

        subprocess.run(
            [os.path.join(fslbin, "fslmaths"), data_out, "-thr", "0", data_out],
            check=True,
        )

        runtime.returncode = 0
        return runtime


# ======================================================================
# 5. RunEddyQC
# ======================================================================

class RunEddyQCInputSpec(BaseInterfaceInputSpec):
    eddy_base = traits.Str(mandatory=True, desc="eddy out base (same as RunEddy.out)")
    outdir = Directory(mandatory=True, desc="output directory for QC")

    mask = File(exists=True, mandatory=True, desc="brain mask")
    acqp = File(exists=True, mandatory=True, desc="acqparams.txt")
    index = File(exists=True, mandatory=True, desc="index.txt")
    bvals = File(exists=True, mandatory=True, desc="bvals")
    bvecs = File(exists=True, mandatory=True, desc="bvecs")


class RunEddyQCOutputSpec(TraitedSpec):
    qc_json = File(desc="eddy QC summary JSON (eddy.qc/qc.json)")


class RunEddyQC(BaseInterface):
    input_spec = RunEddyQCInputSpec
    output_spec = RunEddyQCOutputSpec

    def _run_interface(self, runtime):
        eddy_base = os.path.abspath(self.inputs.eddy_base)
        outdir = os.path.abspath(self.inputs.outdir)
        _ensure_dir(outdir)

        qc_dir = os.path.join(outdir, "eddy.qc")
        mask = os.path.abspath(self.inputs.mask)
        acqp = os.path.abspath(self.inputs.acqp)
        index = os.path.abspath(self.inputs.index)
        bvals = os.path.abspath(self.inputs.bvals)
        bvecs = os.path.abspath(self.inputs.bvecs)

        cmd = [
            "eddy_quad",
            eddy_base,
            "-idx", index,
            "-par", acqp,
            "-m", mask,
            "-b", bvals,
            "-g", bvecs,
            "-o", qc_dir,
        ]
        _run(cmd)

        qc_json = os.path.join(qc_dir, "qc.json")
        if not os.path.isfile(qc_json):
            raise RuntimeError(f"Expected QC JSON at {qc_json} but it was not created.")

        self._qc_json = qc_json
        return runtime

    def _list_outputs(self):
        return {
            "qc_json": getattr(self, "_qc_json", None),
        }
