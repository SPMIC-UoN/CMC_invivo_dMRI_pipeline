# src/invivo_dmri_pipeline/pipeline.py
import os
import re
import glob
import shutil
import subprocess

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
from nipype.interfaces.utility import Function
from nipype import config as nipyconfig, logging as nplog

from .config import Config
from .utils import FSL, require_fsl, pkg_file

from .nodes import (
    # denoise
    BuildPairs, MakeDenoiseArgs, Denoise, CopySidecars,
    # combine, gibbs / N4
    Combine, GibbsN4,
    # drift (temporal)
    Drift, CopyPair,
    # topup / eddy prep + runners
    PrepareTopupEddy, RunTopup, RunEddy, PostEddyCombine, RunEddyQC,
    # skullstrip
    Skullstrip,
    # DTI + registration + tractography
    SelectLowerShell, DTIFIT, Reg2Std, Bedpostx, Xtract, QAReport
)

VALID_TAGS = ("AP", "PA", "LR", "RL")


def _detect_tag_from_name(basename: str) -> str:
    m = re.match(r"^(AP|PA|LR|RL)", basename)
    return m.group(1) if m else ""


def _choose_scheme_and_tags(root: str):
    ap = glob.glob(os.path.join(root, "AP", "*.nii*"))
    pa = glob.glob(os.path.join(root, "PA", "*.nii*"))
    lr = glob.glob(os.path.join(root, "LR", "*.nii*"))
    rl = glob.glob(os.path.join(root, "RL", "*.nii*"))
    found_ap_pa = len(ap) + len(pa) > 0
    found_lr_rl = len(lr) + len(rl) > 0
    if found_ap_pa and found_lr_rl:
        raise RuntimeError("Mixed AP/PA and LR/RL layouts present under input root.")
    if found_ap_pa:
        return ("AP", "PA")
    if found_lr_rl:
        return ("LR", "RL")
    raise RuntimeError("No tag subfolders found. Expect AP/PA or LR/RL (and *_ph).")


def _require_sidecars(dirpath: str, tag: str):
    missing = []
    for nii in sorted(glob.glob(os.path.join(dirpath, tag, f"{tag}_*.nii*"))):
        stem = (
            os.path.splitext(os.path.splitext(nii)[0])[0]
            if nii.endswith(".nii.gz")
            else os.path.splitext(nii)[0]
        )
        bvec = stem + ".bvec"
        bval = stem + ".bval"
        if not os.path.isfile(bvec) or not os.path.isfile(bval):
            missing.append(os.path.basename(stem))
    if missing:
        raise RuntimeError(f"Missing sidecars for MAG files in {tag}/: {missing}")


def _check_phase_counts(root: str, tag: str):
    mags = sorted(glob.glob(os.path.join(root, tag, f"{tag}_*.nii*")))
    ph_dir = f"{tag}_ph"
    phases = sorted(glob.glob(os.path.join(root, ph_dir, f"{ph_dir}_*.nii*")))
    if len(phases) > len(mags):
        raise RuntimeError(f"More PHASE than MAG in {tag}: {len(phases)} > {len(mags)}")

    def _idxs(paths, prefix):
        out = []
        for p in paths:
            m = re.search(rf"{re.escape(prefix)}_(\d+)\.nii(\.gz)?$", os.path.basename(p))
            out.append(int(m.group(1)) if m else -1)
        return set(out)

    if phases and not (_idxs(phases, f"{tag}_ph") <= _idxs(mags, tag)):
        raise RuntimeError(f"PHASE indices in {ph_dir}/ are not a subset of MAG indices in {tag}/")

def get_run_lengths(files):
    """
    Return a list with dim4 (number of volumes) for each file in `files`.
    """
    import os
    import subprocess
    from invivo_dmri_pipeline.utils import FSL

    if not isinstance(files, (list, tuple)):
        files = [files]

    files = [f for f in files if f]
    out = []
    fslval = os.path.join(FSL, "bin", "fslval")
    for f in files:
        v = (
            subprocess.check_output([fslval, f, "dim4"], text=True)
            .strip()
            .split()[0]
        )
        out.append(int(v))
    return out


def build_workflow(
    cfg: Config,
    verbose: bool = False,
    run_mode: str = "preproc_only",
) -> pe.Workflow:
    """
    Build the Nipype workflow.

    Parameters
    ----------
    cfg : Config
        Parsed YAML configuration.
    verbose : bool
        Verbosity flag.
    run_mode : {"preproc_only", "full", "tract_only"}
        - "preproc_only": Run preprocessing up to DTIFIT + registration + QA.
                          No BedpostX or XTRACT.
        - "full": Run preprocessing + BedpostX + XTRACT in a single workflow.
        - "tract_only": Assume preprocessing+Reg2Std already run; run BedpostX
                        (if missing) + XTRACT + QA.
    """
    if run_mode not in ("preproc_only", "full", "tract_only"):
        raise RuntimeError(
            f"Invalid run_mode='{run_mode}'. "
            "Expected 'preproc_only', 'full', or 'tract_only'."
        )

    # -------- Nipype verbosity --------
    if verbose:
        interface_level = "DEBUG"
        workflow_level = "INFO"
        monitoring_on = True
        poll_sleep = 0.1
    else:
        interface_level = "WARNING"
        workflow_level = "WARNING"
        monitoring_on = False
        poll_sleep = 0.25

    nipyconfig.update_config({
        "execution": {
            "stop_on_first_crash": True,
            "hash_method": "content",
            "remove_unnecessary_outputs": False,
            "poll_sleep_duration": poll_sleep,
        },
        "logging": {"interface_level": interface_level, "workflow_level": workflow_level},
        "monitoring": {"enabled": monitoring_on},
    })
    nplog.update_logging(nipyconfig)

    require_fsl()

    # -------- Mode (nhp|hum) + constants from cfg --------
    MODE = cfg.MODE  # 'nhp' or 'hum'
    print(f"[pipeline] MODE = {MODE}")
    print(f"[pipeline] run_mode = {run_mode}")

    B0RANGE = float(cfg.B0RANGE)
    ECHO_MS = float(cfg.ECHO_MS)
    PIFACTOR = int(cfg.PIFACTOR)
    LOWER_B = int(cfg.LOWER_B)
    COMBINE_MATCHED_FLAG = int(cfg.COMBINE_MATCHED_FLAG)
    PTX_STEPLENGTH = float(cfg.PTX_STEPLENGTH)
    DENOISE = bool(getattr(cfg, "DENOISE", True))

    # Optional T1/brainmask for skullstrip
    T1_PATH = cfg.t1 if getattr(cfg, "t1", None) else None
    BRAIN_MASK = cfg.brain_mask if getattr(cfg, "brain_mask", None) else None

    # -------- Packaged resources (NHP) + FSL stdrefs (HUM) --------
    STD_NMT = pkg_file("files", "MACAQUE_NMT.nii.gz")
    NMT_FA = pkg_file("files", "NMT_FA.nii.gz")
    NMT_TENSOR = pkg_file("files", "NMT_tensor.nii.gz")
    MMORF_CFG = pkg_file("files", "mmorf_config_template.ini")

    FSLDIR = os.environ["FSLDIR"]
    FSL_STD = os.path.join(FSLDIR, "data", "standard")
    MNI_T1_1MM = os.path.join(FSL_STD, "MNI152_T1_1mm.nii.gz")
    HCP1065_FA_1MM = os.path.join(FSL_STD, "FSL_HCP1065_FA_1mm.nii.gz")
    HCP1065_TENSOR_1MM = os.path.join(FSL_STD, "FSL_HCP1065_tensor_1mm.nii.gz")

    # Helper scripts shipped with the package
    combine_py = pkg_file("nodes", "invivo_combine.py")
    bvals_round_py = pkg_file("nodes", "bvals_round.py")
    drift_py = pkg_file("nodes", "drift.py")
    reg2std_py = pkg_file("nodes", "reg2std.py")
    
    # External denoise wrapper (required only if denoising is enabled)
    if DENOISE:
        if not cfg.abs_denoise_sh or not os.path.isfile(cfg.abs_denoise_sh):
            raise RuntimeError(
                "Config key 'denoise_sh' must be set to the NORDIC/denoise "
                "wrapper script path when DENOISE is true."
            )

    # -------- Workspace paths --------
    dmri_root = cfg.abs_dmri_root
    input_root = (
        os.path.abspath(cfg.input_root)
        if getattr(cfg, "input_root", None)
        else (
            os.path.abspath(os.path.commonpath(cfg.mag_files))
            if getattr(cfg, "mag_files", None)
            else dmri_root
        )
    )
    input_root = os.path.abspath(input_root)

    combined_dir = os.path.join(dmri_root, "data_combined")
    combined_gibbs_dir = os.path.join(dmri_root, "data_combined_gibbs")
    combined_n4_dir = os.path.join(dmri_root, "data_combined_gibbs_N4")
    combined_drift_dir = os.path.join(dmri_root, "data_combined_drift")
    topup_dir_root = os.path.join(dmri_root, "topup")
    eddy_dir_root = os.path.join(dmri_root, "eddy")
    data_dir = os.path.join(dmri_root, "data")
    for d in (
        combined_dir,
        combined_gibbs_dir,
        combined_n4_dir,
        combined_drift_dir,
        topup_dir_root,
        eddy_dir_root,
        data_dir,
    ):
        os.makedirs(d, exist_ok=True)

    # -------- Copy the input YAML into the processing root --------
    if getattr(cfg, "abs_yaml_path", None) and os.path.isfile(cfg.abs_yaml_path):
        dst_yaml = os.path.join(dmri_root, "invivo_dmri_config.yaml")
        try:
            shutil.copy2(cfg.abs_yaml_path, dst_yaml)
            print(f"[pipeline] Copied config to {dst_yaml}")
        except Exception as e:
            print(
                f"[pipeline] Warning: could not copy config file ({cfg.abs_yaml_path}): {e}"
            )

    # -------- Build workflow container --------
    # Use a different logging dir for tract-only so we do not clash with preproc
    wf_name = "nipype_logging" if run_mode in ("preproc_only", "full") else "nipype_logging_tract"
    wf = pe.Workflow(name=wf_name)
    wf.base_dir = dmri_root

    # ------------------------------------------------------------------
    # TRACT-ONLY MODE: run BedpostX if missing, then XTRACT + QA
    # ------------------------------------------------------------------
    if run_mode == "tract_only":
        print("[pipeline] Building tract-only workflow (BedpostX → XTRACT → QA).")

        # --------------------------------------------------------------
        # Minimal required preprocessing outputs
        # --------------------------------------------------------------
        data_dir = os.path.join(dmri_root, "data")
        data_4d = os.path.join(data_dir, "data.nii.gz")
        bvals = os.path.join(data_dir, "bvals")
        bvecs = os.path.join(data_dir, "bvecs")
        mask = os.path.join(data_dir, "nodif_brain_mask.nii.gz")

        # DTIFIT + stdreg outputs from the preproc run
        fa_native = os.path.join(dmri_root, "dtifit", "dti_FA.nii.gz")
        v1_native = os.path.join(dmri_root, "dtifit", "dti_V1.nii.gz")
        fa_std = os.path.join(dmri_root, "stdreg", "FA_in_STD.nii.gz")

        required = [data_4d, bvals, bvecs, mask, fa_native, v1_native, fa_std]
        missing = [p for p in required if not os.path.isfile(p)]
        if missing:
            raise RuntimeError(
                "Tract-only mode requested but required preprocessing outputs "
                "are missing.\n"
                "Missing:\n  - " + "\n  - ".join(missing) +
                "\nRun the pipeline once without --tractography first."
            )

        # --------------------------------------------------------------
        # Required registration warps
        # --------------------------------------------------------------
        stdreg_dir = os.path.join(dmri_root, "stdreg")
        warp_std2anat = os.path.join(stdreg_dir, "stdreg_std_to_anat_warp.nii.gz")
        warp_anat2std = os.path.join(stdreg_dir, "stdreg_anat_to_std_warp.nii.gz")

        if not os.path.isfile(warp_std2anat) or not os.path.isfile(warp_anat2std):
            raise RuntimeError(
                "Tractography requested but registration warps not found.\n"
                "Run preprocessing once first."
            )

        # --------------------------------------------------------------
        # BedpostX directory + check for output
        # --------------------------------------------------------------
        bpx_dir = data_dir + ".bedpostX"
        f1 = os.path.join(bpx_dir, "merged_f1samples.nii.gz")
        need_bpx = not os.path.isfile(f1)

        # --------------------------------------------------------------
        # Run BedpostX if needed
        # --------------------------------------------------------------
        if need_bpx:
            print("[pipeline] No BedpostX outputs found → running BedpostX now.")

            bpx = pe.Node(Bedpostx(), name="bedpostx")
            bpx.inputs.datadir = data_dir
            bpx.inputs.use_gpu = not bool(getattr(cfg, "NO_GPU", False))

            wf.add_nodes([bpx])
            bpx_outnode = bpx
        else:
            print("[pipeline] BedpostX outputs found → skipping BedpostX.")
            bpx_outnode = None

        # --------------------------------------------------------------
        # XTRACT
        # --------------------------------------------------------------
        xtract = pe.Node(Xtract(), name="xtract")
        xtract_outdir = os.path.join(dmri_root, "xtract")
        xtract.inputs.outdir = xtract_outdir
        if need_bpx:
            wf.connect(bpx_outnode, "bpx_dir", xtract, "bpx_dir")
        else:
            xtract.inputs.bpx_dir = bpx_dir

        # species / refs
        if MODE == "nhp":
            xtract.inputs.species = "MACAQUE"
            xtract.inputs.stdref = STD_NMT
        else:
            xtract.inputs.species = "HUMAN"
            xtract.inputs.stdref = MNI_T1_1MM

        # optional profiles
        if getattr(cfg, "abs_xtract_profiles_dir", None):
            xtract.inputs.profiles_dir = cfg.abs_xtract_profiles_dir

        # warps
        xtract.inputs.warp_std2anat = warp_std2anat
        xtract.inputs.warp_anat2std = warp_anat2std

        # streamlines options
        xtract.inputs.do_streamlines = bool(getattr(cfg, "STREAMLINES_DO", True))
        xtract.inputs.streamlines_density_threshold = float(
            getattr(cfg, "STREAMLINES_DENSITY_THRESHOLD", 1e-3)
        )
        xtract.inputs.streamlines_format = getattr(
            cfg, "STREAMLINES_FORMAT", "trk"
        )
        xtract.inputs.streamlines_ptx2_prefix = getattr(
            cfg, "STREAMLINES_PTX2_PREFIX", "densityNorm"
        )
        xtract.inputs.streamlines_num_jobs = int(
            getattr(cfg, "STREAMLINES_NUM_JOBS", 1)
        )

        # viewer
        xtract.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract.inputs.do_viewer = bool(getattr(cfg, "DO_VIEWER", True))

        # --------------------------------------------------------------
        # QA (tractography-only report)
        # --------------------------------------------------------------
        qa = pe.Node(QAReport(), name="qa_report")
        qa.inputs.dmri_root = dmri_root
        qa.inputs.outdir = os.path.join(dmri_root, "QA")
        qa.inputs.pipeline_yaml = os.path.join(dmri_root, "invivo_dmri_config.yaml")

        qa.inputs.run_mode = run_mode  # "tract_only"

        # Nipype dirs / graphs: preproc from main run, tract from this run
        qa.inputs.nipype_dir_preproc = os.path.join(dmri_root, "nipype_logging")
        qa.inputs.nipype_dir_tract = os.path.join(dmri_root, wf_name)
        qa.inputs.nipype_graph_preproc = os.path.join(
            dmri_root, "nipype_logging", "graph.png"
        )
        qa.inputs.nipype_graph_tract = os.path.join(
            dmri_root, wf_name, "graph.png"
        )

        # Eddy QC from original preproc
        qa.inputs.eddy_qc_json = os.path.join(
            eddy_dir_root, "eddy.qc", "qc.json"
        )

        qa.inputs.nmt_mask_img = (
            pkg_file("files", "NMT_v2.0_sym_05mm_LR_brainmask.nii.gz")
            if MODE == "nhp"
            else os.path.join(
                FSLDIR, "data", "standard", "MNI152_T1_1mm_brain_mask.nii.gz"
            )
        )

        # Core diffusion inputs
        qa.inputs.data_4d = data_4d
        qa.inputs.bvals = bvals
        qa.inputs.bvecs = bvecs
        qa.inputs.mask_file = mask

        # Native FA / V1
        qa.inputs.fa_img = fa_native
        qa.inputs.v1_img = v1_native

        # XTRACT directory
        qa.inputs.xtract_dir = xtract_outdir

        # Run-last dummy
        qa.inputs.deps = []

        # Ensure QA sees the final XTRACT dir path (and is recomputed if changed)
        wf.connect(xtract, "outdir", qa, "xtract_dir")

        return wf

    # ------------------------------------------------------------------
    # PREPROC / FULL MODES: full pipeline (with optional bpx and XTRACT)
    # ------------------------------------------------------------------
    # Step 1: Validate inputs
    TAG_POS, TAG_NEG = _choose_scheme_and_tags(input_root)
    _require_sidecars(input_root, TAG_POS)
    _require_sidecars(input_root, TAG_NEG)
    _check_phase_counts(input_root, TAG_POS)
    _check_phase_counts(input_root, TAG_NEG)

    # ------------------------------------------------------------------
    # STEP 2: Denoising planning + calls
    # ------------------------------------------------------------------
    pairs = pe.Node(BuildPairs(), name="pairs")
    pairs.inputs.root = input_root

    ap_args = pe.Node(MakeDenoiseArgs(direction=TAG_POS), name="ap_args")
    ap_args.inputs.root = dmri_root
    pa_args = pe.Node(MakeDenoiseArgs(direction=TAG_NEG), name="pa_args")
    pa_args.inputs.root = dmri_root

    wf.connect(pairs, "ap_mag", ap_args, "mags")
    wf.connect(pairs, "ap_ph", ap_args, "phases")
    wf.connect(pairs, "pa_mag", pa_args, "mags")
    wf.connect(pairs, "pa_ph", pa_args, "phases")

    # These will be the inputs to Combine
    if DENOISE:
        # Run denoising
        ap_dn = pe.MapNode(
            Denoise(),
            name="ap_dn",
            iterfield=["domain", "magn", "phase", "name", "oPath"],
        )
        pa_dn = pe.MapNode(
            Denoise(),
            name="pa_dn",
            iterfield=["domain", "magn", "phase", "name", "oPath"],
        )
        ap_dn.inputs.denoise_sh = cfg.abs_denoise_sh
        pa_dn.inputs.denoise_sh = cfg.abs_denoise_sh

        wf.connect(ap_args, "domains", ap_dn, "domain")
        wf.connect(ap_args, "magn", ap_dn, "magn")
        wf.connect(ap_args, "phase", ap_dn, "phase")
        wf.connect(ap_args, "names", ap_dn, "name")
        wf.connect(ap_args, "outdirs", ap_dn, "oPath")

        wf.connect(pa_args, "domains", pa_dn, "domain")
        wf.connect(pa_args, "magn", pa_dn, "magn")
        wf.connect(pa_args, "phase", pa_dn, "phase")
        wf.connect(pa_args, "names", pa_dn, "name")
        wf.connect(pa_args, "outdirs", pa_dn, "oPath")

        pre_pos_src = (ap_dn, "out_nii")
        pre_neg_src = (pa_dn, "out_nii")
    else:
        pre_pos_src = (pairs, "ap_mag")
        pre_neg_src = (pairs, "pa_mag")

    indat_pos_src = pre_pos_src
    indat_neg_src = pre_neg_src

    ap_cp = pe.Node(CopySidecars(), name="ap_cp")
    pa_cp = pe.Node(CopySidecars(), name="pa_cp")
    wf.connect(ap_args, "magn", ap_cp, "magn")
    wf.connect(ap_args, "names", ap_cp, "names")
    wf.connect(pa_args, "magn", pa_cp, "magn")
    wf.connect(pa_args, "names", pa_cp, "names")

    ap_outdir = os.path.join(dmri_root, f"{TAG_POS}_denoised")
    pa_outdir = os.path.join(dmri_root, f"{TAG_NEG}_denoised")
    os.makedirs(ap_outdir, exist_ok=True)
    os.makedirs(pa_outdir, exist_ok=True)
    ap_cp.inputs.outdir = ap_outdir
    pa_cp.inputs.outdir = pa_outdir

    # ------------------------------------------------------------------
    # STEP 4: Combine runs
    # ------------------------------------------------------------------
    combine_pos = pe.Node(Combine(), name="combine_pos")
    combine_neg = pe.Node(Combine(), name="combine_neg")
    combine_pos.inputs.script = combine_py
    combine_neg.inputs.script = combine_py
    combine_pos.inputs.outprefix = os.path.join(combined_dir, f"{TAG_POS}_combined")
    combine_neg.inputs.outprefix = os.path.join(combined_dir, f"{TAG_NEG}_combined")
    combine_pos.inputs.b0range = B0RANGE
    combine_pos.inputs.bvals_round_py = bvals_round_py
    combine_neg.inputs.b0range = B0RANGE
    combine_neg.inputs.bvals_round_py = bvals_round_py

    wf.connect(*indat_pos_src, combine_pos, "indat")
    wf.connect(*indat_neg_src, combine_neg, "indat")

    # make Combine wait for sidecar copies via a Merge node
    deps_combine_pos = pe.Node(util.Merge(2), name="deps_combine_pos")
    deps_combine_neg = pe.Node(util.Merge(2), name="deps_combine_neg")

    wf.connect(ap_cp, "out_bvecs", deps_combine_pos, "in1")
    wf.connect(ap_cp, "out_bvals", deps_combine_pos, "in2")
    wf.connect(deps_combine_pos, "out", combine_pos, "deps")

    wf.connect(pa_cp, "out_bvecs", deps_combine_neg, "in1")
    wf.connect(pa_cp, "out_bvals", deps_combine_neg, "in2")
    wf.connect(deps_combine_neg, "out", combine_neg, "deps")

    # ------------------------------------------------------------------
    # STEP 4.5: Gibbs + N4 (optional)
    # ------------------------------------------------------------------
    deps_after_combine = pe.Node(util.Merge(2), name="deps_after_combine")
    wf.connect(combine_pos, "out_file", deps_after_combine, "in1")
    wf.connect(combine_neg, "out_file", deps_after_combine, "in2")

    gibbs_n4 = pe.Node(GibbsN4(), name="gibbs_n4")
    gibbs_n4.inputs.combined_dir = combined_dir
    gibbs_n4.inputs.outdir_gibbs = combined_gibbs_dir
    gibbs_n4.inputs.outdir_n4 = combined_n4_dir

    # Config-driven toggles
    gibbs_n4.inputs.run_gibbs = getattr(cfg, "run_gibbs", True)
    gibbs_n4.inputs.run_n4 = getattr(cfg, "run_n4", True)
    gibbs_n4.inputs.use_docker = getattr(cfg, "use_docker", True)
    gibbs_n4.inputs.container_runtime = getattr(cfg, "container_runtime", "docker")
    # gibbs_n4.inputs.gibbs_image = cfg.gibbs_image
    # gibbs_n4.inputs.n4_image = cfg.n4_image
    gibbs_n4.inputs.docker_image = cfg.docker_image
    if hasattr(gibbs_n4.inputs, "interactive_tty"):
        gibbs_n4.inputs.interactive_tty = getattr(cfg, "interactive_tty", True)
    
    deps_gibbs = pe.Node(util.Merge(2), name="deps_gibbs")
    wf.connect(combine_pos, "out_file", deps_gibbs, "in1")
    wf.connect(combine_neg, "out_file", deps_gibbs, "in2")
    wf.connect(deps_gibbs, "out", gibbs_n4, "deps")

    # ------------------------------------------------------------------
    # STEP 5: Drift correction
    # ------------------------------------------------------------------
    merge_inputs = pe.Node(util.Merge(2), name="merge_inputs")
    merge_bvals = pe.Node(util.Merge(2), name="merge_bvals")

    # Decide which images to feed into drift:
    #  - If N4 is enabled: use N4 outputs
    #  - Else if Gibbs is enabled: use mrdegibbs outputs
    #  - Else: fall back to combined images directly
    if getattr(cfg, "run_n4", True):
        # Use N4-corrected images
        pick_pos_n4 = pe.Node(util.Select(index=0), name="pick_pos_n4")
        pick_neg_n4 = pe.Node(util.Select(index=1), name="pick_neg_n4")
        wf.connect(gibbs_n4, "n4_files", pick_pos_n4, "inlist")
        wf.connect(gibbs_n4, "n4_files", pick_neg_n4, "inlist")
        wf.connect(pick_pos_n4, "out", merge_inputs, "in1")
        wf.connect(pick_neg_n4, "out", merge_inputs, "in2")

    elif getattr(cfg, "run_gibbs", True):
        # Use mrdegibbs outputs
        pick_pos_gibbs = pe.Node(util.Select(index=0), name="pick_pos_gibbs")
        pick_neg_gibbs = pe.Node(util.Select(index=1), name="pick_neg_gibbs")
        wf.connect(gibbs_n4, "gibbs_files", pick_pos_gibbs, "inlist")
        wf.connect(gibbs_n4, "gibbs_files", pick_neg_gibbs, "inlist")
        wf.connect(pick_pos_gibbs, "out", merge_inputs, "in1")
        wf.connect(pick_neg_gibbs, "out", merge_inputs, "in2")

    else:
        # No Gibbs, no N4: use the raw combined images
        wf.connect(combine_pos, "out_file", merge_inputs, "in1")
        wf.connect(combine_neg, "out_file", merge_inputs, "in2")

    # bvals are always taken from the combined outputs
    wf.connect(combine_pos, "out_bval", merge_bvals, "in1")
    wf.connect(combine_neg, "out_bval", merge_bvals, "in2")

    drift = pe.Node(Drift(), name="drift")
    drift.inputs.script = drift_py
    drift.inputs.model = "single"
    drift.inputs.output = [
        f"{TAG_POS}_driftcorr.nii.gz",
        f"{TAG_NEG}_driftcorr.nii.gz",
    ]
    drift.inputs.outdir = combined_drift_dir
    wf.connect(merge_inputs, "out", drift, "input")
    wf.connect(merge_bvals, "out", drift, "bvals")

    copy_pos_corr = pe.Node(CopyPair(), name="copy_pos_corr")
    copy_neg_corr = pe.Node(CopyPair(), name="copy_neg_corr")

    copy_pos_corr.inputs.dst_prefix = os.path.join(
        combined_drift_dir, f"{TAG_POS}_driftcorr"
    )
    copy_neg_corr.inputs.dst_prefix = os.path.join(
        combined_drift_dir, f"{TAG_NEG}_driftcorr"
    )
    
    wf.connect(combine_pos, "out_bval", copy_pos_corr, "src_bval")
    wf.connect(combine_pos, "out_bvec", copy_pos_corr, "src_bvec")
    wf.connect(combine_neg, "out_bval", copy_neg_corr, "src_bval")
    wf.connect(combine_neg, "out_bvec", copy_neg_corr, "src_bvec")
    wf.connect(drift, "out_files", copy_pos_corr, "deps")
    wf.connect(drift, "out_files", copy_neg_corr, "deps")

    # --- Per-run lengths for session-aware EDDY ---
    ap_run_lengths = pe.Node(
        Function(
            input_names=["files"],
            output_names=["lengths"],
            function=get_run_lengths,
        ),
        name="ap_run_lengths",
    )
    pa_run_lengths = pe.Node(
        Function(
            input_names=["files"],
            output_names=["lengths"],
            function=get_run_lengths,
        ),
        name="pa_run_lengths",
    )
    wf.connect(*indat_pos_src, ap_run_lengths, "files")
    wf.connect(*indat_neg_src, pa_run_lengths, "files")

    # ------------------------------------------------------------------
    # STEP 6: Prepare TOPUP/EDDY inputs
    # ------------------------------------------------------------------
    pick_pos_drift = pe.Node(util.Select(index=0), name="pick_pos_drift")
    pick_neg_drift = pe.Node(util.Select(index=1), name="pick_neg_drift")
    wf.connect(drift, "out_files", pick_pos_drift, "inlist")
    wf.connect(drift, "out_files", pick_neg_drift, "inlist")

    prep_te = pe.Node(PrepareTopupEddy(), name="prep_topup_eddy")
    prep_te.inputs.out_root = dmri_root
    prep_te.inputs.pedir_axis = "auto"
    prep_te.inputs.echo_ms = ECHO_MS
    prep_te.inputs.pifactor = PIFACTOR
    prep_te.inputs.b0max = B0RANGE

    wf.connect(pick_pos_drift, "out", prep_te, "ap_file")
    wf.connect(pick_neg_drift, "out", prep_te, "pa_file")
    wf.connect(copy_pos_corr, "out_bval", prep_te, "ap_bval")
    wf.connect(copy_pos_corr, "out_bvec", prep_te, "ap_bvec")
    wf.connect(copy_neg_corr, "out_bval", prep_te, "pa_bval")
    wf.connect(copy_neg_corr, "out_bvec", prep_te, "pa_bvec")
    wf.connect(ap_run_lengths, "lengths", prep_te, "ap_run_lengths")
    wf.connect(pa_run_lengths, "lengths", prep_te, "pa_run_lengths")

    # ------------------------------------------------------------------
    # STEP 7: Run TOPUP
    # ------------------------------------------------------------------
    run_topup = pe.Node(RunTopup(), name="run_topup")
    run_topup.inputs.outdir = topup_dir_root
    run_topup.inputs.out_base = os.path.join(topup_dir_root, "topup_Pos_Neg_b0")
    run_topup.inputs.bet4animal_z = int(cfg.bet4animal_z)  # 2 for nhp, 0 for hum
    wf.connect(prep_te, "pos_neg_b0", run_topup, "imain")
    wf.connect(prep_te, "acqparams", run_topup, "acqparams")
    wf.connect(prep_te, "topup_config", run_topup, "config")

    # ------------------------------------------------------------------
    # STEP 8: Run EDDY
    # ------------------------------------------------------------------
    eddy = pe.Node(RunEddy(), name="eddy")
    eddy.inputs.topup_base = os.path.join(topup_dir_root, "topup_Pos_Neg_b0")
    eddy.inputs.out = os.path.join(eddy_dir_root, "eddy_unwarped_images")

    extra = getattr(cfg, "eddy_extra_args", [])
    if isinstance(extra, str):
        extra = extra.strip()
    eddy.inputs.extra_args = extra

    wf.connect(prep_te, "pos_neg", eddy, "imain")
    wf.connect(run_topup, "mask_file", eddy, "mask")
    wf.connect(prep_te, "idx_txt", eddy, "index")
    wf.connect(prep_te, "acqparams_eddy", eddy, "acqp")
    wf.connect(prep_te, "bvecs_all", eddy, "bvecs")
    wf.connect(prep_te, "bvals_all", eddy, "bvals")
    wf.connect(prep_te, "series_idx", eddy, "session_file")

    # Run eddy_quad to generate eddy QC
    eddy_qc = pe.Node(RunEddyQC(), name="eddy_qc")
    eddy_qc.inputs.outdir = eddy_dir_root
    eddy_qc.inputs.eddy_base = os.path.join(
        eddy_dir_root, "eddy_unwarped_images"
    )
    wf.connect(run_topup, "mask_file", eddy_qc, "mask")
    wf.connect(prep_te, "acqparams_eddy", eddy_qc, "acqp")
    wf.connect(prep_te, "idx_txt", eddy_qc, "index")
    wf.connect(prep_te, "bvals_all", eddy_qc, "bvals")
    wf.connect(prep_te, "bvecs_all", eddy_qc, "bvecs")

    # ------------------------------------------------------------------
    # STEP 8.5: Post-EDDY combine to data/
    # ------------------------------------------------------------------
    post_eddy = pe.Node(PostEddyCombine(), name="post_eddy")
    post_eddy.inputs.eddy_dir = eddy_dir_root
    post_eddy.inputs.out_dir = data_dir
    post_eddy.inputs.combine_matched_flag = COMBINE_MATCHED_FLAG
    wf.connect(eddy, "out_file", post_eddy, "eddy_out")
    wf.connect(copy_pos_corr, "out_bval", post_eddy, "pos_bval")
    wf.connect(copy_pos_corr, "out_bvec", post_eddy, "pos_bvec")
    wf.connect(copy_neg_corr, "out_bval", post_eddy, "neg_bval")
    wf.connect(copy_neg_corr, "out_bvec", post_eddy, "neg_bvec")

    # ------------------------------------------------------------------
    # STEP 9: Skullstrip AFTER EDDY
    # ------------------------------------------------------------------
    post_eddy_mask = pe.Node(Skullstrip(), name="skullstrip_post_eddy")
    post_eddy_mask.inputs.outdir = data_dir
    post_eddy_mask.inputs.lower_b = LOWER_B
    post_eddy_mask.inputs.b0max = int(B0RANGE)
    post_eddy_mask.inputs.bet4animal_z = int(cfg.bet4animal_z)
    if T1_PATH:
        post_eddy_mask.inputs.t1 = T1_PATH
    if BRAIN_MASK:
        post_eddy_mask.inputs.brain_mask = BRAIN_MASK

    wf.connect(post_eddy, "data_file", post_eddy_mask, "data_4d")
    wf.connect(post_eddy, "bval_file", post_eddy_mask, "bvals")

    # ------------------------------------------------------------------
    # STEP 9.5: Bundle final outputs for downstream steps
    # ------------------------------------------------------------------
    final_data = pe.Node(
        util.IdentityInterface(
            fields=["data_file", "bval_file", "bvec_file", "mask_file"]
        ),
        name="final_data",
    )
    wf.connect(post_eddy, "data_file", final_data, "data_file")
    wf.connect(post_eddy, "bval_file", final_data, "bval_file")
    wf.connect(post_eddy, "bvec_file", final_data, "bvec_file")
    wf.connect(post_eddy_mask, "mask_file", final_data, "mask_file")

    # ------------------------------------------------------------------
    # STEP 10: Select lower shell + DTIFIT
    # ------------------------------------------------------------------
    select_ls = pe.Node(SelectLowerShell(), name="select_lower_shell")
    select_ls.inputs.outprefix = os.path.join(data_dir, "data_lowershell")
    select_ls.inputs.approx_b = LOWER_B
    select_ls.inputs.db = int(B0RANGE)

    wf.connect(final_data, "data_file", select_ls, "data")
    wf.connect(final_data, "bval_file", select_ls, "bvals")
    wf.connect(final_data, "bvec_file", select_ls, "bvecs")

    dtifit = pe.Node(DTIFIT(), name="dtifit")
    dtifit.inputs.o = os.path.join(dmri_root, "dtifit", "dti")
    wf.connect(select_ls, "out_data", dtifit, "k")
    wf.connect(select_ls, "out_bvecs", dtifit, "r")
    wf.connect(select_ls, "out_bvals", dtifit, "b")
    wf.connect(final_data, "mask_file", dtifit, "m")

    # ------------------------------------------------------------------
    # STEP 11/12: Registration to standard (+ optional BedpostX/XTRACT)
    # ------------------------------------------------------------------
    stdreg = pe.Node(Reg2Std(), name="reg2std")
    stdreg.inputs.mode = MODE

    # default method is 'mmorf'; allow override via YAML
    if getattr(cfg, "stdreg_method", None) in ("mmorf", "fnirt"):
        stdreg.inputs.method = cfg.stdreg_method

    if MODE == "nhp":
        # NHP: packaged NMT
        stdreg.inputs.atl_fa = NMT_FA
        stdreg.inputs.atl_tensor = NMT_TENSOR
        stdreg.inputs.mmorf_config_template = MMORF_CFG
    else:
        # HUM: HCP1065
        stdreg.inputs.atl_fa = HCP1065_FA_1MM
        stdreg.inputs.atl_tensor = HCP1065_TENSOR_1MM
        stdreg.inputs.mmorf_config_template = MMORF_CFG

    stdreg.inputs.outdir = os.path.join(dmri_root, "stdreg")

    wf.connect(dtifit, "fa", stdreg, "fa")
    wf.connect(dtifit, "tensor", stdreg, "tensor")

    # BedpostX + XTRACT only if run_mode == "full"
    xtract_outdir = os.path.join(dmri_root, "xtract")
    xtract = None
    bpx = None

    if run_mode == "full":
        # BedpostX
        deps_bpx = pe.Node(util.Merge(2), name="deps_bpx")
        wf.connect(final_data, "data_file", deps_bpx, "in1")
        wf.connect(dtifit, "fa", deps_bpx, "in2")  # forces dtifit to finish

        bpx = pe.Node(Bedpostx(), name="bedpostx")
        bpx.inputs.datadir = data_dir
        bpx.inputs.use_gpu = not bool(getattr(cfg, "NO_GPU", False))
        wf.connect(deps_bpx, "out", bpx, "deps")

        # XTRACT
        xtract = pe.Node(Xtract(), name="xtract")
        xtract.inputs.outdir = xtract_outdir
        xtract.inputs.ptx_steplength = PTX_STEPLENGTH
        xtract.inputs.no_gpu = bool(getattr(cfg, "NO_GPU", False))

        if MODE == "nhp":
            xtract.inputs.species = "MACAQUE"
            xtract.inputs.stdref = STD_NMT
            if cfg.abs_xtract_profiles_dir:
                xtract.inputs.profiles_dir = cfg.abs_xtract_profiles_dir
        else:
            xtract.inputs.species = "HUMAN"
            xtract.inputs.stdref = MNI_T1_1MM
            if cfg.abs_xtract_profiles_dir:
                xtract.inputs.profiles_dir = cfg.abs_xtract_profiles_dir

        xtract.inputs.do_streamlines = bool(getattr(cfg, "STREAMLINES_DO", True))
        xtract.inputs.streamlines_density_threshold = float(
            getattr(cfg, "STREAMLINES_DENSITY_THRESHOLD", 1e-3)
        )
        xtract.inputs.streamlines_format = getattr(
            cfg, "STREAMLINES_FORMAT", "trk"
        )
        xtract.inputs.streamlines_ptx2_prefix = getattr(
            cfg, "STREAMLINES_PTX2_PREFIX", "densityNorm"
        )
        xtract.inputs.streamlines_num_jobs = int(
            getattr(cfg, "STREAMLINES_NUM_JOBS", 1)
        )

        wf.connect(bpx, "bpx_dir", xtract, "bpx_dir")
        wf.connect(stdreg, "warp_std2anat", xtract, "warp_std2anat")
        wf.connect(stdreg, "warp_anat2std", xtract, "warp_anat2std")

        # tract viewer (QC)
        xtract.inputs.viewer_script = pkg_file("nodes", "tract_viewer.py")
        xtract.inputs.do_viewer = bool(getattr(cfg, "DO_VIEWER", True))

    # ------------------------------------------------------------------
    # STEP 15: QA REPORT
    # ------------------------------------------------------------------
    qa = pe.Node(QAReport(), name="qa_report")
    qa.inputs.dmri_root = dmri_root
    qa.inputs.outdir = os.path.join(dmri_root, "QA")
    qa.inputs.pipeline_yaml = os.path.join(dmri_root, "invivo_dmri_config.yaml")

    qa.inputs.run_mode = run_mode

    # Nipype dirs and preproc graph
    qa.inputs.nipype_dir_preproc = os.path.join(dmri_root, "nipype_logging")
    qa.inputs.nipype_graph_preproc = os.path.join(
        dmri_root, "nipype_logging", "graph.png"
    )

    # Eddy QC summary from EDDY
    qa.inputs.eddy_qc_json = os.path.join(eddy_dir_root, "eddy.qc", "qc.json")

    # Standard-space mask / FA
    qa.inputs.nmt_mask_img = (
        pkg_file("files", "NMT_v2.0_sym_05mm_LR_brainmask.nii.gz")
        if MODE == "nhp"
        else os.path.join(
            FSLDIR, "data", "standard", "MNI152_T1_1mm_brain_mask.nii.gz"
        )
    )
    qa.inputs.nmt_fa_img = NMT_FA if MODE == "nhp" else HCP1065_FA_1MM

    # Core diffusion inputs to QA
    wf.connect(post_eddy, "data_file", qa, "data_4d")
    wf.connect(post_eddy, "bval_file", qa, "bvals")
    wf.connect(post_eddy, "bvec_file", qa, "bvecs")
    wf.connect(post_eddy_mask, "mask_file", qa, "mask_file")

    # Native FA / V1
    wf.connect(dtifit, "fa", qa, "fa_img")
    wf.connect(dtifit, "v1", qa, "v1_img")

    # XTRACT dir only in full mode
    if run_mode == "full" and xtract is not None:
        wf.connect(xtract, "outdir", qa, "xtract_dir")

    # --- Enforce run-last: merge late files into a single 'deps' connection ---
    qa_deps = pe.Node(util.Merge(3), name="qa_deps")
    wf.connect(dtifit, "fa", qa_deps, "in1")
    wf.connect(stdreg, "fa_in_std", qa_deps, "in2")
    wf.connect(post_eddy, "bval_file", qa_deps, "in3")
    wf.connect(qa_deps, "out", qa, "deps")

    return wf
