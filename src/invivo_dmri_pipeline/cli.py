# src/invivo_dmri_pipeline/cli.py
import argparse
import os

from .config import Config
from .pipeline import build_workflow


def _check_tractography_prereqs(cfg: Config) -> None:
    """
    Minimal on-disk sanity checks for a tract-only run.

    We assume:
      - Preprocessing has already completed.
      - The final diffusion data + bvals/bvecs/mask exist in dmri_root/data/.
      - Standard-space registration warps exist.

    BedpostX is NOT required here – tract-only mode will run it if missing.
    """
    dmri_root = cfg.abs_dmri_root
    data_dir = os.path.join(dmri_root, "data")

    # Required files for BedpostX + XTRACT
    data_4d = os.path.join(data_dir, "data.nii.gz")
    bvals   = os.path.join(data_dir, "bvals")
    bvecs   = os.path.join(data_dir, "bvecs")
    mask    = os.path.join(data_dir, "nodif_brain_mask.nii.gz")

    missing = [p for p in (data_4d, bvals, bvecs, mask) if not os.path.isfile(p)]
    if missing:
        raise SystemExit(
            "[cmc_invivo_pipeline] Tractography requested, but preprocessing "
            "outputs are missing in dmri_root/data/.\n"
            "Missing:\n  - " + "\n  - ".join(missing) +
            "\nRun the pipeline once with --preproc first."
        )

    # Standard-space warps required for XTRACT
    stdreg_dir = os.path.join(dmri_root, "stdreg")
    warp_a2s = os.path.join(stdreg_dir, "stdreg_anat_to_std_warp.nii.gz")
    warp_s2a = os.path.join(stdreg_dir, "stdreg_std_to_anat_warp.nii.gz")

    missing_warps = [p for p in (warp_a2s, warp_s2a) if not os.path.isfile(p)]
    if missing_warps:
        raise SystemExit(
            "[cmc_invivo_pipeline] Tractography requested, but registration warps "
            "are missing.\nMissing:\n  - " + "\n  - ".join(missing_warps) +
            "\nRun the preprocessing stage first (with --preproc)."
        )


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="cmc_invivo_pipeline",
        description="CMC in-vivo dMRI pipeline",
    )
    p.add_argument("config", help="Path to YAML config file")

    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose / debug logs",
    )

    # Stage flags
    p.add_argument(
        "--preproc",
        action="store_true",
        help="Run preprocessing stage.\n"
             "If no stage flags are given, this is the default behaviour.",
    )
    p.add_argument(
        "--tractography",
        action="store_true",
        help=(
            "Run tractography stage.\n\n"
            "  * With --preproc: full run (preproc + BedpostX + XTRACT + QA).\n"
            "  * Without --preproc: tract-only run (BedpostX + XTRACT + QA) "
            "using existing preprocessed data.\n"
        ),
    )

    args = p.parse_args(argv)
    cfg = Config.from_yaml(args.config)

    # ---------------------------
    # Determine run mode
    # ---------------------------
    if args.preproc and args.tractography:
        # Full pipeline in one go
        run_mode = "full"
    elif args.tractography:
        # Tract-only, using existing preproc outputs
        run_mode = "tract_only"
    else:
        # Default: preprocessing only
        run_mode = "preproc_only"

    # Tract-only needs existing preproc outputs on disk
    if run_mode == "tract_only":
        _check_tractography_prereqs(cfg)

    # Build and run the Nipype workflow
    wf = build_workflow(cfg, verbose=args.verbose, run_mode=run_mode)
    wf.write_graph(graph2use="flat")
    wf.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
