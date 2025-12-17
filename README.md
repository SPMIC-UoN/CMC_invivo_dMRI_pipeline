# CMC In Vivo dMRI Pipeline (Macaque, 10.5 T)

**Processing pipeline for in vivo macaque ultra-high-field (10.5 T) dMRI data acquired as part of the Center for Mesoscale Connectomics (CMC).**

**Citation:** Warrington et al. (in prep)

---

## 1) Overview

This pipeline takes in-vivo macaque diffusion (phase blip pairs required) MRI, performs denoising, signal and sample drift correction, distortion corrections, skull-stripping, diffusion tensor modelling, regististration to standard space, crossing fibre modelling, white matter bundle segmentation with XTRACT, and generates an overview QA report.

Command line entrypoint:

```bash
cmc_invivo_pipeline config.yaml -v
```

The workflow is Nipype-based and modular. Most heavy lifting is in FSL, with optional Docker helpers for Gibbs and N4 steps.
If the pipeline fails, or you need to adjust something, you can re-run from any point through Nipype caching framework.

---

## 2) Requirements

### Core software
- **Linux** with bash, coreutils
- **Python 3.10+**  
- **FSL 6.0.7.x or newer**
- **MRtrix3 3.0.7+**  
  Used for optional Gibbs and N4 steps
  **Note on Docker:** you can run MRtrix tools in the `mrtrix3/mrtrix3` container
- **CUDA** for GPU `eddy`, bedpostx and XTRACT - these should be installed with FSL already  
  If no GPU, set `NO_GPU: true` in config to force CPU paths

### External scripts and resources
- **Denoising wrapper script** (from https://github.com/SPMIC-UoN/EDDEN/tree/main/code). Path supplied in `config.yaml` as `denoise_sh`.

---

## 3) Installation

Below is a minimal, reproducible setup using Conda.

1. **Install core software**
   - Install **FSL** and **MRtrix3**, and set environments.

2. **Create a Python environment**
   ```bash
   mamba create -n cmc_invivo python=3.10 -y
   mamba activate cmc_invivo
   ```

3. **Install the pipeline package**
   - From source (recommended during development):
     ```bash
     pip install -e .
     ```
     This exposes the CLI `cmc_invivo_pipeline`.
---

## 4) Input data requirements

The pipeline assumes a **root folder** with either AP/PA or LR/RL phase-encoding schemes. Use exactly one scheme per dataset.

```
INPUT_ROOT/
  AP/         # magnitude NIfTI files named like AP_*.nii.gz with sidecars
  AP_ph/      # optional phase images (AP_ph_*.nii.gz)
  PA/         # or LR/
  PA_ph/      # or RL_ph/
```

- **Sidecars are required** for **magnitude** NIfTIs:
  - `AP_xxx.nii.gz` must have `AP_xxx.bvec` and `AP_xxx.bval`
  - Same for `PA` (or `LR`/`RL`)
- The pipeline checks that phase indices are a subset of matching magnitude indices.
- Multiple runs per direction are supported. They will be denoised, optionally have an initial b0 removed, combined, and then drift-corrected.

---

## 5) The `config.yaml`

A minimal example illustrates the key fields. Paths may be relative or absolute; the pipeline writes all products into `dmri_root`.

```yaml
# Required roots
dmri_root: /path/to/processing/proc
input_root: /path/to/INPUT_ROOT

denoise_sh: /path/to/denoise_wrapper.sh

B0RANGE: 60                  # threshold for identifying b0 volumes
ECHO_MS: 0.34                # echo spacing (ms)
PIFACTOR: 3                  # parallel imaging factor (GRAPPA or equivalent)

run_gibbs: true # run Gibbs unringing?
run_n4: true # run bias field correction?
use_docker: true
docker_image: mrtrix3/mrtrix3
interactive_tty: true

LOWER_B: 1000
brain_mask: null # option to supply a precomputed brain mask in T1 space
t1: null # required if providing T1 space brain mask

# Registration and tractography
stdreg_method: mmorf         # mmorf or fnirt

```

### Notes on key parameters
- **`B0RANGE`** defines what counts as b0 when identifying or removing initial volumes and reporting counts.  
- **`ECHO_MS` and `PIFACTOR`** are used to compute readout time for TOPUP/EDDY acqparams.  
- **`eddy_extra_args`** is passed directly to `eddy`. Add `--session` here if you want session-aware modeling. The pipeline will supply `--session=<series_index.txt>` automatically when `--session` is present.  
- **`COMBINE_MATCHED_FLAG`** controls whether to keep all volumes after EDDY or only matched AP/PA pairs.  
- **`LOWER_B`** defines the lower shell used for the DTI tensor fit.  
- **Docker helpers**: when `use_docker: true`, Gibbs and N4 steps run inside a container you name in `docker_image`.

---

## 6) Outputs (selected)

Under `dmri_root` the pipeline creates:

```
dmri_root/
  topup/                                  # Pos_Neg_b0 and topup products + nodif_brain_mask
  eddy/
    eddy_unwarped_images.nii.gz
    eddy.qc/ qc.json + PNGs               # eddy_quad output directory
  data/
    data.nii.gz
    bvals / bvecs
    nodif_brain_mask.nii.gz               
  dtifit/
    dti_FA.nii.gz, dti_V1.nii.gz, dti_tensor.nii.gz, ...
  stdreg/
    FA_in_std.nii.gz, warps
  xtract/
    tract PNGs (viewer exports), tracts containging xtract tract density maps
  QA/
    QA_report.pdf                         # multi-page report
```

---

## 7) Running the pipeline

```bash
cmc_invivo_pipeline /path/to/config.yaml -v
```
- `--preproc` sets the pipeline to only run pre-processing up to and including diffusion tensor modelling and standard space registrations.
- `--tractography` sets the pipeline to additionally run crossing fibre modelling and landmark-based tractoraphy with XTRACT.
- `-v` enables informative Nipype logging to help with debugging.
- The pipeline copies your YAML into `dmri_root/invivo_dmri_config.yaml` for provenance.

---
