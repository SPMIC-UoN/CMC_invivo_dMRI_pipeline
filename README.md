# CMC In-Vivo dMRI Pipeline (Macaque, 10.5 T)

**Processing pipeline for in-vivo macaque ultra-high-field (10.5 T) dMRI data acquired as part of the Center for Mesoscale Connectomics (CMC).**

**Citation:** Warrington et al. (in prep)

---

## 1) Overview

This pipeline takes in-vivo macaque diffusion (phase blip pairs required) MRI, performs denoising, signal and sample drift correction, distortion corrections, skull-stripping, diffusion tensor modelling, regististration to NMT space, crossing fibre modelling, white matter bundle segmentation with XTRACT, and generates an overview QA report.

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
  Required tools: `topup`, `eddy` (or `eddy_cuda*`), `eddy_quad`, `bet4animal` or `bet`, and FSL utilities 
  Set `FSLDIR` and add `${FSLDIR}/bin` to `PATH`
- **MRtrix3 3.0.7+**  
  Used for optional Gibbs step
  **Note on Docker:** you can run MRtrix tools in the `mrtrix3/mrtrix3` container
- **ANTs**
  Used for optional N4 bias correction step
- **CUDA** (optional) for `eddy_cudaX` and GPU bedpostx  
  If no GPU, set `NO_GPU: true` in config to force CPU paths

### Python packages (installed by the package)
- `nipype`
- `nibabel`
- `numpy`, `scipy`
- `matplotlib`
- `reportlab`
- `pyyaml`

### External scripts and resources
- **Denoising wrapper script** (from https://github.com/SPMIC-UoN/EDDEN/tree/main/code). Path supplied in `config.yaml` as `denoise_sh`.
- **XTRACT profiles directory** containing `structureList` and tract definitions. Path supplied as `xtract_profiles_dir` - can be the FSL default.
- **MMORF config template** and NMT reference files are bundled in the package.

---

## 3) Installation

Below is a minimal, reproducible setup using Conda.

1. **Install core software**
   - Install **FSL**, **MRtrix3** and **ANTs**, and set environments.

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

4. **Verify external assets**
   - Ensure your **denoising wrapper** exists and is executable.
   - Ensure **XTRACT profiles directory** is available and readable.

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
dmri_root: /path/to/processing/proc             # output root
input_root: /path/to/INPUT_ROOT                 # top of AP/PA (or LR/RL) folders

# Denoising
denoise_sh: /path/to/denoise_wrapper.sh         # site-specific NORDIC/denoise entry

# Drift and acquisition
B0RANGE: 60                                     # threshold for identifying b0 volumes
ECHO_MS: 0.34                                   # EPI echo spacing (ms) used for readout calc
PIFACTOR: 3                                     # partial-Fourier factor for readout calc

# Pre-EDDY and combine
REMOVE_B0: true                                 # remove an initial b0 per run before combine
run_gibbs: true                                 # apply Gibbs (via Docker if configured)
run_n4: true                                    # apply N4 bias correction (via Docker if configured)
use_docker: true                                # let pipeline call Docker for Gibbs/N4 helpers
gibbs_image: mrtrix3/mrtrix3                    # Docker image to use for Gibbs
n4_image: antsx/ants:latest                     # Docker image to use for N4
interactive_tty: true                           # pass -it to docker (helps with HPC output)

# Eddy / Topup
eddy_extra_args: ""
COMBINE_MATCHED_FLAG: 1                         # 0: keep all volumes; 1: matched only

# Skullstrip after EDDY (final mask)
LOWER_B: 1000                                   # approximate value for “lower shell” selection
brain_mask: null                                # optional precomputed mask
t1: null                                        # optional T1 to assist skullstrip node

# Registration and tractography
stdreg_method: mmorf                            # mmorf (default) or fnirt
xtract_profiles_dir: /path/to/XTRACT_profiles   # must contain structureList + tracts
PTX_STEPLENGTH: 0.1                             # tractography step for XTRACT
NO_GPU: false                                   # set true on CPU-only systems

# Streamlines (if enabled via XTRACT)
STREAMLINES_DO: true
STREAMLINES_DENSITY_THRESHOLD: 1e-3
STREAMLINES_FORMAT: trk
STREAMLINES_PTX2_PREFIX: densityNorm
STREAMLINES_NUM_JOBS: 1

```

### Notes on key parameters
- **`B0RANGE`** defines what counts as b0 when identifying or removing initial volumes and reporting counts.  
- **`ECHO_MS` and `PIFACTOR`** are used to compute readout time for TOPUP/EDDY acqparams.  
- **`eddy_extra_args`** is passed directly to `eddy`. Add `--session` here if you want session-aware modeling. The pipeline will supply `--session=<series_index.txt>` automatically when `--session` is present.  
- **`COMBINE_MATCHED_FLAG`** controls whether to keep all volumes after EDDY or only matched AP/PA pairs.  
- **`LOWER_B`** defines the lower shell used for the DTI tensor fit.  
- **`xtract_profiles_dir`** must point to valid XTRACT tract definitions for your species template.  
- **Docker helpers**: when `use_docker: true`, Gibbs and N4 steps run inside containers you name in `gibbs_image` and `n4_image`.

---

## 6) Outputs (selected)

Under `dmri_root` the pipeline creates:

```
dmri_root/
  AP_denoised/, PA_denoised/              # per-run denoise outputs and sidecars
  data_combined*/                         # combined AP/PA (and Gibbs/N4 variants)
  data_combined_drift/                    # drift-corrected pairs
  topup/                                  # Pos_Neg_b0 and topup products + nodif_brain_mask
  eddy/
    eddy_unwarped_images.nii.gz
    Pos_Neg.bvals / Pos_Neg.bvecs
    index.txt / acqparams.txt / series_index.txt
    eddy.qc/ qc.json + PNGs               # eddy_quad output directory
  data/
    data.nii.gz
    bvals / bvecs
    nodif_brain_mask.nii.gz               # final mask after EDDY
  dtifit/
    dti_FA.nii.gz, dti_V1.nii.gz, dti_tensor.nii.gz, ...
  NMTreg/
    FA_in_std.nii.gz, warps, QC PNGs
  xtract/
    tract PNGs (viewer exports), tracts, stats
  QA/
    QA_report.pdf                         # multi-page report
```

---

## 7) Running the pipeline

```bash
cmc_invivo_pipeline /path/to/config.yaml -v
```

- `-v` enables informative Nipype logging to help with debugging.
- The pipeline copies your YAML into `dmri_root/invivo_dmri_config.yaml` for provenance.

---

## 8) Troubleshooting

- **`eddy_quad` fails with “directory already exists”**  
  The pipeline explicitly sets `-o <dmri_root>/eddy/eddy.qc`. If you re-run and want a fresh QC, delete that folder first:
  ```bash
  rm -rf <dmri_root>/eddy/eddy.qc
  ```
- **No `eddy_cuda` on GPU systems**  
  Ensure your CUDA install matches the `eddy_cuda*` binary you intend to use. Otherwise remove CUDA from `PATH` or set `NO_GPU: true` to force CPU EDDY.
- **Missing sidecars**  
  The pipeline requires `.bval` and `.bvec` for magnitude NIfTIs in AP/PA (or LR/RL). It will stop early if any are missing.
- **Mixed PE schemes**  
  Do not mix AP/PA and LR/RL in the same `input_root`. Use one scheme only.
- **Masks or T1**  
  You can provide a precomputed mask or T1 image if you need to override defaults for skull-strip after EDDY.

---