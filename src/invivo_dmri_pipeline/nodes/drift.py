#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from tqdm import tqdm


def run(cmd, **kwargs):
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def get_fsl_bin():
    fsldir = os.getenv('FSLDIR')
    if not fsldir:
        print("Error: FSLDIR not set")
        sys.exit(1)
    return Path(fsldir) / 'bin'


def combine_bvals(bvals_files):
    all_bvals = []
    for b in bvals_files:
        arr = np.loadtxt(b)
        if arr.ndim == 0:
            all_bvals.append(float(arr))
        else:
            all_bvals.extend(arr.tolist())
    return np.array(all_bvals, dtype=float)


def prepare_masks(indat, brainmask, ventmask, stdref, toolsdir, outdir):
    FSLBIN = get_fsl_bin()
    print("\n--- Preparing masks ---")
    firstvol = outdir / 'firstvol.nii.gz'

    run([FSLBIN / 'fslroi', indat, firstvol, '0', '1'])
    run(['fslpython', f'{toolsdir}/reg2std.py', '-fa', firstvol, '-atl', stdref, '-out', outdir / 'NMTreg'])

    print('Warping masks to native diffusion space...')
    for img, name in zip([brainmask, ventmask], ['brainmask', 'ventmask']):
        run([FSLBIN / 'applywarp',
             '-i', img,
             '-r', firstvol,
             '-w', outdir / 'NMTreg' / 'stdreg_std_to_anat_warp',
             '--interp=nn',
             '-o', outdir / name])
        run([FSLBIN / 'fslmaths', outdir / name, '-dilM', '-bin', outdir / name])

    run([FSLBIN / 'fslmaths', outdir / 'brainmask', '-sub', outdir / 'ventmask', '-bin', outdir / 'brainmask'])

    firstvol.unlink(missing_ok=True)
    shutil.rmtree(outdir / 'NMTreg', ignore_errors=True)


def build_input_index(images):
    nvols = [img.shape[3] for img in images]
    cum = np.cumsum([0] + nvols)
    total = int(cum[-1])
    return nvols, cum, total


def global_to_local(gidx, cum):
    j = int(np.searchsorted(cum, gidx, side='right') - 1)
    local_t = int(gidx - cum[j])
    return j, local_t


def load_mask(mask_path, like_img):
    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata(dtype=np.float32)
    return (mask > 0.1)


def compute_b0_means(images, b0_indices, cum, use_mask=False, brain_mask=None, vent_mask=None):
    raw_means = []
    vent_means = [] if (use_mask and vent_mask is not None) else None

    for gidx in tqdm(b0_indices, desc="Computing b0 means"):
        j, t = global_to_local(gidx, cum)
        dataobj = images[j].dataobj
        vol = np.asanyarray(dataobj[..., t], dtype=np.float64)

        if use_mask and brain_mask is not None:
            m = brain_mask
            if m.shape != vol.shape:
                raise ValueError("Mask and data volume shapes do not match. Check registration.")
            raw_means.append(vol[m].mean())
            if vent_means is not None:
                vm = vent_mask
                if vm.shape != vol.shape:
                    raise ValueError("Ventricle mask and data volume shapes do not match. Check registration.")
                vent_means.append(vol[vm].mean())
        else:
            raw_means.append(np.mean(vol))

    raw_means = np.asarray(raw_means, dtype=np.float64)
    vent_means = np.asarray(vent_means, dtype=np.float64) if vent_means is not None else None
    return raw_means, vent_means


def fit_model_and_factors(model, total_vols, b0_indices, raw_means):
    n_idx = np.arange(len(b0_indices), dtype=np.float64)

    if model in ['linear', 'quadratic']:
        print(f"Fitting {model} model...")
        vpos = np.interp(np.arange(total_vols, dtype=np.float64), b0_indices, n_idx)

        if model == 'quadratic':
            coefs = Polynomial.fit(n_idx, raw_means, 2).convert().coef
            s0, d1, d2 = coefs[0], coefs[1], coefs[2]
            fit_vals = s0 + d1 * n_idx + d2 * n_idx**2
            interp_vals = s0 + d1 * vpos + d2 * vpos**2
            print(f"Coefficients: intercept={s0:.6f}, linear={d1:.6f}, quadratic={d2:.6f}")
        else:
            coefs = Polynomial.fit(n_idx, raw_means, 1).convert().coef
            s0, d1 = coefs[0], coefs[1]
            fit_vals = s0 + d1 * n_idx
            interp_vals = s0 + d1 * vpos
            print(f"Coefficients: intercept={s0:.6f}, slope={d1:.6f}")

        ref = raw_means[0]
        correction_factors = ref / np.asarray(interp_vals, dtype=np.float64)

    else:
        print("Using step-wise correction (single mode)...")
        fit_vals = raw_means.copy()
        correction_factors = np.ones(total_vols, dtype=np.float64)

        segment_idx = np.zeros(total_vols, dtype=int)
        j = 0
        for i in range(total_vols):
            if j + 1 < len(b0_indices) and i >= b0_indices[j + 1]:
                j += 1
            segment_idx[i] = j

        ref = raw_means[0]
        for i in range(total_vols):
            b0_j = segment_idx[i]
            if b0_j >= 1:
                correction_factors[i] = ref / raw_means[b0_j]

    return correction_factors, fit_vals


def apply_corrections_and_write(images, out_paths, correction_factors, cum, float32_out=False, brain_mask=None):
    assert len(images) == len(out_paths)
    total_vols = int(cum[-1])
    per_vol_raw_means = np.zeros(total_vols, dtype=np.float64)
    per_vol_corr_means = np.zeros(total_vols, dtype=np.float64)

    for file_idx, (img, outp) in enumerate(zip(images, out_paths)):
        hdr = img.header.copy()
        affine = img.affine
        shape = img.shape
        T = shape[3]

        if float32_out:
            out_dtype = np.float32
        else:
            orig_dtype = np.dtype(img.get_data_dtype())
            out_dtype = orig_dtype if np.issubdtype(orig_dtype, np.floating) else np.float32

        out_data = np.empty(shape, dtype=out_dtype)

        start = cum[file_idx]
        for t in tqdm(range(T), desc=f"Applying correction [{Path(outp).name}]"):
            gidx = start + t
            factor = correction_factors[gidx]
            vol = np.asanyarray(img.dataobj[..., t], dtype=np.float32)

            if brain_mask is not None:
                raw_mean = vol[brain_mask].mean()
            else:
                raw_mean = float(vol.mean())

            vol = vol * factor

            if brain_mask is not None:
                corr_mean = vol[brain_mask].mean()
            else:
                corr_mean = float(vol.mean())

            per_vol_raw_means[gidx] = raw_mean
            per_vol_corr_means[gidx] = corr_mean

            out_data[..., t] = vol.astype(out_dtype, copy=False)

        print('Saving output')
        corrected = nib.Nifti1Image(out_data, affine, header=hdr)
        corrected.header.set_slope_inter(1.0, 0.0)
        nib.save(corrected, str(outp))

    return per_vol_raw_means, per_vol_corr_means


def save_metrics_and_plots(outdir, model, b0_indices, raw_means, fit_vals,
                           vent_means, correction_factors,
                           per_vol_raw_means, per_vol_corr_means):
    n_idx = np.arange(len(b0_indices))
    corr_b0_factors = correction_factors[np.asarray(b0_indices, dtype=int)]
    corrected_means = raw_means * corr_b0_factors
    corrected_vent_means = vent_means * corr_b0_factors if vent_means is not None else None

    df = {
        'b0_order': n_idx,
        'b0_global_index': b0_indices,
        'raw_mean': raw_means,
        'fit_mean': fit_vals,
        'corrected_mean': corrected_means,
        'correction_factor_at_b0': corr_b0_factors
    }
    if vent_means is not None:
        df['vent_mean'] = vent_means
        df['corrected_vent_mean'] = corrected_vent_means

    pd.DataFrame(df).to_csv(outdir / "correction_metrics.csv", index=False)

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(top=0.88)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_top = fig.add_subplot(gs[0, 0])
    x_all = np.arange(len(per_vol_raw_means))
    ax_top.plot(x_all, per_vol_raw_means, '-', label='Mean (raw)', c='forestgreen')
    ax_top.plot(x_all, per_vol_corr_means, '--', label='Mean (corrected)', c='forestgreen')
    ax_top.set_xlabel("Volume index")
    ax_top.set_ylabel("Mean signal")
    ax_top.set_title("Volume-wise means")
    ax_top.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), borderaxespad=0)

    ax_bottom = fig.add_subplot(gs[1, 0])
    b0_labels = [f"{i}\n({idx})" for i, idx in enumerate(b0_indices)]
    b0_xticks = list(range(len(b0_indices)))
    if len(b0_labels) > 20:
        step = max(1, len(b0_labels) // int(len(b0_labels) * 0.3))
        display_indices = list(range(0, len(b0_labels), step))
        ticks = [b0_xticks[i] for i in display_indices]
        labels = [b0_labels[i] for i in display_indices]
    else:
        ticks = b0_xticks
        labels = b0_labels

    ax_bottom.plot(n_idx, raw_means, 'o-', label='Brain (raw)', c='cornflowerblue')
    if model in {'linear', 'quadratic'}:
        ax_bottom.plot(n_idx, fit_vals, 'r--', label=f'{model.capitalize()} fit (brain)', c='slategrey')
    ax_bottom.plot(n_idx, corrected_means, 'o--', label='Brain (corrected)', c='cornflowerblue')

    if vent_means is not None:
        ax_bottom.plot(n_idx, vent_means, 's-', label='Ventricle (raw)', c='coral')
        ax_bottom.plot(n_idx, corrected_vent_means, 's--', label='Ventricle (corrected)', c='coral')

    ax_bottom.set_xticks(ticks)
    ax_bottom.set_xticklabels(labels)
    ax_bottom.set_xlabel("b=0 # (index in full series)")
    ax_bottom.set_ylabel("Mean signal")
    
    if vent_means is not None:
        ax_bottom.set_yscale('log')
        ax_bottom.set_title("b=0 means")
        ax_bottom.legend(ncol=2, loc='lower right', bbox_to_anchor=(1.0, 1.02), borderaxespad=0)
    else:
        ax_bottom.set_title("b=0 means")
        ax_bottom.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), borderaxespad=0)

    fig.tight_layout()
    fig.savefig(outdir / "drift_plot.png")
    plt.close(fig)

    print(f"Plots saved to: {outdir}")


def validate_inputs(input_files, bvals_files, outdir, use_mask, mask, vent, std, toolsdir):
    missing = []
    for p in input_files:
        if not p.exists() or not p.is_file():
            missing.append(str(p))
    for b in bvals_files:
        if not b.exists() or not b.is_file():
            missing.append(str(b))
    if use_mask:
        for m in [mask, vent, std]:
            if m is None or not Path(m).exists() or not Path(m).is_file():
                missing.append(str(m))
        if toolsdir is None or not Path(toolsdir).exists() or not Path(toolsdir).is_dir():
            missing.append(f"{toolsdir}/")
        else:
            reg_script = Path(toolsdir) / "reg2std.py"
            if not reg_script.exists() or not reg_script.is_file():
                missing.append(str(reg_script))
        try:
            FSLBIN = get_fsl_bin()
            for exe in ['fslroi', 'applywarp', 'fslmaths']:
                if not (FSLBIN / exe).exists():
                    missing.append(str(FSLBIN / exe))
        except SystemExit:
            missing.append("FSLDIR")
    if missing:
        print("Error: the following required files or tools were not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        testfile = outdir / ".write_test"
        with open(testfile, "w") as f:
            f.write("ok")
        testfile.unlink(missing_ok=True)
    except Exception as e:
        print(f"Error: cannot create or write to outdir {outdir}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Optimised signal drift correction (nibabel + numpy)")
    parser.add_argument("-i", "--input", nargs='+', required=True, help="Input 4D NIfTI(s)")
    parser.add_argument("-bvals", nargs='+', required=True, help="Matching .bval file(s) in same order as inputs")
    parser.add_argument("-model", choices=["single", "linear", "quadratic"], default="single")
    parser.add_argument("-outdir", default="drift_analysis")
    parser.add_argument("--output", nargs='+', help="Output filenames (one per input); default adds _corrected")
    parser.add_argument("--mask", help="Brain mask in standard space (used only when a single input is provided)")
    parser.add_argument("--vent", help="Ventricle mask in standard space (optional, requires --mask)")
    parser.add_argument("--std", help="Standard-space reference for registration (required with --mask)")
    parser.add_argument("--toolsdir", default="", help="Path containing reg2std.py helper")
    parser.add_argument("--float32_out", action="store_true", help="Force float32 output dtype")
    args = parser.parse_args()

    print(f"\n--- Signal drift correction ---")

    input_files = [Path(f).resolve() for f in args.input]
    bvals_files = [Path(b).resolve() for b in args.bvals]
    outdir = Path(args.outdir).resolve()

    if len(input_files) != len(bvals_files):
        print("Error: mismatch in input and bvals count")
        sys.exit(1)

    use_mask = len(input_files) == 1 and args.mask is not None
    validate_inputs(input_files, bvals_files, outdir, use_mask, args.mask, args.vent, args.std, args.toolsdir)

    print("\n--- Loading inputs ---")
    images = [nib.load(str(p)) for p in input_files]
    for p, img in zip(input_files, images):
        if img.ndim != 4:
            print(f"Error: {p} is not 4D")
            sys.exit(1)

    nvols, cum, total_vols = build_input_index(images)

    bvals = combine_bvals(bvals_files)
    if bvals.shape[0] != total_vols:
        print(f"Error: combined bvals length {bvals.shape[0]} does not match total volumes {total_vols}")
        sys.exit(1)
    b0_indices = np.where(bvals == 0)[0]
    if b0_indices.size == 0:
        print("Error: no b=0 volumes found in bvals")
        sys.exit(1)

    brain_mask = vent_mask = None
    if use_mask:
        prepare_masks(input_files[0], Path(args.mask), Path(args.vent), Path(args.std), args.toolsdir, outdir)
        brain_mask = load_mask(outdir / 'brainmask.nii.gz', images[0])
        vent_mask = load_mask(outdir / 'ventmask.nii.gz', images[0])

    raw_means, vent_means = compute_b0_means(
        images, b0_indices, cum, use_mask=use_mask, brain_mask=brain_mask, vent_mask=vent_mask
    )

    correction_factors, fit_vals = fit_model_and_factors(
        args.model, total_vols, b0_indices, raw_means
    )

    output_names = args.output or [p.name.replace('.nii.gz', '_corrected.nii.gz') for p in input_files]
    out_paths = [str(outdir / name) for name in output_names]

    print("\n--- Generating corrected outputs ---")
    per_vol_raw_means, per_vol_corr_means = apply_corrections_and_write(
        images, out_paths, correction_factors, cum, float32_out=args.float32_out, brain_mask=brain_mask
    )
    for p in out_paths:
        print(f" → {p}")

    print("\n--- Saving QC metrics and plots ---")
    save_metrics_and_plots(outdir, args.model, b0_indices, raw_means, fit_vals, vent_means, correction_factors,
                           per_vol_raw_means, per_vol_corr_means)

    print("\n--- Done! ---")


if __name__ == "__main__":
    main()
