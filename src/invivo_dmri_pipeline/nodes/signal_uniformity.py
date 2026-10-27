"""Lightweight low-frequency signal-uniformity assessment for 3D dMRI images.

This is a QA estimator, not an N4 replacement. It uses masked, normalised
Gaussian smoothing in log-signal space to estimate a multiplicative field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def _otsu(values: np.ndarray, bins: int = 512) -> float:
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("No finite intensities available for automatic masking")
    lo, hi = np.percentile(values, (0.5, 99.5))
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    centres = (edges[:-1] + edges[1:]) / 2
    weight0 = np.cumsum(hist)
    weight1 = hist.sum() - weight0
    mean0 = np.cumsum(hist * centres) / np.maximum(weight0, 1)
    reverse = np.cumsum((hist * centres)[::-1])[::-1]
    mean1 = reverse / np.maximum(weight1, 1)
    score = weight0 * weight1 * (mean0 - mean1) ** 2
    return float(centres[int(np.nanargmax(score[:-1]))])


def automatic_mask(data: np.ndarray) -> np.ndarray:
    """Create a conservative largest-component mask from a reference shell."""
    positive = data[np.isfinite(data) & (data > 0)]
    threshold = _otsu(positive)
    mask = np.isfinite(data) & (data > threshold)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        mask = labels == int(np.argmax(sizes))
    mask = ndimage.binary_fill_holes(mask)
    if np.count_nonzero(mask) < 1000:
        raise ValueError("Automatic brain mask is implausibly small; supply --mask")
    return mask


def estimate_signal_field(
    data: np.ndarray,
    mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
    fwhm_mm: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return multiplicative field, observed/expected ratio, low-signal score and metrics."""
    valid = mask & np.isfinite(data) & (data > 0)
    if np.count_nonzero(valid) < 1000:
        raise ValueError("Mask contains too few positive finite voxels")
    signal = data[valid].astype(np.float64)
    low, high = np.percentile(signal, (1, 99))
    floor = max(float(low), np.finfo(np.float32).tiny)
    clipped = np.clip(data.astype(np.float64), floor, float(high))
    log_signal = np.zeros(data.shape, dtype=np.float64)
    log_signal[valid] = np.log(clipped[valid])
    sigma_mm = float(fwhm_mm) / np.sqrt(8 * np.log(2))
    sigma_vox = tuple(sigma_mm / max(float(size), 1e-6) for size in voxel_sizes)
    weights = ndimage.gaussian_filter(valid.astype(np.float64), sigma_vox,
                                      mode="constant", cval=0)
    smooth = ndimage.gaussian_filter(log_signal, sigma_vox,
                                     mode="constant", cval=0)
    smooth = smooth / np.maximum(weights, 1e-6)
    centre = float(np.median(smooth[valid]))
    field = np.zeros(data.shape, dtype=np.float32)
    field[mask] = np.exp(smooth[mask] - centre).astype(np.float32)
    expected_scale = float(np.exp(centre))
    ratio = np.zeros(data.shape, dtype=np.float32)
    ratio[valid] = (data[valid] / np.maximum(field[valid] * expected_scale, floor)).astype(np.float32)
    # A continuous QA score: 0 at/above 75% of the local expectation, 1 at zero.
    low_signal = np.zeros(data.shape, dtype=np.float32)
    low_signal[valid] = np.clip((0.75 - ratio[valid]) / 0.75, 0, 1)
    corrected = data[valid] / np.maximum(field[valid], 1e-6)
    metrics = {
        "fwhm_mm": float(fwhm_mm),
        "mask_voxels": int(np.count_nonzero(mask)),
        "field_p05_p50_p95": [float(v) for v in np.percentile(field[valid], (5, 50, 95))],
        "observed_expected_p05_p50_p95": [float(v) for v in np.percentile(ratio[valid], (5, 50, 95))],
        "fraction_below_0.75_expected": float(np.mean(ratio[valid] < 0.75)),
        "robust_cv_before": float(np.std(signal[(signal >= low) & (signal <= high)]) /
                                  np.mean(signal[(signal >= low) & (signal <= high)])),
        "robust_cv_after": float(np.std(corrected) / np.mean(corrected)),
    }
    return field, ratio, low_signal, metrics


def _save_like(reference: nib.Nifti1Image, data: np.ndarray, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), reference.affine, header), str(path))


def process_image(image_path: Path, output_dir: Path, mask_path: Path | None,
                  fwhm_mm: float) -> dict:
    image = nib.load(str(image_path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {image_path}")
    if mask_path:
        mask_image = nib.load(str(mask_path))
        if mask_image.shape != image.shape or not np.allclose(mask_image.affine, image.affine):
            raise ValueError("Mask and shell average do not share the same grid")
        mask = np.asarray(mask_image.dataobj) > 0
        mask_source = str(mask_path)
    else:
        mask = automatic_mask(data)
        mask_source = "automatic"
    field, ratio, low_signal, metrics = estimate_signal_field(
        data, mask, image.header.get_zooms()[:3], fwhm_mm)
    stem = image_path.name.removesuffix(".nii.gz").removesuffix(".nii")
    target = output_dir / stem
    target.mkdir(parents=True, exist_ok=True)
    _save_like(image, mask.astype(np.float32), target / "mask.nii.gz")
    _save_like(image, field, target / "smooth_signal_field.nii.gz")
    _save_like(image, ratio, target / "observed_expected_ratio.nii.gz")
    _save_like(image, low_signal, target / "low_signal_score.nii.gz")
    metrics.update({"input": str(image_path.resolve()), "mask_source": mask_source})
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path, help="3D shell-average NIfTI images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path,
                        help="Brain mask on the same grid; strongly recommended so "
                             "dropout is not excluded by automatic masking")
    parser.add_argument("--fwhm-mm", type=float, default=30.0,
                        help="Smooth-field FWHM in mm (default: 30)")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for path in args.images:
        summary[path.name] = process_image(path, args.output_dir, args.mask, args.fwhm_mm)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
