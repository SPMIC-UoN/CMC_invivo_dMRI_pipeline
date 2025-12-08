# src/invivo_dmri_pipeline/nodes/drift_assess.py

import argparse, subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go

def get_mean_signal_per_volume(nifti_path, mask_path=None):
    """Get mean signal per volume using fslstats -t -k <mask> -m"""
    cmd = ['fslstats', '-t', str(nifti_path)]
    if mask_path:
        cmd += ['-k', str(mask_path)]
    cmd += ['-m']
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return np.array([float(x) for x in result.stdout.strip().split()])

def plot_log_signal_interactive(raw, corrected, outfile_html):
    vols = np.arange(len(raw))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=vols,
        y=np.log(raw),
        mode='lines+markers',
        name='Raw (log)',
        line=dict(color='black')
    ))

    fig.add_trace(go.Scatter(
        x=vols,
        y=np.log(corrected),
        mode='lines+markers',
        name='Corrected (log)',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Log of Signal Intensity per Volume',
        xaxis_title='DWI volume #',
        yaxis_title='Log Signal Intensity',
        template='simple_white',
        width=1000,
        height=500
    )

    fig.write_html(outfile_html)
    print(f"✓ Saved interactive plot to {outfile_html}")

def main():
    parser = argparse.ArgumentParser(description="Log-scale plot of raw and corrected mean signal per volume.")
    parser.add_argument("--raw", required=True, help="Path to uncorrected 4D NIfTI file")
    parser.add_argument("--corrected", required=True, help="Path to drift-corrected 4D NIfTI file")
    parser.add_argument("--out", default="signal_plot.html", help="Output interactive HTML plot")
    parser.add_argument("--mask", help="Optional brainmask for signal estimation")
    args = parser.parse_args()

    print(f"\n--- Assess signal drift correction ---")

    raw_path = Path(args.raw).resolve()
    corrected_path = Path(args.corrected).resolve()
    mask = Path(args.mask).resolve() if args.mask else None

    raw_means = get_mean_signal_per_volume(raw_path, mask)
    corrected_means = get_mean_signal_per_volume(corrected_path, mask)

    if len(raw_means) != len(corrected_means):
        raise ValueError("Mismatch in number of volumes between raw and corrected data.")

    plot_log_signal_interactive(raw_means, corrected_means, args.out)

if __name__ == "__main__":
    main()
