"""Self-contained visual QA for CMC in-vivo diffusion processing.

The HTML report is the primary, theme-switchable deliverable. Two PDF exports
and the individual PNG panels are written alongside it. Missing inputs become
visible QA notes rather than fatal exceptions.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import shlex
import shutil
import subprocess
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from nipype.interfaces.base import (
    BaseInterfaceInputSpec, Directory, File, SimpleInterface, TraitedSpec,
    isdefined, traits,
)

from .signal_uniformity import process_image as process_signal_uniformity


THEMES = {
    "light": {"bg": "#ffffff", "fg": "#172033", "grid": "#e5eaf0",
              "blue": "#3274a1", "amber": "#d78322", "green": "#178a78",
              "red": "#c45367"},
    "dark": {"bg": "#111827", "fg": "#f3f4f6", "grid": "#374151",
             "blue": "#60a5fa", "amber": "#fbbf24", "green": "#34d399",
             "red": "#fb7185"},
}
TAG_RE = re.compile(r"^(AP|PA|LR|RL)(_ph)?_(\d+)\.nii(?:\.gz)?$", re.I)


def _existing(path):
    path = Path(path).expanduser() if path else None
    return path if path and path.is_file() else None


def _run_key(path):
    match = TAG_RE.match(Path(path).name)
    return (match.group(1).upper(), int(match.group(3))) if match else None


def _image_paths(entries):
    """Expand configuration entries that may be images or image directories."""
    paths = []
    for entry in entries or []:
        path = Path(entry).expanduser()
        if path.is_dir():
            paths.extend(p for p in path.iterdir() if TAG_RE.match(p.name))
        elif path.is_file() and TAG_RE.match(path.name):
            paths.append(path)
    return sorted(set(paths), key=lambda p: (_run_key(p) or ("", 0), str(p)))


def _configured_paths(entries, config_path):
    """Resolve relative paths as the pipeline does, using the saved YAML location."""
    base = Path(config_path).parent if config_path else Path.cwd()
    if isinstance(entries, (str, Path)):
        entries = [entries]
    result = []
    for entry in entries or []:
        path = Path(entry).expanduser()
        result.append(path if path.is_absolute() else base / path)
    return result


def _bvals(path):
    if not _existing(path):
        return None
    try:
        return np.atleast_1d(np.loadtxt(path, dtype=float)).ravel()
    except (OSError, ValueError):
        return None


def _sidecar(path, suffix):
    name = str(path)
    stem = name[:-7] if name.endswith(".nii.gz") else name[:-4] if name.endswith(".nii") else name
    return Path(stem + suffix)


def _shell_groups(values, b0range=60):
    """Group nearby nominal b values without assuming fixed shells."""
    values = np.asarray(values, dtype=float)
    result = {}
    bzero = np.flatnonzero(np.isfinite(values) & (values <= b0range))
    if len(bzero):
        result[0] = bzero
    groups = []
    for value in sorted(set(float(v) for v in values if np.isfinite(v) and v > b0range)):
        if not groups or value - np.median(groups[-1]) > b0range:
            groups.append([value])
        else:
            groups[-1].append(value)
    for group in groups:
        centre = float(np.median(group))
        shell = int(round(centre / 10) * 10)
        result[shell] = np.flatnonzero(np.isin(values, group))
    return result


def _canon(data, affine):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    return np.asarray(nib.as_closest_canonical(image).dataobj)


def _volume(path, index=0):
    image = nib.load(str(path))
    if len(image.shape) == 4:
        index = min(max(int(index), 0), image.shape[3] - 1)
        data = np.asarray(image.dataobj[..., index], dtype=np.float32)
    else:
        data = np.asarray(image.dataobj, dtype=np.float32)
    return _canon(data, image.affine), image


def _mean_volume(path, indices):
    image = nib.load(str(path))
    if len(image.shape) == 3:
        return _volume(path)[0]
    indices = [int(i) for i in indices if 0 <= int(i) < image.shape[3]]
    if not indices:
        raise ValueError(f"No valid volume indices for {path}")
    mean = np.zeros(image.shape[:3], dtype=np.float32)
    # Stream one 3-D volume at a time, so high-resolution DWI does not require
    # holding a full 4-D run in memory.
    for index in indices:
        mean += np.asarray(image.dataobj[..., index], dtype=np.float32)
    mean /= len(indices)
    return _canon(mean, image.affine)


def _views(data, centre=None):
    spatial = data.shape[:3]
    xyz = centre or tuple(s // 2 for s in spatial)
    xyz = tuple(min(max(int(v), 0), spatial[i] - 1) for i, v in enumerate(xyz))
    return [np.rot90(data[xyz[0], ...]),
            np.rot90(data[:, xyz[1], ...]),
            np.rot90(data[:, :, xyz[2], ...])]


def _scale(arrays):
    samples = [np.asarray(a[::5, ::5, ::5]).ravel() for a in arrays]
    values = np.concatenate(samples)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, 99.5)) if values.size else 1.0


def _figure_rc(theme):
    t = THEMES[theme]
    return {"font.family": "DejaVu Sans", "font.size": 11,
            "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
            "axes.edgecolor": t["fg"], "axes.labelcolor": t["fg"],
            "text.color": t["fg"], "xtick.color": t["fg"],
            "ytick.color": t["fg"]}


def _montage(rows, title, output, labels=("Sagittal", "Coronal", "Axial"),
             cmap="gray", vmax=None, masks=None, rgb=False, theme="dark"):
    """Rows of matched, three-plane views with optional mask outlines."""
    with plt.rc_context(_figure_rc(theme)):
        fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.45 * len(rows)),
                                 squeeze=False, layout="constrained")
        fig.suptitle(title, fontsize=18, weight="semibold")
        for i, (name, data) in enumerate(rows):
            views = _views(data)
            overlay = _views(masks[i]) if masks and masks[i] is not None else None
            for j, (axis, view) in enumerate(zip(axes[i], views)):
                axis.imshow(view if rgb else view, cmap=None if rgb else cmap,
                            vmin=None if rgb else 0,
                            vmax=None if rgb else vmax, interpolation="nearest")
                if overlay is not None:
                    axis.contour(overlay[j] > 0.5, levels=[0.5], colors="#fb7185",
                                 linewidths=1.1)
                axis.set_title(labels[j] if i == 0 else "", fontsize=11)
                axis.set_ylabel(name, fontsize=12, weight="semibold")
                axis.set_xticks([])
                axis.set_yticks([])
                for side in axis.spines.values():
                    side.set_visible(False)
        fig.savefig(output, dpi=170, facecolor=THEMES[theme]["bg"])
        plt.close(fig)


def _match_grid(source, target, order=1):
    """Resample a 3-D image only when the affine or shape differs."""
    if source.shape[:3] == target.shape[:3] and np.allclose(source.affine, target.affine):
        return source
    from nibabel.processing import resample_from_to
    return resample_from_to(source, (target.shape[:3], target.affine), order=order)


class QAReportInputSpec(BaseInterfaceInputSpec):
    dmri_root = traits.Str(desc="Processing directory")
    outdir = traits.Str(desc="QA output directory")
    pipeline_yaml = File(exists=False)
    eddy_qc_json = File(exists=False)
    run_mode = traits.Enum("preproc_only", "full", "tract_only", usedefault=True)
    nipype_dir_preproc = traits.Str()
    nipype_dir_tract = traits.Str()
    nipype_graph_preproc = File(exists=False)
    nipype_graph_tract = File(exists=False)
    nmt_mask_img = File(exists=False)
    nmt_fa_img = File(exists=False)
    data_4d = File(exists=False)
    bvals = File(exists=False)
    bvecs = File(exists=False)
    mask_file = File(exists=False)
    fa_img = File(exists=False)
    v1_img = File(exists=False)
    xtract_dir = File(exists=False)
    deps = traits.Any(desc="Dependency token to run QA last")


class QAReportOutputSpec(TraitedSpec):
    report_pdf = File(desc="Light PDF (legacy output name)")
    report_pdf_dark = File(desc="Dark PDF")
    report_html = File(desc="Theme-switchable HTML")
    manifest_json = File(desc="QA source and issue manifest")
    figures_dir = Directory(desc="Individual PNG panels")


class QAReport(SimpleInterface):
    input_spec = QAReportInputSpec
    output_spec = QAReportOutputSpec

    def _run_interface(self, runtime):
        def value(name):
            item = getattr(self.inputs, name)
            return str(item) if isdefined(item) and item else ""

        root = Path(value("dmri_root") or ".").expanduser().resolve()
        outdir = Path(value("outdir") or root / "QA").expanduser().resolve()
        supplied = {name: value(name) for name in (
            "pipeline_yaml", "eddy_qc_json", "nipype_dir_preproc", "nipype_dir_tract",
            "nipype_graph_preproc", "nipype_graph_tract", "nmt_mask_img",
            "nmt_fa_img", "data_4d", "bvals", "bvecs", "mask_file",
            "fa_img", "v1_img", "xtract_dir")}
        composer = _Composer(root, outdir, self.inputs.run_mode, supplied)
        paths = composer.build()
        self._results.update({key: str(path) for key, path in paths.items()})
        return runtime


class _Composer:
    REPORT_VERSION = "2026-10-27.4"

    def __init__(self, root, outdir, run_mode, supplied):
        self.root = root
        self.outdir = outdir
        self.figures = outdir / "figures"
        self.run_mode = run_mode
        self.supplied = supplied
        self.timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        self.sections = []
        self.issues = []
        self.sources = {}
        self.config = {}
        self.mode = "unknown"
        self.raw_runs = {}
        self.phase_runs = {}
        self.steps = []
        self.shell_average_process = None
        self.shell_average_specs = []
        self.shell_average_temps = []
        self.shell_average_error = None

    def issue(self, severity, title, detail):
        self.issues.append({"severity": severity, "title": title, "detail": str(detail)})

    def section(self, key, title, intro):
        section = {"id": key, "title": title, "intro": intro, "figures": [], "notes": []}
        self.sections.append(section)
        return section

    def figure(self, section, stem, title, caption, render, sources=()):
        self.themed_figure(section, stem, title, caption, render, sources)

    def themed_figure(self, section, stem, title, caption, render, sources=()):
        names = {}
        failures = defaultdict(list)
        for theme in THEMES:
            path = self.figures / f"{stem}_{theme}.png"
            try:
                render(path, theme)
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError("rendering did not create a PNG")
                names[theme] = path.name
            except Exception as exc:
                failures[str(exc)].append(theme)
        if failures:
            # Report a failed stage once, rather than one alert per colour theme.
            details = [message if len(themes) == len(THEMES)
                       else f"{', '.join(themes)}: {message}"
                       for message, themes in failures.items()]
            self.issue("warning", f"Could not fully render {title}",
                       "; ".join(details))
        if names:
            section["figures"].append({"title": title, "caption": caption,
                                       "light": names.get("light") or names.get("dark"),
                                       "dark": names.get("dark") or names.get("light"),
                                       "sources": [str(p) for p in sources if p]})
        else:
            section["notes"].append(f"{title} unavailable.")

    def _source(self, key, *candidates):
        for item in candidates:
            if item and Path(item).is_file():
                self.sources[key] = str(Path(item).resolve())
                return Path(item)
        return None

    def _read_config(self):
        path = self._source("config", self.supplied["pipeline_yaml"],
                            self.root / "invivo_dmri_config.yaml")
        if not path:
            self.issue("warning", "Saved configuration unavailable",
                       "Raw inputs and phase usage cannot be fully audited.")
            return
        try:
            self.config = {str(k).lower(): v for k, v in
                           (yaml.safe_load(path.read_text()) or {}).items()}
            self.mode = str(self.config.get("mode", "unknown")).lower()
        except Exception as exc:
            self.issue("error", "Cannot read saved configuration", exc)

    def _discover_raw(self):
        config_path = self.sources.get("config")
        mags = _image_paths(_configured_paths(self.config.get("mag_files", []), config_path))
        phases = _image_paths(_configured_paths(self.config.get("phase_files", []), config_path))
        if not mags:
            candidate = Path(self.config.get("input_root") or self.root.parent)
            for tag in ("AP", "PA", "LR", "RL"):
                mags += _image_paths([candidate / tag])
        self.raw_runs = {_run_key(p): p for p in mags if _run_key(p)}
        self.phase_runs = {_run_key(p): p for p in phases if _run_key(p)}
        self.sources["raw_magnitude_count"] = len(self.raw_runs)
        self.sources["raw_phase_count"] = len(self.phase_runs)

    def _phase_audit(self):
        declared = _configured_paths(self.config.get("phase_files", []),
                                     self.sources.get("config"))
        denoise = bool(self.config.get("denoise", True))
        if not declared:
            self.issue("info", "No phase data declared",
                       "The configuration does not request complex-valued denoising.")
            return
        missing_dirs = [str(p) for p in declared if not p.exists()]
        if missing_dirs:
            self.issue("error", "Configured phase paths not found", ", ".join(missing_dirs))
        if denoise and not self.phase_runs:
            self.issue("error", "Phase data declared but not discovered",
                       "No matching phase NIfTI was found. Denoising may have run magnitude-only.")
            return
        for key, mag in self.raw_runs.items():
            phase = self.phase_runs.get(key)
            if denoise and phase is None:
                self.issue("error", f"Missing phase for {mag.name}",
                           "Expected a phase NIfTI with the same direction and run number.")
            elif phase:
                try:
                    if nib.load(str(mag)).shape != nib.load(str(phase)).shape:
                        self.issue("error", f"Magnitude/phase dimensions differ for {mag.name}",
                                   f"{mag} versus {phase}")
                except Exception as exc:
                    self.issue("warning", f"Could not inspect phase pair {mag.name}", exc)
        extras = sorted(set(self.phase_runs) - set(self.raw_runs))
        if extras:
            self.issue("warning", "Unpaired phase data", ", ".join(
                self.phase_runs[key].name for key in extras))
        if not denoise:
            self.issue("info", "Denoising disabled", "Phase data were not expected to be used.")
            return

        commands = sorted((self.root / "nipype_logging").glob("*_dn/mapflow/*/command.txt"))
        observed = {}
        for path in commands:
            try:
                tokens = shlex.split(path.read_text().strip())
                options = {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1)
                           if tokens[i] in ("-magn", "-phase", "-domain")}
                key = _run_key(options.get("-magn", ""))
                if key:
                    observed[key] = options
            except Exception as exc:
                self.issue("warning", "Could not read a denoising command", f"{path}: {exc}")
        if not observed:
            self.issue("warning", "Denoising domain not verified",
                       "Nipype denoising command.txt files were unavailable; raw phase files alone "
                       "do not prove they were used.")
        for key, mag in self.raw_runs.items():
            expected = self.phase_runs.get(key)
            options = observed.get(key)
            if options is None:
                if expected and observed:
                    self.issue("warning", f"Phase use not verified for {mag.name}",
                               "No matching denoising command.txt was found for this run.")
                continue
            domain = options.get("-domain", "").lower()
            used = Path(options.get("-phase", ""))
            if expected and (domain != "complex" or used.resolve() != expected.resolve()):
                self.issue("error", f"Phase not used for {mag.name}",
                           f"Recorded domain={domain or 'unknown'}, phase={used}. "
                           f"Expected complex with {expected}.")
            elif not expected and domain == "complex":
                self.issue("error", f"Unexpected phase pairing for {mag.name}",
                           f"Pipeline used {used}, but no same-number phase file was found.")

    def _step_status(self):
        expected_xtract = self.run_mode in ("full", "tract_only")
        checks = [
            ("Raw", bool(self.raw_runs), True),
            ("Denoising", bool(list(self.root.glob("*_denoised/*_1.nii.gz"))),
             bool(self.config.get("denoise", True))),
            ("Gibbs / N4", bool(list(self.root.glob("data_combined_gibbs_N4/*.nii.gz"))),
             bool(self.config.get("run_gibbs", True) or self.config.get("run_n4", True))),
            ("Drift", (self.root / "data_combined_drift" / "drift_plot.png").exists(), True),
            ("TOPUP", (self.root / "topup" / "hifib0.nii.gz").exists(), True),
            ("EDDY", (self.root / "eddy" / "eddy_unwarped_images.nii.gz").exists(), True),
            ("DTIFIT", (self.root / "dtifit" / "dti_FA.nii.gz").exists(), True),
            ("Registration", (self.root / "stdreg" / "FA_in_STD.nii.gz").exists(), True),
            ("XTRACT", bool(list((self.root / "xtract" / "tracts").glob("*/densityNorm.nii.gz"))),
             expected_xtract),
        ]
        self.steps = [{"name": name, "state": "complete" if present else
                       "missing" if required else "not requested"}
                      for name, present, required in checks]
        for step in self.steps:
            if step["state"] == "missing":
                self.issue("warning", f"{step['name']} output not found",
                           "This processing stage was expected for the selected run mode; "
                           "its visual QA may be incomplete.")

    def _audit_failures(self):
        logdir = self.root / "nipype_logging"
        if not logdir.is_dir():
            return
        crashes = sorted(p for p in logdir.rglob("crash-*") if p.is_file())
        if crashes:
            sample = ", ".join(str(p.relative_to(self.root)) for p in crashes[:4])
            self.issue("warning", f"{len(crashes)} Nipype crash file(s) found",
                       f"May include stale failures from a previous attempt: {sample}")

    def build(self):
        self.figures.mkdir(parents=True, exist_ok=True)
        self._read_config()
        self._discover_raw()
        self._phase_audit()
        self._step_status()
        self._audit_failures()
        self._start_final_shell_averages()
        self._topup()
        self._denoising()
        self._eddy()
        self._final_shell_averages()
        self._signal_uniformity()
        self._mask()
        self._diffusion_maps()
        self._registration()
        self._xtract()
        paths = self._write_reports()
        return paths

    def _denoising(self):
        section = self.section(
            "denoising", "Before and after denoising",
            "Matched raw and denoised magnitude data are shown for a representative "
            "volume from each shell. Display scaling is shared across each before/after "
            "pair. Raw acquisitions are not shell-averaged.")
        if not bool(self.config.get("denoise", True)):
            section["notes"].append("Denoising was disabled in the saved configuration.")
            return
        if not self.raw_runs:
            section["notes"].append("Raw magnitude runs could not be located from the saved configuration.")
            return
        tolerance = float(self.config.get("b0range", 60))
        representatives = {}
        tags = sorted({key[0] for key in self.raw_runs})
        for tag in tags:
            raw_for_tag = sorted(((key, path) for key, path in self.raw_runs.items()
                                  if key[0] == tag), key=lambda item: item[0][1])
            for ordinal, (key, raw) in enumerate(raw_for_tag, start=1):
                den = self.root / f"{tag}_denoised" / f"{tag}_{ordinal}.nii.gz"
                if not den.is_file():
                    self.issue("warning", f"Denoised run unavailable for {raw.name}", den)
                    continue
                values = _bvals(_sidecar(raw, ".bval"))
                if values is None:
                    values = _bvals(_sidecar(den, ".bval"))
                if values is None:
                    self.issue("warning", f"B-values unavailable for {raw.name}",
                               "Cannot select shell examples or averages.")
                    continue
                try:
                    n_raw = nib.load(str(raw)).shape[3]
                    n_den = nib.load(str(den)).shape[3]
                    if n_raw != n_den or len(values) != n_raw:
                        raise ValueError(f"raw={n_raw}, denoised={n_den}, bvals={len(values)}")
                except Exception as exc:
                    self.issue("error", f"Denoising dimensions mismatch for {raw.name}", exc)
                    continue
                for shell, indices in _shell_groups(values, tolerance).items():
                    # One representative raw/denoised example per shell across
                    # the session. Raw shell averaging is intentionally avoided.
                    representatives.setdefault(shell, (tag, raw, den, indices, ordinal))
        for shell, (tag, raw, den, indices, ordinal) in sorted(representatives.items()):
            label = f"{tag} run {ordinal}, b={shell}"
            example = int(indices[len(indices) // 2])
            prefix = f"denoise_{tag.lower()}_b{shell}"
            self.figure(
                section, prefix + "_example", label + " - example volume",
                f"Volume {example} of {raw.name}. These panels show the same acquisition "
                "before and after denoising; structured differences merit review.",
                lambda path, theme, raw=raw, den=den, example=example, label=label:
                    self._denoise_panel(raw, den, [example], label + " - example volume", path, theme),
                (raw, den, _sidecar(raw, ".bval")))
        if not section["figures"]:
            section["notes"].append("No complete raw/denoised run with usable b-values was found.")

    def _start_final_shell_averages(self):
        """Start reusable FSL shell means while the remaining QA panels render."""
        data_path = self._source("final_corrected_data", self.supplied["data_4d"],
                                 self.root / "data" / "data.nii.gz")
        bvals_path = self._source("final_corrected_bvals", self.supplied["bvals"],
                                  self.root / "data" / "bvals")
        values = _bvals(bvals_path)
        if not data_path or values is None:
            self.shell_average_error = "Final corrected data or b-values were unavailable."
            return
        try:
            shape = nib.load(str(data_path)).shape
            if len(shape) != 4 or shape[3] != len(values):
                raise ValueError(f"data shape={shape}, b-values={len(values)}")
        except Exception as exc:
            self.shell_average_error = str(exc)
            return
        output_dir = self.outdir / "shell_averages"
        output_dir.mkdir(parents=True, exist_ok=True)
        tolerance = float(self.config.get("b0range", 60))
        commands = []
        select = shutil.which("fslselectvols")
        maths = shutil.which("fslmaths")
        for shell, indices in sorted(_shell_groups(values, tolerance).items()):
            mean_path = output_dir / f"b{shell}_mean.nii.gz"
            self.shell_average_specs.append((shell, indices, mean_path))
            if mean_path.is_file():
                continue
            if not select or not maths:
                self.shell_average_error = "fslselectvols or fslmaths was not found on PATH."
                continue
            selected = output_dir / f".b{shell}_selected.nii.gz"
            self.shell_average_temps.append(selected)
            commands.append(shlex.join([select, "-i", str(data_path), "-o", str(selected),
                                         "--vols=" + ",".join(map(str, indices))]))
            commands.append(shlex.join([maths, str(selected), "-Tmean", str(mean_path)]))
        if commands:
            self.shell_average_process = subprocess.Popen(
                ["bash", "-c", " && ".join(commands)], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)

    def _final_shell_averages(self):
        section = self.section(
            "final-shells", "Final distortion-corrected shell averages",
            "Shell averages are calculated only from the final denoised, motion- and "
            "distortion-corrected data. FSL means are saved in QA/shell_averages for "
            "reuse; raw acquisitions are not averaged for QA.")
        if self.shell_average_process:
            _, stderr = self.shell_average_process.communicate()
            if self.shell_average_process.returncode:
                self.shell_average_error = stderr.strip() or "FSL shell averaging failed."
        for temporary in self.shell_average_temps:
            temporary.unlink(missing_ok=True)
        if self.shell_average_error:
            self.issue("warning", "Final shell averaging unavailable", self.shell_average_error)
        if not self.shell_average_specs:
            section["notes"].append(self.shell_average_error or "No diffusion shells were found.")
            return
        for shell, indices, mean_path in self.shell_average_specs:
            if not mean_path.is_file():
                section["notes"].append(f"The b={shell} mean was not generated.")
                continue
            def render(path, theme, shell=shell, mean_path=mean_path):
                average = _volume(mean_path)[0]
                _montage([(f"b={shell} mean", average)],
                         f"Final corrected shell average: b={shell}", path,
                         vmax=_scale((average,)), theme=theme)
            self.figure(
                section, f"final_shell_b{shell}", f"Final b={shell} shell average",
                f"Mean of {len(indices)} final corrected volumes in the b={shell} shell.",
                render, (mean_path,))

    def _signal_uniformity(self):
        section = self.section(
            "signal-uniformity", "Spatial signal uniformity",
            "A broad log-domain smooth field is estimated independently for each final "
            "shell average within the pipeline brain mask. This is descriptive dMRI "
            "signal QA and does not assign the variation to a specific physical cause.")
        mask = self._source("uniformity_mask", self.supplied["mask_file"],
                            self.root / "data" / "nodif_brain_mask.nii.gz")
        if not mask:
            section["notes"].append("The pipeline brain mask was unavailable.")
            self.issue("warning", "Signal-uniformity QA unavailable",
                       "A pipeline-derived brain mask is required.")
            return
        output_dir = self.outdir / "signal_uniformity"
        for shell, _, mean_path in self.shell_average_specs:
            if not mean_path.is_file():
                continue
            try:
                metrics = process_signal_uniformity(mean_path, output_dir, mask, 30.0)
                product_dir = output_dir / mean_path.name.removesuffix(".nii.gz")
                field = product_dir / "smooth_signal_field.nii.gz"
                ratio = product_dir / "observed_expected_ratio.nii.gz"
                low = product_dir / "low_signal_score.nii.gz"
                self.figure(
                    section, f"signal_uniformity_b{shell}", f"b={shell} signal uniformity",
                    "Top: median-normalised smooth signal field. Bottom: observed/local-"
                    "expected ratio; spatially coherent low values merit review. "
                    f"Field 5th–95th percentiles: {metrics['field_p05_p50_p95'][0]:.2f}–"
                    f"{metrics['field_p05_p50_p95'][2]:.2f}.",
                    lambda path, theme, field=field, ratio=ratio, shell=shell:
                        self._uniformity_panel(field, ratio, shell, path, theme),
                    (mean_path, mask, field, ratio, low))
            except Exception as exc:
                self.issue("warning", f"Could not assess b={shell} signal uniformity", exc)
        if not section["figures"]:
            section["notes"].append("No completed shell average could be assessed.")

    @staticmethod
    def _uniformity_panel(field_path, ratio_path, shell, output, theme):
        field = _volume(field_path)[0]
        ratio = _volume(ratio_path)[0]
        with plt.rc_context(_figure_rc(theme)):
            fig, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
            fig.suptitle(f"Spatial signal uniformity: b={shell}", fontsize=18,
                         weight="semibold")
            for row, (label, data, cmap, limits) in enumerate((
                    ("Smooth field", field, "coolwarm", (0.5, 1.5)),
                    ("Observed / expected", ratio, "coolwarm", (0.5, 1.5)))):
                artist = None
                for column, view in enumerate(_views(data)):
                    artist = axes[row, column].imshow(
                        view, cmap=cmap, vmin=limits[0], vmax=limits[1],
                        interpolation="nearest")
                    axes[row, column].set_title(
                        ("Sagittal", "Coronal", "Axial")[column] if row == 0 else "")
                    axes[row, column].set_ylabel(label, weight="semibold")
                    axes[row, column].axis("off")
                colourbar = fig.colorbar(artist, ax=axes[row, :], fraction=.018, pad=.012)
                colourbar.set_label("Relative signal (median = 1)")
            fig.savefig(output, dpi=170, facecolor=THEMES[theme]["bg"])
            plt.close(fig)

    @staticmethod
    def _denoise_panel(raw, den, indices, title, path, theme):
        if len(indices) == 1:
            before = _volume(raw, indices[0])[0]
            after = _volume(den, indices[0])[0]
        else:
            before = _mean_volume(raw, indices)
            after = _mean_volume(den, indices)
        if before.shape != after.shape:
            raise ValueError(f"Different display grids: {before.shape} and {after.shape}")
        _montage([("Raw", before), ("Denoised", after)], title, path,
                 vmax=_scale((before, after)), theme=theme)

    def _topup(self):
        section = self.section(
            "topup", "Susceptibility distortion correction",
            "The opposite-PE b0 pair passed to TOPUP is compared with the HiFi b0 "
            "written by applytopup. Assess geometric agreement and residual signal loss, "
            "particularly near frontal and temporal regions.")
        before = self._source("topup_input", self.root / "topup" / "Pos_Neg_b0.nii.gz")
        after = self._source("topup_hifib0", self.root / "topup" / "hifib0.nii.gz")
        if not before or not after:
            section["notes"].append("Pos_Neg_b0 or hifib0 is missing; TOPUP visual comparison skipped.")
            self.issue("warning", "TOPUP comparison incomplete",
                       "Expected topup/Pos_Neg_b0.nii.gz and topup/hifib0.nii.gz.")
            return

        def render(path, theme):
            input_img = nib.load(str(before))
            rows = []
            for index in range(min(2, input_img.shape[3] if len(input_img.shape) == 4 else 1)):
                direction = "Positive PE" if index == 0 else "Negative PE"
                rows.append((f"Before - {direction}", _volume(before, index)[0]))
            corrected = nib.load(str(after))
            corrected_data = (_mean_volume(after, range(corrected.shape[3]))
                              if len(corrected.shape) == 4 else _volume(after)[0])
            rows.append(("After - HiFi b0", corrected_data))
            _montage(rows, "TOPUP: opposite-PE b0s and corrected HiFi b0", path,
                     vmax=_scale([row[1] for row in rows]), theme=theme)

        self.figure(section, "topup_pre_post", "TOPUP before / after",
                    "First positive- and negative-PE b0 inputs, followed by the corrected "
                    "HiFi b0. Differences between opposite-PE images are expected before "
                    "correction; inspect whether anatomy becomes more consistent afterward.",
                    render, (before, after))

    def _eddy(self):
        section = self.section(
            "eddy", "EDDY and QUAD quality metrics",
            "The b0 panel is voxelwise mean/std across b0s; diffusion CNR panels "
            "summarise directional contrast relative to model residuals. Motion and "
            "slice-outlier timecourses are aligned to the input-volume index. Run "
            "transitions are marked separately from within-run motion.")
        qc_path = self._source("eddy_qc_json", self.supplied["eddy_qc_json"],
                               self.root / "eddy" / "eddy.qc" / "qc.json")
        if not qc_path:
            self.issue("warning", "EDDY QUAD QC missing", "eddy/eddy.qc/qc.json was not found.")
            section["notes"].append("EDDY QUAD QC files were not found.")
            return
        try:
            qc = json.loads(qc_path.read_text())
        except Exception as exc:
            self.issue("error", "Cannot parse EDDY QUAD QC", exc)
            return
        self.qc = qc
        values = qc.get("qc_cnr_avg") or []
        # QUAD lists b=0 in data_unique_bvals, while qc_cnr_avg uses its first
        # entry for b0 SNR. Only nonzero shells should therefore receive CNR
        # labels; otherwise every diffusion shell is shifted by one label.
        shells = [shell for shell in (qc.get("data_unique_bvals") or [])
                  if float(shell) > 0]
        if values:
            signal_maps = tuple(qc_path.parent / f"cnr{i:04d}.nii.gz"
                                for i in range(len(values)))
            self.themed_figure(
                section, "eddy_signal", "b0 SNR and shell CNR",
                "QUAD b0 SNR (first panel) and CNR per nonzero shell. The b0 SNR "
                "map can look speckled when few b0 repetitions make the voxelwise "
                "standard-deviation estimate unstable; examine averaged b0s as well.",
                lambda path, theme: self._eddy_signal_panel(qc, qc_path.parent, path, theme),
                (qc_path, *signal_maps))
        else:
            section["notes"].append("QUAD SNR/CNR values were absent from qc.json.")
        eddy_dir = self.root / "eddy"
        motion = self._source("eddy_motion", eddy_dir / "eddy_unwarped_images.eddy_movement_rms")
        outliers = self._source("eddy_outliers", eddy_dir / "eddy_unwarped_images.eddy_outlier_map")
        bvals = self._source("eddy_bvals", eddy_dir / "Pos_Neg.bvals")
        if motion and outliers and bvals:
            self.themed_figure(
                section, "eddy_motion_outliers", "Motion and slice outliers",
                "Absolute RMS is relative to the first volume; relative RMS is "
                "relative to the preceding volume. Transition peaks at an AP/PA "
                "or run boundary should be distinguished from sustained within-run "
                "movement. Outlier bars show slices flagged by EDDY.",
                lambda path, theme: self._eddy_motion_panel(qc, eddy_dir, path, theme),
                (motion, outliers, bvals, eddy_dir / "series.txt", eddy_dir / "index.txt"))
        else:
            section["notes"].append("Per-volume movement/outlier files were incomplete.")

    @staticmethod
    def _eddy_signal_panel(qc, qc_dir, output, theme):
        values = qc.get("qc_cnr_avg") or []
        shells = [shell for shell in (qc.get("data_unique_bvals") or [])
                  if float(shell) > 0]
        if not values:
            raise ValueError("qc_cnr_avg missing")
        t = THEMES[theme]
        labels = ["b0 SNR"] + [f"b={shell} CNR" for shell in shells]
        rows = len(values)
        numeric_maps = [qc_dir / f"cnr{i:04d}.nii.gz" for i in range(rows)]
        use_numeric_maps = all(path.is_file() for path in numeric_maps)
        with plt.rc_context(_figure_rc(theme)):
            if use_numeric_maps:
                fig, axes = plt.subplots(rows, 3, figsize=(15, max(6, rows * 3.45)),
                                         squeeze=False, layout="constrained")
            else:
                compact_rows = (rows + 1) // 2
                fig, axes = plt.subplots(compact_rows, 2,
                                         figsize=(15, max(5.5, compact_rows * 4.2)),
                                         squeeze=False, layout="constrained")
            fig.suptitle("EDDY QUAD - SNR and CNR", fontsize=19, weight="semibold")
            for i in range(rows):
                title = labels[i] if i < len(labels) else f"shell {i}"
                if use_numeric_maps:
                    nii = numeric_maps[i]
                    data = _volume(nii)[0]
                    finite = data[np.isfinite(data) & (data >= 0)]
                    vmax = float(np.percentile(finite, 99)) if finite.size else 1
                    artist = None
                    for column, view in enumerate(_views(data)):
                        artist = axes[i, column].imshow(
                            view, cmap="viridis", vmin=0, vmax=vmax,
                            interpolation="nearest")
                        axes[i, column].set_title(
                            ("Sagittal", "Coronal", "Axial")[column] if i == 0 else "")
                        axes[i, column].axis("off")
                    axes[i, 0].set_ylabel(f"{title}\nmean {values[i]:.2f}",
                                          weight="semibold", color=t["fg"])
                    colourbar = fig.colorbar(artist, ax=axes[i, :], fraction=.018, pad=.012)
                    colourbar.set_label("SNR" if i == 0 else "CNR")
                else:
                    png = qc_dir / f"cnr{i:04d}.nii.gz.png"
                    axis = axes.flat[i]
                    axis.axis("off")
                    axis.set_title(f"{title}: {values[i]:.2f}", fontsize=14,
                                   weight="semibold", color=t["fg"])
                    if png.is_file():
                        axis.imshow(plt.imread(png), interpolation="nearest")
                    else:
                        axis.text(0.5, 0.5, "Spatial map unavailable", ha="center",
                                  va="center", transform=axis.transAxes, color=t["fg"])
            if not use_numeric_maps:
                for axis in axes.flat[rows:]:
                    axis.axis("off")
            fig.savefig(output, dpi=175, facecolor=t["bg"])
            plt.close(fig)

    @staticmethod
    def _eddy_motion_panel(qc, eddy_dir, output, theme):
        prefix = eddy_dir / "eddy_unwarped_images"
        motion = np.atleast_2d(np.loadtxt(str(prefix) + ".eddy_movement_rms"))
        outliers = np.atleast_2d(np.loadtxt(str(prefix) + ".eddy_outlier_map", skiprows=1))
        bvals = np.atleast_1d(np.loadtxt(eddy_dir / "Pos_Neg.bvals")).ravel()
        n = len(motion)
        if motion.shape[1] < 2 or outliers.shape[0] != n or len(bvals) != n:
            raise ValueError("Motion, outlier map and bvals have inconsistent volume counts")
        pe_path, series_path = eddy_dir / "index.txt", eddy_dir / "series.txt"
        pe = (np.atleast_1d(np.loadtxt(pe_path, dtype=int)).ravel() if pe_path.is_file()
              else np.ones(n, dtype=int))
        series = (np.atleast_1d(np.loadtxt(series_path, dtype=int)).ravel()
                  if series_path.is_file() else pe.copy())
        if len(pe) != n or len(series) != n:
            raise ValueError("index.txt or series.txt does not match EDDY volume count")
        boundaries = [i for i in range(1, n)
                      if series[i] != series[i - 1] or pe[i] != pe[i - 1]]
        starts = [0] + boundaries
        blocks = []
        counts = defaultdict(int)
        for start, end in zip(starts, starts[1:] + [n]):
            code = int(pe[start])
            counts[code] += 1
            tag = {1: "AP", 2: "PA"}.get(code, f"PE {code}")
            blocks.append((start, end, f"{tag} {counts[code]}"))
        x = np.arange(n)
        relative = motion[:, 1]
        dropout = 100 * np.count_nonzero(outliers, axis=1) / outliers.shape[1]
        boundary_near = np.array([any(abs(i - b) <= 1 for b in boundaries)
                                  for i in range(n)])
        high = relative >= 1.0
        t = THEMES[theme]
        with plt.rc_context(_figure_rc(theme)):
            fig = plt.figure(figsize=(17, 10), layout="constrained")
            grid = fig.add_gridspec(4, 2, width_ratios=[3.4, 1.3],
                                    height_ratios=[1, 1, 1.1, 0.7])
            axes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
            for axis in axes:
                axis.grid(axis="y", color=t["grid"], linewidth=0.7)
                axis.set_axisbelow(True)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                for start, end, _ in blocks[1::2]:
                    axis.axvspan(start - 0.5, end - 0.5, color=t["blue"], alpha=0.035)
                for boundary in boundaries:
                    changed = pe[boundary] != pe[boundary - 1]
                    axis.axvline(boundary - 0.5, color=t["amber"] if changed else t["fg"],
                                 linestyle=":" if changed else "--",
                                 linewidth=1.6 if changed else 1, alpha=0.8)
            axes[0].plot(x, motion[:, 0], color=t["blue"], linewidth=1.25)
            axes[0].set_ylabel("Absolute RMS (mm)")
            axes[0].set_ylim(top=max(1.0, float(np.max(motion[:, 0])) * 1.28))
            axes[0].set_title("Head position and slice outliers across runs",
                              fontsize=18, weight="semibold")
            for start, end, label in blocks:
                axes[0].text((start + end - 1) / 2, 0.98,
                             f"{label}  ·  {start}-{end - 1}",
                             transform=axes[0].get_xaxis_transform(), ha="center",
                             va="top", fontsize=9, weight="semibold", color=t["fg"],
                             bbox={"facecolor": t["bg"], "alpha": 0.88,
                                   "edgecolor": "none", "pad": 2})
            axes[1].plot(x, relative, color=t["blue"], linewidth=1.0)
            axes[1].axhline(1.0, color=t["red"], linestyle=":", linewidth=0.9)
            for select, color, marker, label in (
                (high & ~boundary_near, t["red"], "o", "Away from boundary ≥1 mm"),
                (high & boundary_near, t["amber"], "D", "At/near run boundary")):
                if np.any(select):
                    axes[1].scatter(x[select], relative[select], color=color,
                                    marker=marker, s=30, zorder=4, label=label)
            axes[1].set_ylabel("Relative RMS (mm)")
            if np.any(high):
                axes[1].legend(frameon=False, labelcolor=t["fg"], loc="upper right")
            colors = [t["red"] if value >= 5 else t["amber"] if value > 0 else t["grid"]
                      for value in dropout]
            axes[2].bar(x, dropout, width=0.86, color=colors)
            axes[2].axhline(5, color=t["red"], linestyle=":", linewidth=0.9)
            axes[2].set_ylabel("Outlier slices (%)")
            for index in np.argsort(dropout)[-5:]:
                if dropout[index] > 0:
                    axes[2].annotate(str(index), (index, dropout[index]),
                                     xytext=(0, 4), textcoords="offset points",
                                     ha="center", fontsize=8, color=t["fg"])
            for code in sorted(set(pe)):
                selected = pe == code
                axes[3].scatter(x[selected], bvals[selected], s=9,
                                color=t["blue"] if code == min(pe) else t["amber"],
                                label={1: "AP", 2: "PA"}.get(int(code), f"PE {code}"))
            axes[3].set_ylabel("b-value")
            axes[3].set_xlabel("Volume (0-based)")
            axes[3].legend(frameon=False, labelcolor=t["fg"], loc="upper right")
            axes[3].set_xlim(-1, n)
            summary_ax = fig.add_subplot(grid[:2, 1])
            summary_ax.axis("off")
            within = [float(relative[i]) for i in range(1, n) if i not in boundaries]
            metrics = [
                f"Mean abs RMS      {qc.get('qc_mot_abs', float('nan')):.2f} mm",
                f"Mean rel RMS      {qc.get('qc_mot_rel', float('nan')):.2f} mm (all)",
                f"Within-run rel    {np.mean(within):.2f} mm" if within else "Within-run rel    n/a",
                f"Run starts        {', '.join(map(str, boundaries)) or 'none'}",
                f"Boundary peaks    {int(np.sum(high & boundary_near))}",
                f"Other ≥1 mm       {int(np.sum(high & ~boundary_near))}",
                f"Outlier slices    {100 * qc.get('qc_outliers_tot', 0):.2f}%",
                f"Volumes ≥5%      {int(np.sum(dropout >= 5))}",
                f"repol / mporder   {qc.get('eddy_input', {}).get('repol', '?')} / "
                f"{qc.get('eddy_input', {}).get('mporder', '?')}",
            ]
            summary_ax.set_title("At a glance", loc="left", fontsize=15, weight="semibold")
            summary_ax.text(0, 0.95, "\n".join(metrics), va="top", fontsize=9,
                            fontfamily="DejaVu Sans Mono", linespacing=1.55)
            shell_ax = fig.add_subplot(grid[2:, 1])
            shell_values = [float(v) for v in qc.get("data_unique_bvals", []) if float(v) > 0]
            if not shell_values:
                shell_values = sorted(set(int(round(v / 100) * 100)
                                          for v in bvals if v > 60))
            shell_pct = []
            for shell in shell_values:
                selected = (np.abs(bvals - shell) <= 60) & (bvals > 60)
                shell_pct.append(float(np.mean(dropout[selected])) if np.any(selected) else 0)
            shell_ax.barh(range(len(shell_values)), shell_pct, color=t["amber"])
            shell_ax.set_yticks(range(len(shell_values)), [f"b={v:g}" for v in shell_values])
            shell_ax.invert_yaxis()
            shell_ax.set_title("Outlier burden by shell", loc="left", fontsize=13)
            shell_ax.set_xlabel("Outlier slices (%)")
            shell_ax.grid(axis="x", color=t["grid"], linewidth=0.7)
            shell_ax.set_axisbelow(True)
            fig.savefig(output, dpi=170, facecolor=t["bg"])
            plt.close(fig)

    def _mask(self):
        section = self.section(
            "mask", "Native-space brain masking",
            "An EDDY-corrected b0 volume is shown with the final native-space "
            "brain mask outlined in red. Check for excluded cortex/cerebellum and "
            "included skull, eyes or residual ghosting.")
        data_path = self._source("corrected_data", self.supplied["data_4d"],
                                  self.root / "data" / "data.nii.gz")
        mask_path = self._source("native_mask", self.supplied["mask_file"],
                                  self.root / "data" / "nodif_brain_mask.nii.gz")
        bvals_path = self._source("corrected_bvals", self.supplied["bvals"],
                                   self.root / "data" / "bvals")
        if not data_path or not mask_path:
            self.issue("error", "Brain-mask QA unavailable",
                       "Expected corrected data/data.nii.gz and data/nodif_brain_mask.nii.gz.")
            section["notes"].append("Corrected data or native mask is missing.")
            return

        def render(path, theme):
            values = _bvals(bvals_path)
            b0range = float(self.config.get("b0range", 60))
            selected = np.flatnonzero(values <= b0range) if values is not None else []
            index = int(selected[0]) if len(selected) else 0
            corrected_img = nib.load(str(data_path))
            corrected = _volume(data_path, index)[0]
            mask_img = _match_grid(nib.load(str(mask_path)), corrected_img, order=0)
            mask = _canon(np.asarray(mask_img.dataobj), mask_img.affine)
            if corrected.shape != mask.shape:
                raise ValueError(f"Corrected data {corrected.shape} and mask {mask.shape} differ")
            _montage([("Corrected b0 + mask", corrected)],
                     f"Native-space mask on corrected b0 (volume {index})", path,
                     vmax=_scale((corrected,)), masks=[mask], theme=theme)

        self.figure(section, "native_mask_overlay", "Corrected b0 and final mask",
                    "Red contour is the mask used for downstream diffusion modelling; "
                    "the background is a distortion-corrected native-space b0.",
                    render, (data_path, mask_path, bvals_path))

    def _diffusion_maps(self):
        section = self.section(
            "tensor", "Diffusion tensor fit",
            "FA and MD are fitted from the configured lower shell. Direction-encoded "
            "V1 is shown with brightness modulated by FA, which helps expose "
            "orientation discontinuities and registration or gradient errors.")
        fa = self._source("fa_native", self.supplied["fa_img"],
                          self.root / "dtifit" / "dti_FA.nii.gz")
        md = self._source("md_native", self.root / "dtifit" / "dti_MD.nii.gz")
        v1 = self._source("v1_native", self.supplied["v1_img"],
                          self.root / "dtifit" / "dti_V1.nii.gz")
        if not fa or not md or not v1:
            self.issue("warning", "Tensor-map QA incomplete",
                       "Expected dtifit/dti_FA.nii.gz, dti_MD.nii.gz and dti_V1.nii.gz.")
            section["notes"].append("FA, MD or V1 was missing; tensor overview skipped.")
            return

        def render(path, theme):
            fa_img = nib.load(str(fa))
            md_img = _match_grid(nib.load(str(md)), fa_img)
            v1_img = nib.load(str(v1))
            if (v1_img.shape[:3] != fa_img.shape[:3]
                    or not np.allclose(v1_img.affine, fa_img.affine)):
                raise ValueError("V1 and FA are on different grids; vector resampling needs explicit reorientation")
            fa_data = _canon(np.asarray(fa_img.dataobj), fa_img.affine)
            md_data = _canon(np.asarray(md_img.dataobj), md_img.affine)
            v1_data = _canon(np.asarray(v1_img.dataobj), v1_img.affine)
            if v1_data.ndim != 4 or v1_data.shape[-1] < 3:
                raise ValueError(f"Expected three-vector V1 image, got {v1_data.shape}")
            rgb = np.clip(np.abs(v1_data[..., :3]) *
                          np.clip(fa_data[..., None], 0, 1), 0, 1)
            max_md = max(0.003, float(np.nanpercentile(md_data[np.isfinite(md_data)], 99)))
            with plt.rc_context(_figure_rc(theme)):
                fig, axes = plt.subplots(3, 3, figsize=(15, 10.5), layout="constrained")
                fig.suptitle("DTIFIT: FA, MD and FA-modulated V1", fontsize=19,
                             weight="semibold")
                rows = [("FA", fa_data, "gray", 1.0),
                        ("MD (mm²/s)", md_data, "gray", max_md),
                        ("V1 × FA", rgb, None, None)]
                for i, (label, data, cmap, maximum) in enumerate(rows):
                    for j, plane in enumerate(_views(data)):
                        axis = axes[i, j]
                        axis.imshow(plane, cmap=cmap, vmin=0 if cmap else None,
                                    vmax=maximum, interpolation="nearest")
                        axis.set_title(("Sagittal", "Coronal", "Axial")[j] if i == 0 else "")
                        axis.set_ylabel(label, fontsize=12, weight="semibold")
                        axis.set_xticks([])
                        axis.set_yticks([])
                        for side in axis.spines.values():
                            side.set_visible(False)
                fig.savefig(path, dpi=170, facecolor=THEMES[theme]["bg"])
                plt.close(fig)

        self.figure(section, "tensor_maps", "FA, MD and V1×FA",
                    "FA is displayed from 0 to 1; MD is in mm²/s with a common "
                    "display scale across its three planes. V1 color encodes principal "
                    "direction and FA modulates brightness.", render, (fa, md, v1))

    def _registration(self):
        threshold = 0.1 if self.mode == "hum" else 0.2
        section = self.section(
            "registration", "Standard-space registration",
            "The subject FA transformed to standard space is shown with a red "
            f"outline from standard-space white-matter FA (>{threshold:g}). The outline must "
            "come from the template, not from the subject FA itself.")
        transformed = self._source("fa_standard", self.root / "stdreg" / "FA_in_STD.nii.gz")
        reference = self._source("fa_standard_template", self.supplied["nmt_fa_img"])
        if not transformed or not reference:
            self.issue("warning", "Registration overlay unavailable",
                       "Subject FA_in_STD or the human/NHP standard-space FA reference is missing.")
            section["notes"].append("Standard-space subject FA or reference FA was unavailable.")
            return

        def render(path, theme):
            self._registration_fsleyes(transformed, reference, path, threshold)

        self.figure(section, "standard_fa_overlay", "FA in standard space",
                    f"The red contour is template white matter (FA > {threshold:g}). "
                    "Inspect cortical, callosal and cerebellar alignment.",
                    render, (transformed, reference))

    @staticmethod
    def _registration_fsleyes(subject, reference, output, threshold):
        """Render subject FA with a true template-derived FA outline."""
        executable = shutil.which("fsleyes")
        if not executable:
            raise FileNotFoundError("fsleyes was not found on PATH")
        ref_img = nib.load(str(reference))
        ref_data = np.asarray(ref_img.dataobj)
        coords = np.argwhere(np.isfinite(ref_data) & (ref_data > threshold))
        centre = coords.mean(axis=0) if coords.size else np.asarray(ref_img.shape[:3]) / 2
        world = nib.affines.apply_affine(ref_img.affine, centre)
        if coords.size:
            lower, upper = coords.min(axis=0), coords.max(axis=0)
            corners = np.asarray([
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ])
            world_x = nib.affines.apply_affine(ref_img.affine, corners)[:, 0]
            lateral = float(world_x.min())
            midline = float((world_x.min() + world_x.max()) / 2)
            world[0] = (lateral + midline) / 2
        command = [
            executable, "render", "--outfile", str(output), "--size", "3000", "900",
            "--scene", "ortho", "--worldLoc", *(f"{value:.8f}" for value in world),
            "--displaySpace", str(reference),
            "--xzoom", "100", "--yzoom", "100", "--zzoom", "100",
            "--hideLabels", "--showLocation", "no", "--layout", "horizontal",
            "--hideCursor", "--bgColour", "0", "0", "0", "--fgColour", "1", "1", "1",
            str(subject), "--name", "FA_in_STD", "--overlayType", "volume",
            "--alpha", "100", "--cmap", "greyscale", "--displayRange", "0", "1",
            "--interpolation", "none", "--volume", "0",
            str(reference), "--name", "Template_FA", "--overlayType", "mask",
            "--alpha", "100", "--maskColour", "1", "0", "0", "--maskInvert",
            "--threshold", "0", str(threshold), "--outline", "--outlineWidth", "3",
            "--interpolation", "none", "--volume", "0",
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)
        if not Path(output).is_file() or not Path(output).stat().st_size:
            raise RuntimeError("FSLeyes completed without creating the registration PNG")

    def _xtract(self):
        section = self.section(
            "xtract", "XTRACT tractography overview",
            "Standard-space tract-density visualisations show whether major "
            "pathways are present and plausibly located. This is a spatial check, "
            "not a quantitative tract-reliability score.")
        xdir = Path(self.supplied["xtract_dir"] or self.root / "xtract")
        tracts = sorted((xdir / "tracts").glob("*/densityNorm.nii.gz"))
        self.sources["xtract_tract_count"] = len(tracts)
        waytotals = {}
        missing_waytotals = []
        for tract_dir in sorted((xdir / "tracts").glob("*")):
            if not tract_dir.is_dir():
                continue
            path = tract_dir / "waytotal"
            try:
                waytotals[tract_dir.name] = float(path.read_text().split()[0])
            except Exception:
                missing_waytotals.append(tract_dir.name)
        self.sources["xtract_waytotal_count"] = len(waytotals)
        failed = sorted(name for name, value in waytotals.items() if value == 0)
        low = sorted(((name, value) for name, value in waytotals.items()
                      if 0 < value < 1000), key=lambda item: item[1])
        if failed:
            self.issue("error", "XTRACT tract failure",
                       "Zero waytotal: " + ", ".join(failed))
        if low:
            self.issue("warning", "Very low XTRACT waytotals",
                       ", ".join(f"{name}={value:g}" for name, value in low))
        if missing_waytotals:
            self.issue("warning", "XTRACT waytotals incomplete",
                       "Missing or unreadable: " + ", ".join(missing_waytotals))
        if waytotals:
            self.figure(section, "xtract_waytotals", "XTRACT waytotal check",
                        "All tract waytotals on a logarithmic scale. Zero means a failed "
                        "tract; positive values below 1,000 are flagged as very low.",
                        lambda path, theme: self._xtract_waytotal_panel(waytotals, path, theme),
                        [xdir / "tracts" / name / "waytotal" for name in waytotals])
        if tracts:
            tract_map = {path.parent.name: path for path in tracts}
            for view in ("sagittal", "coronal", "axial"):
                selected = [tract_map[name] for name, *_ in self._xtract_view_spec(view)
                            if name in tract_map]
                missing = [name for name, *_ in self._xtract_view_spec(view)
                           if name not in tract_map]
                if missing:
                    section["notes"].append(
                        f"{view.title()} XTRACT panel omitted missing tracts: {', '.join(missing)}.")
                self.figure(section, f"xtract_{view}", f"XTRACT {view} view",
                            "Targeted standard-space tract-density MIPs on the standard FA "
                            "template; the slice location is derived from template anatomy.",
                            lambda path, theme, view=view, tract_map=tract_map:
                                self._xtract_fsleyes(view, tract_map, path),
                            [self.supplied["nmt_fa_img"], *selected])
        elif self.run_mode == "preproc_only":
            section["notes"].append("XTRACT was not requested in this preprocessing-only run.")
        else:
            self.issue("warning", "XTRACT outputs unavailable",
                       "No tract-density files or viewer PNGs were found.")
            section["notes"].append("No XTRACT tract-density files or viewer PNGs were found.")

    @staticmethod
    def _xtract_waytotal_panel(waytotals, output, theme):
        ordered = sorted(waytotals.items(), key=lambda item: item[1])
        names = [item[0] for item in ordered]
        values = np.asarray([item[1] for item in ordered], dtype=float)
        shown = np.maximum(values, 0.5)
        t = THEMES[theme]
        with plt.rc_context(_figure_rc(theme)):
            fig, axis = plt.subplots(figsize=(15, max(8, len(names) * 0.22)),
                                     layout="constrained")
            colors = [t["red"] if value == 0 else t["amber"] if value < 1000
                      else t["blue"] for value in values]
            axis.barh(range(len(names)), shown, color=colors)
            axis.set_yticks(range(len(names)), names, fontsize=7)
            axis.invert_yaxis()
            axis.set_xscale("log")
            axis.axvline(1000, color=t["amber"], linestyle="--", linewidth=1)
            axis.set_xlabel("waytotal (log scale; zeros displayed at 0.5)")
            axis.set_title("XTRACT waytotal check", loc="left", fontsize=18,
                           weight="semibold")
            axis.grid(axis="x", color=t["grid"], linewidth=0.7)
            fig.savefig(output, dpi=170, facecolor=t["bg"])
            plt.close(fig)

    @staticmethod
    def _xtract_view_spec(view):
        return {
            "sagittal": (
                ("mcp", "red", .001, .05), ("slf1_r", "green", .005, .03),
                ("cbd_r", "blue", .003, .02), ("cbp_r", "yellow", .01, .2),
                ("cst_r", "fsleyes_hot", .003, .07),
                ("cbt_r", "blue-lightblue", .004, .1),
                ("fx_r", "fsleyes_copper", .005, .15),
                ("vof_r", "fsleyes_cool", .005, .05)),
            "coronal": (
                ("fx_r", "fsleyes_copper", .005, .1),
                ("fx_l", "fsleyes_copper", .005, .1),
                ("cst_r", "fsleyes_hot", .001, .05),
                ("cst_l", "fsleyes_hot", .001, .05),
                ("fa_l", "fsleyes_cool", .001, .02),
                ("fa_r", "fsleyes_cool", .001, .02),
                ("str_r", "blue", .004, .025), ("str_l", "blue", .004, .025),
                ("ar_r", "yellow", .02, .1), ("ar_l", "yellow", .02, .1)),
            "axial": (
                ("fma", "yellow", .005, .04), ("fmi", "yellow", .002, .01),
                ("or_l", "blue", .006, .05), ("or_r", "blue", .006, .05),
                ("ac", "green", .001, .1),
                ("uf_l", "blue-lightblue", .003, .02),
                ("uf_r", "blue-lightblue", .003, .02),
                ("atr_r", "fsleyes_cool", .005, .03),
                ("atr_l", "fsleyes_cool", .005, .03)),
        }[view]

    def _xtract_fsleyes(self, view, tract_map, output):
        executable = shutil.which("fsleyes")
        if not executable:
            raise FileNotFoundError("fsleyes was not found on PATH")
        reference = self._source("xtract_standard_template", self.supplied["nmt_fa_img"])
        if not reference:
            raise FileNotFoundError("The standard-space FA template was unavailable")
        ref_img = nib.load(str(reference))
        ref_data = np.asarray(ref_img.dataobj)
        coords = np.argwhere(np.isfinite(ref_data) & (ref_data > 0.1))
        centre = coords.mean(axis=0) if coords.size else np.asarray(ref_img.shape[:3]) / 2
        world = nib.affines.apply_affine(ref_img.affine, centre)
        if view == "sagittal" and coords.size:
            world_points = nib.affines.apply_affine(ref_img.affine, coords)
            world[0] = (world[0] + float(world_points[:, 0].max())) / 2
        hidden = {"sagittal": ("--hidey", "--hidez"),
                  "coronal": ("--hidex", "--hidez"),
                  "axial": ("--hidex", "--hidey")}[view]
        command = [executable, "render", "--outfile", str(output), "--size", "1400", "1000",
                   "--scene", "ortho", "--worldLoc", *(f"{value:.8f}" for value in world),
                   "--displaySpace", str(reference), "--xzoom", "100", "--yzoom", "100",
                   "--zzoom", "100", *hidden, "--hideLabels", "--showLocation", "no",
                   "--hideCursor", str(reference),
                   "--name", "Standard_FA", "--overlayType", "volume", "--cmap",
                   "greyscale", "--brightness", "58", "--contrast", "65",
                   "--displayRange", "0", "0.5", "--interpolation", "spline"]
        for name, cmap, low, high in self._xtract_view_spec(view):
            tract = tract_map.get(name)
            if tract:
                command += [str(tract), "--name", name, "--overlayType", "mip",
                            "--cmap", cmap, "--displayRange", str(low), str(high),
                            "--interpolation", "spline"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)

    def _write_reports(self):
        html_path = self.outdir / "CMC_invivo_QA.html"
        pdf_light = self.outdir / "CMC_invivo_QA.pdf"
        pdf_dark = self.outdir / "CMC_invivo_QA_dark.pdf"
        manifest_path = self.outdir / "QA_manifest.json"
        data_path = self.sources.get("corrected_data")
        data_info = {}
        if data_path:
            try:
                image = nib.load(data_path)
                data_info = {"shape": list(image.shape),
                             "voxel_size_mm": [round(float(v), 3)
                                               for v in image.header.get_zooms()[:3]]}
            except Exception as exc:
                self.issue("warning", "Cannot read corrected-data header", exc)
        config_summary = {key: self.config.get(key) for key in (
            "mode", "lower_b", "b0range", "pifactor", "total_readout_time",
            "denoise", "run_gibbs", "run_n4", "stdreg_method", "eddy_extra_args")
            if key in self.config}
        metadata = {
            "report_version": self.REPORT_VERSION,
            "generated": self.timestamp, "dmri_root": str(self.root),
            "run_mode": self.run_mode, "species_mode": self.mode,
            "config": config_summary, "data": data_info,
        }
        milestones = [self.root / "topup" / "hifib0.nii.gz",
                      self.root / "eddy" / "eddy_unwarped_images.nii.gz",
                      self.root / "dtifit" / "dti_FA.nii.gz",
                      self.root / "stdreg" / "FA_in_STD.nii.gz"]
        dates = [(p.name, dt.datetime.fromtimestamp(p.stat().st_mtime).astimezone())
                 for p in milestones if p.is_file()]
        metadata["output_timestamps"] = {key: value.isoformat(timespec="seconds")
                                         for key, value in dates}
        manifest = {"metadata": metadata, "issues": self.issues,
                    "processing_steps": self.steps, "sources": self.sources,
                    "sections": self.sections}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        html_path.write_text(self._html(metadata), encoding="utf-8")
        for theme, path in (("light", pdf_light), ("dark", pdf_dark)):
            try:
                self._pdf(path, theme, metadata)
            except Exception as exc:
                self.issue("warning", f"Could not export {theme} PDF", exc)
                if path.exists():
                    path.unlink()
        # Include any PDF-export warning in both portable metadata outputs.
        manifest["issues"] = self.issues
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        html_path.write_text(self._html(metadata), encoding="utf-8")
        paths = {"report_html": html_path, "manifest_json": manifest_path,
                 "figures_dir": self.figures}
        if pdf_light.is_file():
            paths["report_pdf"] = pdf_light
        if pdf_dark.is_file():
            paths["report_pdf_dark"] = pdf_dark
        return paths

    def _html(self, metadata):
        esc = lambda item: html.escape(str(item), quote=True)
        problems = sorted(self.issues, key=lambda item: {"error": 0, "warning": 1}.get(
            item["severity"], 2))
        if problems:
            alert_cards = "".join(
                f'<div class="alert {esc(item["severity"])}"><strong>{esc(item["title"])}</strong>'
                f'<p>{esc(item["detail"])}</p></div>' for item in problems)
        else:
            alert_cards = '<div class="alert ok"><strong>No automated issues detected</strong>' \
                          '<p>Visual inspection is still required.</p></div>'
        step_cards = "".join(
            f'<span class="step {esc(step["state"].replace(" ", "-"))}">'
            f'{esc(step["name"])} <small>{esc(step["state"])}</small></span>'
            for step in self.steps)
        meta_items = [
            ("Generated", metadata["generated"]), ("Mode", metadata["species_mode"].upper()),
            ("Run", metadata["run_mode"].replace("_", " ")),
            ("Latest output", max(metadata["output_timestamps"].values(), default="unavailable")),
            ("Data", " × ".join(map(str, metadata["data"].get("shape", []))) or "unavailable"),
            ("Voxel (mm)", " × ".join(map(str, metadata["data"].get("voxel_size_mm", [])))
             or "unavailable"),
        ]
        stat_cards = "".join(f'<div class="stat"><span>{esc(key)}</span>'
                             f'<strong>{esc(value)}</strong></div>' for key, value in meta_items)
        config_rows = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>"
                              for key, value in metadata["config"].items())
        date_rows = "".join(f"<tr><th>{esc(key)}</th><td>{esc(value)}</td></tr>"
                            for key, value in metadata["output_timestamps"].items())
        nav = "".join(f'<a href="#{esc(section["id"])}">{esc(section["title"])}</a>'
                      for section in self.sections)
        section_html = []
        for section in self.sections:
            cards = []
            for figure in section["figures"]:
                light = f'figures/{esc(figure["light"])}'
                dark = f'figures/{esc(figure["dark"])}'
                sources = "".join(f"<li>{esc(path)}</li>" for path in figure["sources"])
                launch = self._fsleyes_launch_command(figure["sources"])
                cards.append(
                    f'<article class="figure"><h3>{esc(figure["title"])}</h3>'
                    f'<img class="theme-light" src="{light}" alt="{esc(figure["title"])}">'
                    f'<img class="theme-dark" src="{dark}" alt="{esc(figure["title"])}">'
                    f'<p>{esc(figure["caption"])}</p>'
                    + (f'<details><summary>Source files</summary><ul>{sources}</ul></details>'
                       if sources else "")
                    + (f'<details class="launch"><summary>Launch interactive view</summary>'
                       f'<p>Copy into a terminal on the processing host:</p>'
                       f'<pre><code>{esc(launch)}</code></pre></details>' if launch else "")
                    + '</article>')
            notes = "".join(f'<p class="note">{esc(note)}</p>' for note in section["notes"])
            section_html.append(
                f'<section id="{esc(section["id"])}"><div class="section-head">'
                f'<span class="eyebrow">VISUAL QA</span><h2>{esc(section["title"])}</h2>'
                f'<p>{esc(section["intro"])}</p></div>{notes}'
                f'<div class="figure-grid">{"".join(cards)}</div></section>')
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CMC in-vivo dMRI QA · {esc(self.root.parent.name)}</title>
<style>
:root {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; color-scheme: light dark; }}
body {{ --bg:#f5f7fb; --paper:#ffffff; --text:#172033; --sub:#586579; --line:#dfe5ee;
        --accent:#3274a1; --error:#c43d56; --warn:#9d620a; --good:#087e68;
        margin:0; background:var(--bg); color:var(--text); line-height:1.48; }}
body[data-theme="dark"] {{ --bg:#0b1320; --paper:#111827; --text:#f3f4f6; --sub:#aeb9c9;
 --line:#324053; --accent:#60a5fa; --error:#fb7185; --warn:#fbbf24; --good:#34d399; }}
* {{ box-sizing:border-box; }} a {{ color:var(--accent); }} .page {{ max-width:1480px; margin:auto; padding:0 28px 80px; }}
header {{ border-bottom:1px solid var(--line); background:var(--paper); }}
.top {{ max-width:1480px; margin:auto; padding:18px 28px; display:flex; align-items:center; gap:25px; }}
.brand {{ font-size:14px; font-weight:800; letter-spacing:.16em; color:var(--accent); }}
.top nav {{ display:flex; gap:15px; flex:1; overflow:auto; white-space:nowrap; font-size:12px; }}
.top nav a {{ text-decoration:none; color:var(--sub); }} .top nav a:hover {{ color:var(--accent); }}
button {{ border:1px solid var(--line); background:var(--bg); color:var(--text); padding:8px 13px;
 border-radius:9px; cursor:pointer; font-weight:650; }}
.hero {{ padding:58px 0 28px; }} .eyebrow {{ font-size:11px; letter-spacing:.18em; font-weight:800;
 color:var(--accent); }} h1 {{ margin:8px 0; font-size:clamp(32px,4vw,54px); line-height:1.05; }}
.subtitle {{ color:var(--sub); max-width:760px; font-size:17px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:28px 0; }}
.stat, .figure, .alert, .config, .overview {{ background:var(--paper); border:1px solid var(--line);
 border-radius:14px; }} .stat {{ padding:17px 20px; }} .stat span {{ color:var(--sub); display:block; font-size:12px; }}
.stat strong {{ display:block; margin-top:5px; font-size:19px; }}
section {{ padding-top:45px; }} .section-head {{ max-width:850px; margin-bottom:18px; }}
h2 {{ font-size:28px; margin:6px 0 8px; }} .section-head p, .note {{ color:var(--sub); }}
.alerts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(310px,1fr)); gap:10px; }}
.alert {{ border-left:4px solid var(--accent); padding:16px 18px; }}
.alert.error {{ border-left-color:var(--error); }} .alert.warning {{ border-left-color:var(--warn); }}
.alert.ok {{ border-left-color:var(--good); }} .alert strong {{ display:block; }}
.alert p {{ margin:7px 0 0; color:var(--sub); font-size:13px; overflow-wrap:anywhere; }}
.overview, .config {{ padding:22px; }} .steps {{ display:flex; gap:8px; flex-wrap:wrap; }}
.step {{ padding:9px 11px; border:1px solid var(--line); border-radius:9px; font-weight:650; }}
.step small {{ display:block; color:var(--sub); font-weight:400; }}
.step.complete {{ border-color:var(--good); }} .step.missing {{ border-color:var(--warn); }}
.figure-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,590px),1fr)); gap:16px; }}
.figure {{ padding:16px; overflow:hidden; }} .figure h3 {{ font-size:18px; margin:0 0 12px; }}
.figure img {{ width:100%; height:auto; display:block; border-radius:7px; background:#111827; }}
.figure p {{ color:var(--sub); font-size:14px; }} .figure details {{ color:var(--sub); font-size:12px; }}
.figure li {{ overflow-wrap:anywhere; margin:4px 0; }} .note {{ background:var(--paper); border:1px dashed var(--line);
 padding:14px 18px; border-radius:10px; }} .theme-dark {{ display:none!important; }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--bg); border:1px solid var(--line);
 padding:11px; border-radius:8px; color:var(--text); user-select:all; }}
body[data-theme="dark"] .theme-dark {{ display:block!important; }}
body[data-theme="dark"] .theme-light {{ display:none!important; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th,td {{ padding:8px 10px;
 border-bottom:1px solid var(--line); text-align:left; }} th {{ width:230px; color:var(--sub); }}
footer {{ color:var(--sub); font-size:12px; padding:42px 0 0; }}
@media print {{ header button, header nav {{ display:none; }} section {{ break-inside:avoid; }}
 body {{ background:#fff; }} .page {{ max-width:none; }} }}
</style></head><body data-theme="light"><header><div class="top"><span class="brand">CMC · DWI QA</span>
<nav>{nav}</nav><button id="theme" type="button" aria-label="Toggle colour mode">Dark mode</button></div></header>
<main class="page"><div class="hero"><span class="eyebrow">PROCESSING QUALITY ASSURANCE</span>
<h1>{esc(self.root.parent.name)}</h1><p class="subtitle">A visual and quantitative audit of denoising,
susceptibility correction, EDDY, tensor fitting, registration and tractography. Missing evidence is
shown explicitly; a completed processing step is not a substitute for visual review.</p></div>
<div class="stats">{stat_cards}</div>
<section id="issues"><div class="section-head"><span class="eyebrow">CHECK FIRST</span>
<h2>Action needed and caveats</h2><p>Configuration, phase pairing and missing output checks are listed before the image review.</p>
</div><div class="alerts">{alert_cards}</div></section>
<section id="processing"><div class="section-head"><span class="eyebrow">PIPELINE</span><h2>Processing overview</h2>
<p>Status reflects files present when the report was generated. It does not imply scientific validity.</p></div>
<div class="overview"><div class="steps">{step_cards}</div></div></section>
{''.join(section_html)}
<section id="configuration"><div class="section-head"><span class="eyebrow">PROVENANCE</span>
<h2>Configuration and sources</h2></div><div class="config"><p><strong>Processing root:</strong> {esc(self.root)}</p>
<p><strong>Saved configuration:</strong> {esc(self.sources.get('config', 'unavailable'))}</p>
<p>File modification times for key outputs (provenance estimates, not execution durations):</p>
<table>{date_rows}</table><h3>Selected parameters</h3><table>{config_rows}</table>
<p>Full source paths, figure inventory and automated issues are in
<a href="QA_manifest.json">QA_manifest.json</a>.</p></div></section>
<footer>CMC in-vivo dMRI QA · generated {esc(self.timestamp)} · inspect all figures before accepting a dataset.</footer>
</main><script>
const body=document.body, button=document.getElementById('theme');
let saved=null; try {{ saved=localStorage.getItem('cmc-qa-theme'); }} catch (_) {{}}
if(saved==='dark' || (!saved && matchMedia('(prefers-color-scheme: dark)').matches)) body.dataset.theme='dark';
function label(){{button.textContent=body.dataset.theme==='dark'?'Light mode':'Dark mode';}}
button.onclick=()=>{{body.dataset.theme=body.dataset.theme==='dark'?'light':'dark';
 try {{ localStorage.setItem('cmc-qa-theme',body.dataset.theme); }} catch (_) {{}} label();}};label();
</script></body></html>'''

    def _fsleyes_launch_command(self, sources):
        paths = []
        for source in sources:
            value = str(source)
            if (value.endswith((".nii", ".nii.gz")) and Path(value).is_file()
                    and value not in paths):
                paths.append(value)
        if not paths:
            return ""
        command = ["fsleyes"]
        tract_options = {}
        for view in ("sagittal", "coronal", "axial"):
            for tract, cmap, low, high in _Composer._xtract_view_spec(view):
                tract_options.setdefault(tract, (cmap, low, high))
        for path in paths:
            name = Path(path).name.lower()
            tract = Path(path).parent.name
            command += [path, "--name", Path(path).name.removesuffix(".nii.gz")]
            if "smooth_signal_field" in name or "observed_expected_ratio" in name:
                command += ["--cmap", "coolwarm", "--displayRange", "0.5", "1.5"]
            elif "low_signal_score" in name:
                command += ["--cmap", "hot", "--displayRange", "0", "1"]
            elif "cnr" in name:
                command += ["--cmap", "viridis", "--displayRange", "0", "99%"]
            elif "mask" in name:
                command += ["--overlayType", "mask", "--maskColour", "1", "0", "0",
                            "--outline", "--outlineWidth", "2"]
            elif "densitynorm" in name:
                cmap, low, high = tract_options.get(tract, ("red-yellow", .001, .1))
                command += ["--overlayType", "mip", "--cmap", cmap,
                            "--displayRange", str(low), str(high)]
            elif "_v1" in name:
                command += ["--overlayType", "rgbvector"]
            elif "_md" in name:
                command += ["--cmap", "greyscale", "--displayRange", "0", "0.003"]
            elif "nmt_fa" in name or "hcp1065_fa" in name:
                threshold = ".1" if self.mode == "hum" else ".2"
                command += ["--overlayType", "mask", "--maskColour", "1", "0", "0",
                            "--maskInvert", "--threshold", "0", threshold,
                            "--outline", "--outlineWidth", "3"]
            elif "_fa" in name or "fa_" in name:
                command += ["--cmap", "greyscale", "--displayRange", "0", "1"]
            else:
                command += ["--cmap", "greyscale", "--displayRange", "0", "99%"]
        return shlex.join(command) + " &"

    def _pdf(self, path, theme, metadata):
        t = THEMES[theme]
        with PdfPages(path) as pdf, plt.rc_context(_figure_rc(theme)):
            fig = plt.figure(figsize=(13.33, 7.5), facecolor=t["bg"])
            fig.text(0.055, 0.88, "CMC  /  IN-VIVO DWI", fontsize=13, weight="bold",
                     color=t["blue"])
            fig.text(0.055, 0.76, "Quality assurance", fontsize=32, weight="bold")
            fig.text(0.055, 0.68, self.root.parent.name, fontsize=21)
            details = [f"Generated: {self.timestamp}", f"Mode: {self.mode} / {self.run_mode}",
                       f"Processing root: {self.root}"]
            fig.text(0.055, 0.57, "\n".join(details), fontsize=10, linespacing=1.6)
            alerts = sorted(self.issues, key=lambda i: {"error": 0, "warning": 1}.get(
                i["severity"], 2))
            fig.text(0.055, 0.43, "CHECK FIRST", fontsize=12, weight="bold", color=t["red"])
            if alerts:
                text = "\n".join(f"{item['severity'].upper()}: {item['title']}" for item in alerts[:7])
                if len(alerts) > 7:
                    text += f"\n... and {len(alerts) - 7} more (see HTML/manifest)"
            else:
                text = "No automated issues detected; visual inspection remains necessary."
            fig.text(0.055, 0.38, text, fontsize=10, linespacing=1.55, va="top")
            fig.text(0.95, 0.06, "CMC dMRI  ·  QA overview", ha="right", fontsize=9,
                     color=t["fg"])
            pdf.savefig(fig, facecolor=t["bg"])
            plt.close(fig)
            if len(alerts) > 7:
                for start in range(0, len(alerts), 12):
                    fig = plt.figure(figsize=(13.33, 7.5), facecolor=t["bg"])
                    fig.text(0.05, 0.91, "Action needed and caveats", fontsize=22,
                             weight="semibold")
                    y = 0.82
                    for item in alerts[start:start + 12]:
                        color = t["red"] if item["severity"] == "error" else t["amber"]
                        fig.text(0.055, y, item["severity"].upper() + "  " + item["title"],
                                 fontsize=11, weight="bold", color=color)
                        detail = textwrap.fill(item["detail"], width=135)
                        fig.text(0.055, y - 0.032, detail, fontsize=8.5, color=t["fg"])
                        y -= 0.075 + 0.022 * detail.count("\n")
                    pdf.savefig(fig, facecolor=t["bg"])
                    plt.close(fig)
            for section in self.sections:
                for item in section["figures"]:
                    image_path = self.figures / item[theme]
                    if not image_path.is_file():
                        continue
                    fig = plt.figure(figsize=(13.33, 7.5), facecolor=t["bg"])
                    fig.text(0.055, 0.92, section["title"].upper(), fontsize=11,
                             weight="bold", color=t["blue"])
                    fig.text(0.055, 0.865, item["title"], fontsize=19, weight="semibold")
                    axis = fig.add_axes((0.055, 0.17, 0.89, 0.65))
                    axis.imshow(plt.imread(image_path))
                    axis.axis("off")
                    caption = textwrap.fill(item["caption"], width=155)
                    fig.text(0.055, 0.12, caption, fontsize=9, va="top", color=t["fg"])
                    fig.text(0.95, 0.035, f"{self.root.parent.name} · {theme} QA",
                             ha="right", fontsize=8, color=t["fg"])
                    pdf.savefig(fig, facecolor=t["bg"])
                    plt.close(fig)
