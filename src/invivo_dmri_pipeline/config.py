import os
import yaml
from typing import Any, Dict, List, Optional, Union


def _abspath(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    return os.path.abspath(os.path.expanduser(p))


def _abspath_list(xs: Optional[List[str]]) -> Optional[List[str]]:
    if not xs:
        return xs
    return [os.path.abspath(os.path.expanduser(x)) for x in xs]


def _lower_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of dict `d` with all keys lowercased."""
    return {k.lower(): v for k, v in d.items()}


class Config:
    """
    Parse and hold pipeline configuration loaded from YAML.
    All keys are case-insensitive (normalised to lowercase).
    """
    def __init__(self, y: Dict[str, Any]) -> None:
        # Normalise key case
        y = _lower_keys(y)

        # ---- required numeric ----
        try:
            self.ECHO_MS  = float(y["echo_ms"])
            self.PIFACTOR = int(y["pifactor"])
            self.LOWER_B  = int(y["lower_b"])
        except KeyError as e:
            raise RuntimeError(f"Missing required config key: {e.args[0]}")

        # ---- mandatory mode ----
        mode_raw = str(y.get("mode", "")).strip().lower()
        if mode_raw not in ("nhp", "hum"):
            raise RuntimeError("Missing or invalid MODE in config: must be 'nhp' or 'hum'.")
        self.MODE = mode_raw

        # ---- optional numeric/flags ----
        self.B0RANGE              = float(y.get("b0range", 60))
        self.COMBINE_MATCHED_FLAG = int(y.get("combine_matched_flag", 2))
        self.PTX_STEPLENGTH       = float(y.get("ptx_steplength", 0.15))
        self.NO_GPU               = bool(y.get("no_gpu", False))
        self.DENOISE            = bool(y.get("denoise", True))

        # ---- EDDY extra args ----
        eea = y.get("eddy_extra_args", [])
        if isinstance(eea, str):
            self.eddy_extra_args = eea.strip()
        elif isinstance(eea, list):
            self.eddy_extra_args = [str(a) for a in eea]
        else:
            self.eddy_extra_args = []

        # ---- post-combine options ----
        self.run_gibbs   = bool(y.get("run_gibbs", True))
        self.run_n4      = bool(y.get("run_n4", True))
        self.use_docker  = bool(y.get("use_docker", True))
        self.container_runtime = str(y.get("container_runtime", "docker"))

        # Docker images (defaults pinned to MRtrix 3.0.7)
        # self.gibbs_image      = str(y.get("gibbs_image", "docker.io/mrtrix3/mrtrix3:3.0.7"))
        # self.n4_image         = str(y.get("n4_image", "docker.io/mrtrix3/mrtrix3:3.0.7"))
        self.docker_image      = str(y.get("docker_image", "docker.io/mrtrix3/mrtrix3:3.0.7"))
        self.interactive_tty  = bool(y.get("interactive_tty", True))

        # Optional masks / inputs
        self.n4_mask    = _abspath(y.get("n4_mask"))
        self.t1         = _abspath(y.get("t1"))
        self.brain_mask = _abspath(y.get("brain_mask"))

        # ---- workspace roots ----
        self.dmri_root     = y.get("dmri_root", "./dmri_proc")
        self.abs_dmri_root = _abspath(self.dmri_root)
        self.proc_root     = self.abs_dmri_root

        # optional raw input root (pipeline can infer from mag_files)
        self.input_root = y.get("input_root")
        if self.input_root is not None:
            self.input_root = _abspath(self.input_root)

        # ---- file groups ----
        self.mag_files   = _abspath_list(y.get("mag_files", [])) or []
        self.phase_files = _abspath_list(y.get("phase_files", [])) or []

        # ---- external resources ----
        self.xtract_profiles_dir     = y.get("xtract_profiles_dir")
        self.abs_xtract_profiles_dir = _abspath(self.xtract_profiles_dir) if self.xtract_profiles_dir else None

        self.denoise_sh     = y.get("denoise_sh")
        self.abs_denoise_sh = _abspath(self.denoise_sh) if self.denoise_sh else None

        # ---- FSLDIR / standard references ----
        self.FSLDIR = os.environ.get("FSLDIR", "")
        if not self.FSLDIR or not os.path.isdir(self.FSLDIR):
            raise RuntimeError("FSLDIR environment variable not set or invalid. Please source FSL before running.")

        self.fsl_std_dir         = os.path.join(self.FSLDIR, "data", "standard")
        self.fsl_mni_t1_1mm      = os.path.join(self.fsl_std_dir, "MNI152_T1_1mm.nii.gz")
        self.fsl_hcp1065_fa_1mm  = os.path.join(self.fsl_std_dir, "FSL_HCP1065_FA_1mm.nii.gz")
        self.fsl_hcp1065_ten_1mm = os.path.join(self.fsl_std_dir, "FSL_HCP1065_tensor_1mm.nii.gz")

        # ---- Mode-aware derived fields ----
        self.bet4animal_z = 2 if self.MODE == "nhp" else 0

        if self.MODE == "nhp":
            self.xtract_species_default = "MACAQUE"
            self.xtract_mac_ext_default = "_NMT"
            self.xtract_stdref_hum = None
        else:
            self.xtract_species_default = "HUMAN"
            self.xtract_mac_ext_default = ""
            self.xtract_stdref_hum = self.fsl_mni_t1_1mm

        if self.MODE == "hum":
            self.reg_ref_fa     = self.fsl_hcp1065_fa_1mm
            self.reg_ref_tensor = self.fsl_hcp1065_ten_1mm
        else:
            self.reg_ref_fa     = None
            self.reg_ref_tensor = None

        self.qa_std_fa = self.fsl_hcp1065_fa_1mm

        self.stdreg_method = str(y.get("stdreg_method", "")).strip().lower() or None

        # ---- Streamlines / viewer options ----
        self.STREAMLINES_DO                 = bool(y.get("streamlines_do", True))
        self.STREAMLINES_DENSITY_THRESHOLD  = float(y.get("streamlines_density_threshold", 1e-3))
        self.STREAMLINES_FORMAT             = str(y.get("streamlines_format", "trk"))
        self.STREAMLINES_PTX2_PREFIX        = str(y.get("streamlines_ptx2_prefix", "densityNorm"))
        self.STREAMLINES_NUM_JOBS           = int(y.get("streamlines_num_jobs", 1))
        self.DO_VIEWER                      = bool(y.get("do_viewer", True))

        # path to YAML
        self.abs_yaml_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, os.PathLike]) -> "Config":
        path = os.fspath(path)
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls(data)
        cfg.abs_yaml_path = os.path.abspath(path)
        return cfg

    def __repr__(self) -> str:
        fields = {
            "MODE": self.MODE,
            "ECHO_MS": self.ECHO_MS,
            "PIFACTOR": self.PIFACTOR,
            "LOWER_B": self.LOWER_B,
            "B0RANGE": self.B0RANGE,
            "DENOISE": self.DENOISE,
            "RUN_GIBBS": self.run_gibbs,
            "RUN_N4": self.run_n4,
            "USE_DOCKER": self.use_docker,
            "dmri_root": self.abs_dmri_root,
            "eddy_extra_args": self.eddy_extra_args,
            "bet4animal_z": self.bet4animal_z,
            "stdreg_method": self.stdreg_method,
        }
        if self.MODE == "hum":
            fields.update({
                "reg_ref_fa": self.reg_ref_fa,
                "reg_ref_tensor": self.reg_ref_tensor,
                "xtract_stdref_hum": self.xtract_stdref_hum,
            })
        kv = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"Config({kv})"
