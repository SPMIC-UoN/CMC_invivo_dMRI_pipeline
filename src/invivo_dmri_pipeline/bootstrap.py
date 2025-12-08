import os, sys, subprocess
from pathlib import Path
from importlib import resources as ir

ENV_ROOT = Path.home() / ".cmc_invivo"
ENV_DIR  = ENV_ROOT / f"py{sys.version_info.major}.{sys.version_info.minor}"
PY_BIN   = ENV_DIR / "bin" / "python"
PIP_BIN  = ENV_DIR / "bin" / "pip"
STAMP    = ENV_DIR / ".ready"  # ← NEW

def ensure_env():
    ENV_ROOT.mkdir(parents=True, exist_ok=True)

    # Already provisioned? bail early
    if PY_BIN.exists() and STAMP.exists():
        return str(PY_BIN)

    # Fresh venv
    if not PY_BIN.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(ENV_DIR)])

    # Optional: allow skipping noisy upgrades
    if os.environ.get("CMC_SKIP_PIP_UPGRADE", "1") not in ("1", "true", "yes"):
        subprocess.check_call([str(PIP_BIN), "install", "--upgrade", "pip", "wheel", "setuptools"])

    # Install deps matched to interpreter
    with ir.as_file(ir.files("invivo_dmri_pipeline") / "files" / "requirements") as req_dir:
        req_dir = Path(req_dir)
        req_file = req_dir / ("py311.txt" if sys.version_info >= (3,11) else "py310.txt")
        wheelhouse = req_dir / "wheelhouse"
        cmd = [str(PIP_BIN), "install"]
        if wheelhouse.exists():
            cmd += ["--no-index", "--find-links", str(wheelhouse)]
        cmd += ["-r", str(req_file)]
        subprocess.check_call(cmd)

    # Install this repo into the env
    repo_root = Path(__file__).resolve().parents[2]
    subprocess.check_call([str(PIP_BIN), "install", "-e", str(repo_root)])

    STAMP.write_text("ok\n")   # ← mark as provisioned
    return str(PY_BIN)

def entrypoint():
    py = ensure_env()
    os.execv(py, [py, "-m", "invivo_dmri_pipeline.cli", *sys.argv[1:]])
