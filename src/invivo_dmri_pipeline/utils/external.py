# src/dmri_pipeline/utils/external.py
from dataclasses import dataclass, field
import os

def env_or(name: str, default: str) -> str:
    # no validation, no which() — just return an override if user set it
    return os.environ.get(name, default)

@dataclass(frozen=True)
class FSL:
    fsl: str    = field(default_factory=lambda: env_or("FSLDIR", "fsldir"))
