from __future__ import annotations

import yaml
from pathlib import Path

_BASE = Path(__file__).parent
_PROJECT_ROOT = _BASE.parent.parent.parent.parent.parent

with open(_BASE / "config.yaml") as f:
    _raw_config: dict = yaml.safe_load(f)


def _resolve_paths(cfg: dict) -> dict:
    resolved = cfg.copy()
    if "data" in resolved:
        resolved["data"] = {
            k: str(_PROJECT_ROOT / v) if isinstance(v, str) else v for k, v in cfg["data"].items()
        }
    return resolved


config: dict = _resolve_paths(_raw_config)
