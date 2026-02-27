from __future__ import annotations

import yaml
from pathlib import Path

_BASE = Path(__file__).parent
with open(_BASE / "config.yaml") as f:
    config: dict = yaml.safe_load(f)
