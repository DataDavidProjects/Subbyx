from __future__ import annotations

import yaml
from pathlib import Path

# Base directory for the routes
_BASE_DIR = Path(__file__).parent.parent

# Load shared configuration
_config_path = _BASE_DIR / "config" / "shared.yaml"
with open(_config_path) as f:
    shared_config: dict = yaml.safe_load(f)
