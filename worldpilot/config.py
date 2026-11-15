"""Configuration loading and typed accessors."""
from pathlib import Path
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "drone_config.yaml"


def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
