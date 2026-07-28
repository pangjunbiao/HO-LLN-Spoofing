"""
Configuration utilities for the AV-GPS causal spoofing detection project.

This file is responsible for:
- loading YAML config files,
- merging multiple config files,
- supporting nested dictionary access,
- supporting future CLI/config overrides,
- keeping config loading safe and reproducible.

Step 1 note:
Only the config system is implemented here. The actual YAML contents will be
filled gradually step by step.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Union

import yaml


PathLike = Union[str, Path]


DEFAULT_CONFIG_ORDER = [
    "paths.yaml",
    "dataset.yaml",
    "preprocessing.yaml",
    "model.yaml",
    "training.yaml",
    "baselines.yaml",
    "experiments.yaml",
    "default.yaml",
]


class Config(dict):
    """
    Dictionary with dot-access support.

    Example:
        cfg = Config({"seed": {"mode": "single"}})
        print(cfg.seed.mode)

    It still behaves like a normal dictionary.
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        data = data or {}
        for key, value in data.items():
            self[key] = self._wrap(value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no attribute '{name}'") from exc

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        if isinstance(value, list):
            return [Config._wrap(item) for item in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config recursively back to a standard Python dict."""
        return _to_plain_dict(self)


def _to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {key: _to_plain_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(item) for item in obj]
    return obj


def deep_merge(
    base: MutableMapping[str, Any],
    update: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Recursively merge update into base.

    If both base[key] and update[key] are dictionaries, merge them.
    Otherwise, update[key] replaces base[key].

    Later config files override earlier config files.
    """
    for key, value in update.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: PathLike, allow_empty: bool = True) -> Dict[str, Any]:
    """
    Load one YAML file.

    Empty YAML files are allowed during early project setup and return {}.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        if allow_empty:
            return {}
        raise ValueError(f"YAML config file is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"YAML config must contain a dictionary at top level: {path}")

    return data


def expand_env_vars_in_config(data: Any) -> Any:
    """
    Recursively expand environment variables inside config strings.

    Example:
        "$HOME/project" -> "/home/user/project"
    """
    if isinstance(data, Mapping):
        return {key: expand_env_vars_in_config(value) for key, value in data.items()}

    if isinstance(data, list):
        return [expand_env_vars_in_config(item) for item in data]

    if isinstance(data, str):
        return os.path.expandvars(os.path.expanduser(data))

    return data


def infer_project_root(config_dir: PathLike) -> Path:
    """
    Infer project root from the configs directory.

    Expected:
        AV_GPS_Spoofing_Project/configs/

    Then project root is:
        AV_GPS_Spoofing_Project/
    """
    config_dir = Path(config_dir).resolve()

    if config_dir.name == "configs":
        return config_dir.parent

    return config_dir.resolve()


def load_project_config(
    config_dir: PathLike = "configs",
    config_files: Optional[Iterable[str]] = None,
    allow_missing_optional: bool = True,
) -> Config:
    """
    Load and merge project config files.

    By default, this loads files in DEFAULT_CONFIG_ORDER.

    Later files override earlier files. This means default.yaml can override
    values from specific YAML files if needed.

    Empty YAML files are allowed so we can build the project step by step.
    """
    config_dir = Path(config_dir).resolve()
    config_files = list(config_files or DEFAULT_CONFIG_ORDER)

    merged: Dict[str, Any] = {}

    for file_name in config_files:
        file_path = config_dir / file_name

        if not file_path.exists():
            if allow_missing_optional:
                continue
            raise FileNotFoundError(f"Missing config file: {file_path}")

        loaded = load_yaml(file_path, allow_empty=True)
        deep_merge(merged, loaded)

    merged = expand_env_vars_in_config(merged)

    project_root = infer_project_root(config_dir)
    merged.setdefault("project", {})
    merged["project"].setdefault("root", str(project_root))

    return Config(merged)


def get_by_path(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot path.

    Example:
        get_by_path(cfg, "seed.mode", default="single")
    """
    current: Any = config

    for key in key_path.split("."):
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default

    return current


def set_by_path(config: MutableMapping[str, Any], key_path: str, value: Any) -> None:
    """
    Set nested config value using dot path.

    Example:
        set_by_path(cfg, "seed.mode", "multi")
    """
    keys = key_path.split(".")
    current: MutableMapping[str, Any] = config

    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value


def apply_overrides(
    config: Union[Config, Dict[str, Any]],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Config:
    """
    Apply dot-path overrides to a config.

    Example:
        overrides = {
            "seed.mode": "multi",
            "run.mode": "preprocess"
        }
    """
    cfg_dict = _to_plain_dict(config)

    if overrides:
        for key_path, value in overrides.items():
            set_by_path(cfg_dict, key_path, value)

    return Config(cfg_dict)


def validate_required_keys(
    config: Mapping[str, Any],
    required_keys: Iterable[str],
) -> None:
    """
    Validate that required nested keys exist.

    This will be useful later when configs become more complete.
    """
    missing = []

    for key_path in required_keys:
        sentinel = object()
        value = get_by_path(config, key_path, default=sentinel)
        if value is sentinel:
            missing.append(key_path)

    if missing:
        missing_str = "\n".join(f"  - {key}" for key in missing)
        raise KeyError(f"Missing required config keys:\n{missing_str}")


def resolve_project_path(config: Mapping[str, Any], path_value: PathLike) -> Path:
    """
    Resolve a path relative to project.root if it is not absolute.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    project_root = Path(get_by_path(config, "project.root", ".")).resolve()
    return (project_root / path).resolve()


def print_config_summary(config: Mapping[str, Any]) -> None:
    """
    Small console summary for Step 1 debugging.
    """
    project_root = get_by_path(config, "project.root", "UNKNOWN")
    run_mode = get_by_path(config, "run.mode", "UNKNOWN")
    seed_mode = get_by_path(config, "seed.mode", "UNKNOWN")
    device_pref = get_by_path(config, "device.preference", "UNKNOWN")

    print("=" * 80)
    print("CONFIG SUMMARY")
    print("=" * 80)
    print(f"Project root      : {project_root}")
    print(f"Run mode          : {run_mode}")
    print(f"Seed mode         : {seed_mode}")
    print(f"Device preference : {device_pref}")
    print("=" * 80)