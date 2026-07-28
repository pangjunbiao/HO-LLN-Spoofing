"""
Raw dataset loading utilities for the AV-GPS causal spoofing detection project.

Step 2 purpose:
- check expected AV-GPS raw CSV files exist,
- load all raw CSV files safely,
- attach source-file metadata when requested,
- provide clean summaries for logging/console inspection.

Step 3 support:
- preserve raw row order,
- provide stable source metadata used by trajectory segmentation,
- expose helper functions for raw bundle loading in later preprocessing steps.

This file does not segment, split, train, or build xi_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from src.utils.config import get_by_path, resolve_project_path
from src.utils.io import (
    check_required_files,
    check_required_files_by_key,
    print_file_check_table,
    print_keyed_file_check_table,
)


@dataclass(frozen=True)
class RawDatasetSpec:
    """
    Specification for one raw AV-GPS dataset file.
    """

    key: str
    file_name: str
    role: str
    expected_has_label: bool = True
    expected_has_ekf_detector: bool = False


@dataclass
class RawDatasetBundle:
    """
    Container for loaded raw datasets.
    """

    dataframes: Dict[str, pd.DataFrame]
    specs: Dict[str, RawDatasetSpec]
    paths: Dict[str, Path]

    def keys(self) -> List[str]:
        """Return dataset keys in loaded order."""
        return list(self.dataframes.keys())

    def get(self, key: str) -> pd.DataFrame:
        """Return DataFrame by dataset key."""
        if key not in self.dataframes:
            raise KeyError(f"Dataset key not found: {key}")
        return self.dataframes[key]

    def spec(self, key: str) -> RawDatasetSpec:
        """Return dataset spec by key."""
        if key not in self.specs:
            raise KeyError(f"Dataset spec not found: {key}")
        return self.specs[key]

    def path(self, key: str) -> Path:
        """Return dataset raw path by key."""
        if key not in self.paths:
            raise KeyError(f"Dataset path not found: {key}")
        return self.paths[key]

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Return compact bundle summary."""
        return {
            key: {
                "path": str(self.paths[key]),
                "role": self.specs[key].role,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
            }
            for key, df in self.dataframes.items()
        }


def default_raw_dataset_specs() -> Dict[str, RawDatasetSpec]:
    """
    Default AV-GPS raw file specifications.

    These defaults match the dataset package:
    - Dataset-1: main mixed file
    - Dataset-1-Normal: normal-only reference file
    - Dataset-2: external/source-shift file
    - Dataset-3: online case-study file with EKF Detector
    """
    return {
        "dataset1": RawDatasetSpec(
            key="dataset1",
            file_name="AV-GPS-Dataset-1.csv",
            role="main_development_mixed",
            expected_has_label=True,
            expected_has_ekf_detector=False,
        ),
        "dataset1_normal": RawDatasetSpec(
            key="dataset1_normal",
            file_name="AV-GPS-Dataset-1-Normal-Data.csv",
            role="normal_reference_not_independent",
            expected_has_label=True,
            expected_has_ekf_detector=False,
        ),
        "dataset2": RawDatasetSpec(
            key="dataset2",
            file_name="AV-GPS-Dataset-2.csv",
            role="external_source_shift_test",
            expected_has_label=True,
            expected_has_ekf_detector=False,
        ),
        "dataset3": RawDatasetSpec(
            key="dataset3",
            file_name="AV-GPS-Dataset-3.csv",
            role="online_case_study_with_ekf",
            expected_has_label=True,
            expected_has_ekf_detector=True,
        ),
    }


def raw_dataset_specs_from_config(config: Mapping[str, Any]) -> Dict[str, RawDatasetSpec]:
    """
    Build raw dataset specs from config.

    If configs/dataset.yaml does not define raw files, defaults are used.

    Expected YAML format:

        dataset:
          raw_files:
            dataset1:
              file_name: "AV-GPS-Dataset-1.csv"
              role: "main_development_mixed"
              expected_has_label: true
              expected_has_ekf_detector: false
    """
    configured = get_by_path(config, "dataset.raw_files", default=None)

    if configured is None:
        return default_raw_dataset_specs()

    if not isinstance(configured, Mapping):
        raise TypeError("dataset.raw_files must be a mapping/dictionary.")

    specs: Dict[str, RawDatasetSpec] = {}

    for key, item in configured.items():
        if not isinstance(item, Mapping):
            raise TypeError(f"dataset.raw_files.{key} must be a mapping/dictionary.")

        file_name = item.get("file_name")
        if not file_name:
            raise ValueError(f"dataset.raw_files.{key}.file_name is required.")

        specs[str(key)] = RawDatasetSpec(
            key=str(key),
            file_name=str(file_name),
            role=str(item.get("role", "unspecified")),
            expected_has_label=bool(item.get("expected_has_label", True)),
            expected_has_ekf_detector=bool(item.get("expected_has_ekf_detector", False)),
        )

    return specs


def get_raw_data_dir(config: Mapping[str, Any]) -> Path:
    """
    Resolve raw data directory from config.

    Uses paths.raw_data_dir if available; otherwise defaults to data/raw.
    """
    raw_dir_value = get_by_path(config, "paths.raw_data_dir", default="data/raw")
    return resolve_project_path(config, raw_dir_value)


def resolve_raw_dataset_paths(
    config: Mapping[str, Any],
    specs: Optional[Dict[str, RawDatasetSpec]] = None,
) -> Dict[str, Path]:
    """
    Resolve full paths for all raw dataset files.
    """
    specs = specs or raw_dataset_specs_from_config(config)
    raw_dir = get_raw_data_dir(config)

    return {
        key: (raw_dir / spec.file_name).resolve()
        for key, spec in specs.items()
    }


def check_raw_dataset_files(
    config: Mapping[str, Any],
    raise_if_missing: bool = True,
    print_table: bool = True,
    keyed_table: bool = True,
) -> Dict[str, bool]:
    """
    Check that all expected raw CSV files exist.

    Returns:
        {
            "dataset1": True,
            "dataset2": False,
            ...
        }
    """
    specs = raw_dataset_specs_from_config(config)
    paths = resolve_raw_dataset_paths(config, specs)

    if print_table and keyed_table:
        keyed_status = check_required_files_by_key(paths)
        print_keyed_file_check_table(keyed_status)

    elif print_table:
        file_status_by_path = check_required_files(paths.values())
        print_file_check_table(file_status_by_path)

    status_by_key = {
        key: paths[key].is_file()
        for key in specs.keys()
    }

    missing = [key for key, exists in status_by_key.items() if not exists]

    if raise_if_missing and missing:
        missing_lines = "\n".join(
            f"  - {key}: {paths[key]}"
            for key in missing
        )
        raise FileNotFoundError(
            "Missing required raw AV-GPS dataset files:\n"
            f"{missing_lines}\n\n"
            "Please place the four CSV files inside data/raw/."
        )

    return status_by_key


def load_single_raw_dataset(
    path: Path,
    key: str,
    add_source_column: bool = True,
    add_raw_order_column: bool = True,
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load a single raw CSV file.

    Args:
        path:
            CSV path.
        key:
            Dataset key, e.g. dataset1.
        add_source_column:
            If True, adds source_key and source_file columns.
        add_raw_order_column:
            If True, adds raw_row_index before any processing.
        low_memory:
            Passed to pandas.read_csv.

    Returns:
        Loaded DataFrame.
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {path}")

    try:
        df = pd.read_csv(path, low_memory=low_memory)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CSV file: {path}") from exc

    if add_raw_order_column and "raw_row_index" not in df.columns:
        df.insert(0, "raw_row_index", range(len(df)))

    if add_source_column:
        if "source_key" not in df.columns:
            insert_at = 1 if "raw_row_index" in df.columns else 0
            df.insert(insert_at, "source_key", key)

        if "source_file" not in df.columns:
            insert_at = 2 if "raw_row_index" in df.columns else 1
            df.insert(insert_at, "source_file", path.name)

    return df


def load_all_raw_datasets(
    config: Mapping[str, Any],
    add_source_column: bool = True,
    add_raw_order_column: bool = True,
    check_files_first: bool = True,
    low_memory: bool = False,
) -> RawDatasetBundle:
    """
    Load all raw AV-GPS datasets.

    This is the main loader used in Step 2 and Step 3.
    """
    specs = raw_dataset_specs_from_config(config)
    paths = resolve_raw_dataset_paths(config, specs)

    if check_files_first:
        check_raw_dataset_files(
            config=config,
            raise_if_missing=True,
            print_table=True,
            keyed_table=True,
        )

    dataframes: Dict[str, pd.DataFrame] = {}

    for key, path in paths.items():
        df = load_single_raw_dataset(
            path=path,
            key=key,
            add_source_column=add_source_column,
            add_raw_order_column=add_raw_order_column,
            low_memory=low_memory,
        )
        dataframes[key] = df

    return RawDatasetBundle(
        dataframes=dataframes,
        specs=specs,
        paths=paths,
    )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_key: str,
) -> None:
    """
    Validate required columns exist in one DataFrame.
    """
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        missing_str = ", ".join(missing)
        raise KeyError(
            f"Dataset '{dataset_key}' is missing required columns: {missing_str}"
        )


def validate_raw_bundle_basic_schema(
    bundle: RawDatasetBundle,
    label_column: str = "Data Type",
    ekf_column: str = "EKF Detector",
) -> None:
    """
    Basic raw schema validation.

    Checks:
    - expected label column exists when required,
    - EKF Detector exists only where expected.
    """
    for key, df in bundle.dataframes.items():
        spec = bundle.specs[key]

        if spec.expected_has_label and label_column not in df.columns:
            raise KeyError(
                f"Dataset '{key}' is expected to contain label column "
                f"'{label_column}', but it is missing."
            )

        has_ekf = ekf_column in df.columns

        if spec.expected_has_ekf_detector and not has_ekf:
            raise KeyError(
                f"Dataset '{key}' is expected to contain '{ekf_column}', "
                "but it is missing."
            )

        if not spec.expected_has_ekf_detector and has_ekf:
            raise ValueError(
                f"Dataset '{key}' unexpectedly contains '{ekf_column}'. "
                "Only Dataset-3 should contain EKF Detector."
            )


def print_raw_loading_summary(bundle: RawDatasetBundle) -> None:
    """
    Print raw loading summary to console.
    """
    print("=" * 100)
    print("RAW DATASET LOADING SUMMARY")
    print("=" * 100)
    print(f"{'Key':20s} | {'Rows':>10s} | {'Cols':>6s} | {'Role':35s} | Path")
    print("-" * 100)

    for key, df in bundle.dataframes.items():
        role = bundle.specs[key].role
        path = bundle.paths[key]
        print(
            f"{key:20s} | {df.shape[0]:10d} | {df.shape[1]:6d} | "
            f"{role:35s} | {path}"
        )

    print("=" * 100)


def load_and_validate_raw_datasets(config: Mapping[str, Any]) -> RawDatasetBundle:
    """
    Convenience function for Step 2 and Step 3.

    Loads all raw datasets and performs basic schema validation.
    """
    bundle = load_all_raw_datasets(
        config=config,
        add_source_column=bool(
            get_by_path(config, "dataset.add_source_columns", default=True)
        ),
        add_raw_order_column=bool(
            get_by_path(config, "dataset.add_raw_order_column", default=True)
        ),
        check_files_first=True,
        low_memory=bool(
            get_by_path(config, "dataset.pandas_low_memory", default=False)
        ),
    )

    label_column = str(get_by_path(config, "dataset.label_column", default="Data Type"))
    ekf_column = str(get_by_path(config, "dataset.ekf_column", default="EKF Detector"))

    validate_raw_bundle_basic_schema(
        bundle=bundle,
        label_column=label_column,
        ekf_column=ekf_column,
    )

    print_raw_loading_summary(bundle)

    return bundle


def combine_raw_datasets_for_inspection(
    bundle: RawDatasetBundle,
    keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Combine selected raw datasets for inspection only.

    This must not be used as final training data.
    """
    selected_keys = keys or bundle.keys()

    frames = []
    for key in selected_keys:
        frames.append(bundle.get(key).copy())

    if not frames:
        raise ValueError("No datasets selected for combination.")

    return pd.concat(frames, axis=0, ignore_index=True, sort=False)