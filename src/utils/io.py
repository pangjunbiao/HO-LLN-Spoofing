"""
Input/output utilities for the AV-GPS causal spoofing detection project.

This file is responsible for:
- creating directories,
- saving/loading JSON,
- saving/loading text,
- saving/loading pickle,
- optional CSV helpers,
- safe path handling,
- simple project file checks,
- Step-2 raw dataset file checks and inspection-output support.

All functions are intentionally general because many later steps will reuse them.
"""

from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union


PathLike = Union[str, Path]


def to_path(path: PathLike) -> Path:
    """Convert string/path input to resolved Path."""
    return Path(path).expanduser().resolve()


def ensure_dir(path: PathLike) -> Path:
    """
    Create a directory if it does not exist.

    Returns the resolved Path.
    """
    path = to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Create parent directory of a file path if needed.

    Returns the resolved file Path.
    """
    path = to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirs(paths: Iterable[PathLike]) -> List[Path]:
    """Create multiple directories."""
    return [ensure_dir(path) for path in paths]


def file_exists(path: PathLike) -> bool:
    """Check whether a file exists."""
    return to_path(path).is_file()


def dir_exists(path: PathLike) -> bool:
    """Check whether a directory exists."""
    return to_path(path).is_dir()


def timestamp(fmt: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """Return current timestamp string."""
    return datetime.now().strftime(fmt)


def make_run_id(prefix: str = "run") -> str:
    """Create a timestamped run ID."""
    return f"{prefix}_{timestamp()}"


def make_json_safe(obj: Any) -> Any:
    """
    Convert common non-JSON-safe objects into JSON-safe values.

    This is useful for saving inspection reports that may contain:
    - pathlib.Path
    - NumPy integer/float/bool values
    - pandas/NumPy missing values
    - tuples/sets
    """
    if obj is None:
        return None

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Mapping):
        return {str(key): make_json_safe(value) for key, value in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, tuple):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, set):
        return sorted([make_json_safe(item) for item in obj])

    # Handle NumPy scalar values without requiring NumPy import globally.
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    # Handle pandas/NumPy NaN-like values.
    try:
        if obj != obj:
            return None
    except Exception:
        pass

    return str(obj)


def save_json(data: Any, path: PathLike, indent: int = 2) -> Path:
    """Save data as JSON."""
    path = ensure_parent_dir(path)

    safe_data = make_json_safe(data)

    with path.open("w", encoding="utf-8") as file:
        json.dump(safe_data, file, indent=indent, ensure_ascii=False)

    return path


def load_json(path: PathLike) -> Any:
    """Load JSON file."""
    path = to_path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_text(text: str, path: PathLike) -> Path:
    """Save text to file, overwriting existing content."""
    path = ensure_parent_dir(path)

    with path.open("w", encoding="utf-8") as file:
        file.write(text)

    return path


def append_text(text: str, path: PathLike) -> Path:
    """Append text to file."""
    path = ensure_parent_dir(path)

    with path.open("a", encoding="utf-8") as file:
        file.write(text)

    return path


def load_text(path: PathLike) -> str:
    """Load text file."""
    path = to_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return file.read()


def save_pickle(obj: Any, path: PathLike) -> Path:
    """Save Python object using pickle."""
    path = ensure_parent_dir(path)

    with path.open("wb") as file:
        pickle.dump(obj, file)

    return path


def load_pickle(path: PathLike) -> Any:
    """Load Python object using pickle."""
    path = to_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with path.open("rb") as file:
        return pickle.load(file)


def save_csv(data: Any, path: PathLike, index: bool = False) -> Path:
    """
    Save CSV.

    Expects a pandas DataFrame-like object with .to_csv().
    """
    path = ensure_parent_dir(path)

    if not hasattr(data, "to_csv"):
        raise TypeError("save_csv expects a pandas DataFrame-like object with .to_csv().")

    data.to_csv(path, index=index)
    return path


def load_csv(path: PathLike, **kwargs: Any) -> Any:
    """
    Load CSV using pandas.

    Pandas import is kept inside the function so early utilities remain lightweight.
    """
    path = to_path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to load CSV files.") from exc

    return pd.read_csv(path, **kwargs)


def copy_file(source: PathLike, destination: PathLike, overwrite: bool = True) -> Path:
    """Copy a file from source to destination."""
    source = to_path(source)
    destination = ensure_parent_dir(destination)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    shutil.copy2(source, destination)
    return destination


def list_files(
    directory: PathLike,
    pattern: str = "*",
    recursive: bool = False,
) -> List[Path]:
    """List files in a directory."""
    directory = to_path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if recursive:
        return sorted([path for path in directory.rglob(pattern) if path.is_file()])

    return sorted([path for path in directory.glob(pattern) if path.is_file()])


def create_empty_file(path: PathLike, overwrite: bool = False) -> Path:
    """
    Create an empty file.

    Useful for project skeleton files and placeholder outputs.
    """
    path = ensure_parent_dir(path)

    if path.exists() and not overwrite:
        return path

    with path.open("w", encoding="utf-8"):
        pass

    return path


def check_required_files(paths: Iterable[PathLike]) -> Dict[str, bool]:
    """
    Check whether required files exist.

    Returns:
        {
            "/path/to/file.csv": True,
            "/path/to/missing.csv": False
        }
    """
    result: Dict[str, bool] = {}

    for path in paths:
        resolved = to_path(path)
        result[str(resolved)] = resolved.is_file()

    return result


def print_file_check_table(file_status: Dict[str, bool]) -> None:
    """Print a simple file-existence table."""
    print("=" * 100)
    print("FILE CHECK")
    print("=" * 100)

    for file_path, exists in file_status.items():
        status = "FOUND" if exists else "MISSING"
        print(f"{status:8s} | {file_path}")

    print("=" * 100)


def check_required_files_by_key(paths: Mapping[str, PathLike]) -> Dict[str, Dict[str, Any]]:
    """
    Check required files with dataset/file keys.

    Returns:
        {
            "dataset1": {"path": "...", "exists": True},
            ...
        }
    """
    result: Dict[str, Dict[str, Any]] = {}

    for key, path in paths.items():
        resolved = to_path(path)
        result[str(key)] = {
            "path": str(resolved),
            "exists": resolved.is_file(),
        }

    return result


def print_keyed_file_check_table(file_status: Mapping[str, Mapping[str, Any]]) -> None:
    """
    Print file check table with keys.

    Useful for AV-GPS raw files.
    """
    print("=" * 100)
    print("KEYED FILE CHECK")
    print("=" * 100)
    print(f"{'Key':22s} | {'Status':8s} | Path")
    print("-" * 100)

    for key, item in file_status.items():
        exists = bool(item.get("exists", False))
        path = str(item.get("path", ""))
        status = "FOUND" if exists else "MISSING"
        print(f"{key:22s} | {status:8s} | {path}")

    print("=" * 100)


def ensure_standard_project_dirs(project_root: PathLike) -> None:
    """
    Create standard project directories.

    This is useful in Step 1 and Step 2 to make sure logs/results/data folders exist.
    """
    project_root = to_path(project_root)

    standard_dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "interim",
        project_root / "data" / "processed",
        project_root / "data" / "splits",
        project_root / "results" / "tables",
        project_root / "results" / "figures",
        project_root / "results" / "figures" / "pr_curves",
        project_root / "results" / "figures" / "roc_curves",
        project_root / "results" / "figures" / "ablation_plots",
        project_root / "results" / "figures" / "dataset3_case_study",
        project_root / "results" / "figures" / "sensitivity_plots",
        project_root / "results" / "figures" / "module_usage",
        project_root / "results" / "models",
        project_root / "results" / "models" / "baselines",
        project_root / "results" / "models" / "ablations",
        project_root / "logs",
    ]

    ensure_dirs(standard_dirs)


def print_directory_summary(project_root: PathLike) -> None:
    """
    Print whether standard project directories exist.

    Useful for console inspection.
    """
    project_root = to_path(project_root)

    dirs = [
        project_root / "data" / "raw",
        project_root / "data" / "interim",
        project_root / "data" / "processed",
        project_root / "data" / "splits",
        project_root / "results" / "tables",
        project_root / "results" / "figures",
        project_root / "results" / "models",
        project_root / "logs",
    ]

    print("=" * 100)
    print("PROJECT DIRECTORY CHECK")
    print("=" * 100)

    for directory in dirs:
        status = "FOUND" if directory.is_dir() else "MISSING"
        print(f"{status:8s} | {directory}")

    print("=" * 100)

def save_npz(path, **arrays) -> None:
    """
    Save numpy arrays to compressed NPZ.

    Used in Step 9 for flattened XGBoost/MLP baseline arrays.
    """
    from pathlib import Path
    import numpy as np

    output_path = Path(path)
    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **arrays)


def load_npz(path):
    """
    Load numpy NPZ file.

    Returns numpy.lib.npyio.NpzFile. Caller can access arrays by key.
    """
    from pathlib import Path
    import numpy as np

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {input_path}")

    return np.load(input_path, allow_pickle=True)