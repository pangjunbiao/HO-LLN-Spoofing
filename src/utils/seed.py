"""
Seed and reproducibility utilities for the AV-GPS causal spoofing detection project.

Step 12 updates:
- supports reproducible PyTorch training,
- supports DataLoader worker seeding,
- supports torch.Generator creation,
- sets CuBLAS workspace config for deterministic CUDA where possible,
- supports single-seed and multi-seed experiments,
- keeps compatibility with previous Step 1-11 code.

Important:
The CuBLAS workspace environment variable should be set before heavy CUDA
operations. This file sets it before importing torch inside set_global_seed().
If torch was imported earlier, PyTorch may still warn. That warning is not a
training failure, but for final multi-seed experiments we should keep this
configured.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable, List, Mapping, Optional

import numpy as np


@dataclass
class SeedSetupSummary:
    """JSON-safe seed setup summary."""

    seed: int
    deterministic: bool
    benchmark: bool
    deterministic_warn_only: bool
    cublas_workspace_config: Optional[str]
    cuda_tf32: bool
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _get_nested(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Lightweight nested config getter.

    Example:
        _get_nested(cfg, "seed.mode", "single")
    """
    current: Any = config

    for key in key_path.split("."):
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default

    return current


def configure_cublas_workspace(
    deterministic: bool = True,
    workspace_config: Optional[str] = ":4096:8",
) -> Optional[str]:
    """
    Configure CuBLAS deterministic workspace.

    Valid common values:
        ":4096:8"
        ":16:8"

    This helps avoid the CUDA deterministic warning seen in Step 11.
    """
    if not deterministic:
        return None

    if workspace_config is None or str(workspace_config).strip() == "":
        return None

    workspace_config = str(workspace_config).strip()

    if workspace_config not in {":4096:8", ":16:8"}:
        raise ValueError(
            "Invalid CUBLAS_WORKSPACE_CONFIG. Expected ':4096:8' or ':16:8', "
            f"got {workspace_config!r}."
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", workspace_config)
    return os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def set_python_numpy_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    seed = int(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)


def set_torch_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    deterministic_warn_only: bool = True,
    cuda_tf32: bool = False,
) -> bool:
    """
    Set PyTorch seed and deterministic controls.

    Returns:
        True if PyTorch was available, False otherwise.
    """
    try:
        import torch
    except ImportError:
        return False

    seed = int(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = bool(benchmark)

    # Disable TF32 by default for tighter reproducibility.
    # You can enable it later for speed if exact reproducibility is less important.
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(cuda_tf32)
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = bool(cuda_tf32)
    except Exception:
        pass

    if deterministic:
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=bool(deterministic_warn_only),
            )
        except TypeError:
            # Older PyTorch may not support warn_only.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    return True


def set_global_seed(
    seed: int,
    deterministic: bool = True,
    benchmark: bool = False,
    verbose: bool = True,
    deterministic_warn_only: bool = True,
    cublas_workspace_config: Optional[str] = ":4096:8",
    cuda_tf32: bool = False,
) -> int:
    """
    Set global seed for Python, NumPy, and PyTorch.

    Args:
        seed:
            Integer random seed.
        deterministic:
            If True, asks PyTorch to use deterministic algorithms where possible.
        benchmark:
            Controls torch.backends.cudnn.benchmark. Usually False for reproducibility.
        verbose:
            If True, prints a seed summary.
        deterministic_warn_only:
            If True, PyTorch warns instead of crashing on nondeterministic CUDA ops.
        cublas_workspace_config:
            CuBLAS workspace config for deterministic CUDA. Use ':4096:8' by default.
        cuda_tf32:
            Whether to allow TF32. False is more reproducible.

    Returns:
        The seed value as int.
    """
    seed = int(seed)

    active_cublas_config = configure_cublas_workspace(
        deterministic=deterministic,
        workspace_config=cublas_workspace_config,
    )

    set_python_numpy_seed(seed)

    torch_available = set_torch_seed(
        seed=seed,
        deterministic=deterministic,
        benchmark=benchmark,
        deterministic_warn_only=deterministic_warn_only,
        cuda_tf32=cuda_tf32,
    )

    if verbose:
        print("=" * 80)
        print("SEED SETUP")
        print("=" * 80)
        print(f"Seed                         : {seed}")
        print(f"Deterministic                : {deterministic}")
        print(f"Deterministic warn only       : {deterministic_warn_only}")
        print(f"cuDNN benchmark              : {benchmark}")
        print(f"CUBLAS_WORKSPACE_CONFIG      : {active_cublas_config}")
        print(f"CUDA TF32 allowed            : {cuda_tf32}")
        print(f"PyTorch available            : {torch_available}")
        print("=" * 80)

    return seed


def setup_seed_from_config(
    config: Mapping[str, Any],
    seed: Optional[int] = None,
    verbose: bool = True,
) -> int:
    """
    Set seed using config.

    If seed is provided, it overrides config seed for the current run.
    This is useful for multi-seed loops.
    """
    if seed is None:
        seed = int(_get_nested(config, "seed.single_seed", 42))

    deterministic = bool(_get_nested(config, "seed.deterministic", True))
    benchmark = bool(_get_nested(config, "seed.benchmark", False))
    deterministic_warn_only = bool(
        _get_nested(config, "seed.deterministic_warn_only", True)
    )
    cublas_workspace_config = _get_nested(
        config,
        "seed.cublas_workspace_config",
        ":4096:8",
    )
    cuda_tf32 = bool(_get_nested(config, "seed.cuda_tf32", False))

    return set_global_seed(
        seed=int(seed),
        deterministic=deterministic,
        benchmark=benchmark,
        verbose=verbose,
        deterministic_warn_only=deterministic_warn_only,
        cublas_workspace_config=cublas_workspace_config,
        cuda_tf32=cuda_tf32,
    )


def get_seed_setup_summary(
    config: Mapping[str, Any],
    seed: Optional[int] = None,
) -> SeedSetupSummary:
    """Return JSON-safe seed setup summary without changing state."""
    if seed is None:
        seed = int(_get_nested(config, "seed.single_seed", 42))

    deterministic = bool(_get_nested(config, "seed.deterministic", True))
    benchmark = bool(_get_nested(config, "seed.benchmark", False))
    deterministic_warn_only = bool(
        _get_nested(config, "seed.deterministic_warn_only", True)
    )
    cublas_workspace_config = _get_nested(
        config,
        "seed.cublas_workspace_config",
        ":4096:8",
    )
    cuda_tf32 = bool(_get_nested(config, "seed.cuda_tf32", False))

    return SeedSetupSummary(
        seed=int(seed),
        deterministic=deterministic,
        benchmark=benchmark,
        deterministic_warn_only=deterministic_warn_only,
        cublas_workspace_config=None if cublas_workspace_config is None else str(cublas_workspace_config),
        cuda_tf32=cuda_tf32,
        status="PASSED",
    )


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.

    Usage:
        DataLoader(..., worker_init_fn=seed_worker)

    This keeps NumPy/Python randomness reproducible inside each worker.
    """
    try:
        import torch

        worker_seed = torch.initial_seed() % 2**32
    except ImportError:
        worker_seed = int(worker_id)

    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(
    seed: int,
    device: Optional[str] = None,
) -> Optional[Any]:
    """
    Create a torch.Generator seeded with the given seed.

    Args:
        seed:
            Random seed.
        device:
            Optional generator device. Use None/CPU for DataLoader compatibility.

    Returns:
        torch.Generator or None if PyTorch is unavailable.
    """
    try:
        import torch

        if device is None:
            generator = torch.Generator()
        else:
            generator = torch.Generator(device=device)

        generator.manual_seed(int(seed))
        return generator

    except ImportError:
        return None


def resolve_seed_list(config: Mapping[str, Any]) -> List[int]:
    """
    Resolve seed list from config.

    Expected config:

        seed:
          mode: "single"
          single_seed: 42
          multi_seeds: [11, 22, 33, 44, 55]

    Returns:
        List of integer seeds.
    """
    mode = str(_get_nested(config, "seed.mode", "single")).lower().strip()

    if mode not in {"single", "multi"}:
        raise ValueError(
            f"Invalid seed.mode='{mode}'. Expected 'single' or 'multi'."
        )

    if mode == "single":
        single_seed = _get_nested(config, "seed.single_seed", 42)
        return [int(single_seed)]

    multi_seeds = _get_nested(config, "seed.multi_seeds", None)

    if multi_seeds is None:
        raise ValueError("seed.mode is 'multi', but seed.multi_seeds is missing.")

    if not isinstance(multi_seeds, Iterable) or isinstance(multi_seeds, (str, bytes)):
        raise TypeError("seed.multi_seeds must be a list of integers.")

    seeds = [int(seed) for seed in multi_seeds]

    if len(seeds) == 0:
        raise ValueError("seed.multi_seeds is empty.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("seed.multi_seeds contains duplicate seed values.")

    return seeds


def print_seed_plan(config: Mapping[str, Any]) -> None:
    """
    Print seed execution plan.

    This verifies from console whether we are running single-seed or multi-seed.
    """
    mode = str(_get_nested(config, "seed.mode", "single")).lower().strip()
    seeds = resolve_seed_list(config)

    print("=" * 80)
    print("SEED PLAN")
    print("=" * 80)
    print(f"Seed mode                    : {mode}")
    print(f"Number of runs               : {len(seeds)}")
    print(f"Seeds                        : {seeds}")
    print(f"Deterministic                : {_get_nested(config, 'seed.deterministic', True)}")
    print(f"cuDNN benchmark              : {_get_nested(config, 'seed.benchmark', False)}")
    print(f"Deterministic warn only       : {_get_nested(config, 'seed.deterministic_warn_only', True)}")
    print(f"CUBLAS_WORKSPACE_CONFIG      : {_get_nested(config, 'seed.cublas_workspace_config', ':4096:8')}")
    print(f"CUDA TF32 allowed            : {_get_nested(config, 'seed.cuda_tf32', False)}")
    print("=" * 80)


def reproducibility_environment_summary() -> dict:
    """Return JSON-safe reproducibility environment summary."""
    payload = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }

    try:
        import torch

        payload.update(
            {
                "torch_available": True,
                "torch_version": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            }
        )

        try:
            payload["deterministic_algorithms_enabled"] = bool(
                torch.are_deterministic_algorithms_enabled()
            )
        except Exception:
            payload["deterministic_algorithms_enabled"] = None

    except ImportError:
        payload.update(
            {
                "torch_available": False,
                "torch_version": None,
                "cuda_available": False,
                "cuda_device_count": 0,
            }
        )

    return payload


# Compatibility alias for future trainer code.
seed_everything = set_global_seed


__all__ = [
    "SeedSetupSummary",
    "configure_cublas_workspace",
    "set_python_numpy_seed",
    "set_torch_seed",
    "set_global_seed",
    "setup_seed_from_config",
    "get_seed_setup_summary",
    "seed_worker",
    "make_torch_generator",
    "resolve_seed_list",
    "print_seed_plan",
    "reproducibility_environment_summary",
    "seed_everything",
]