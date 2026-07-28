"""
Device utilities for the AV-GPS causal spoofing detection project.

Step 12 updates:
- stronger GPU/CPU device setup for training,
- mixed precision configuration helper,
- recursive batch transfer to device,
- CUDA memory summaries for logs,
- compatibility with previous Step 1-11 code.

The trainer will use:
- setup_device_from_config()
- move_to_device()
- get_mixed_precision_config()
- clear_cuda_cache()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional


def _get_nested(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Lightweight nested config getter.

    Example:
        _get_nested(cfg, "device.preference", "cuda")
    """
    current: Any = config

    for key in key_path.split("."):
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default

    return current


@dataclass
class DeviceInfo:
    """Container for device information."""

    device: Any
    device_type: str
    use_gpu: bool
    cuda_available: bool
    gpu_name: str
    gpu_count: int
    gpu_index: int = 0
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "device": str(self.device),
            "device_type": self.device_type,
            "use_gpu": bool(self.use_gpu),
            "cuda_available": bool(self.cuda_available),
            "gpu_name": self.gpu_name,
            "gpu_count": int(self.gpu_count),
            "gpu_index": int(self.gpu_index),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


@dataclass
class MixedPrecisionConfig:
    """Configuration for automatic mixed precision."""

    enabled: bool = False
    dtype: str = "float16"
    use_grad_scaler: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def torch_available() -> bool:
    """Return True if PyTorch is installed."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def cuda_available() -> bool:
    """Return True if CUDA is available through PyTorch."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def get_gpu_name(index: int = 0) -> str:
    """Return GPU name if available, otherwise 'N/A'."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > index:
            return str(torch.cuda.get_device_name(index))
        return "N/A"
    except ImportError:
        return "N/A"


def get_gpu_count() -> int:
    """Return number of CUDA devices visible to PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 0
    except ImportError:
        return 0


def get_torch_version() -> Optional[str]:
    """Return PyTorch version if available."""
    try:
        import torch

        return str(torch.__version__)
    except ImportError:
        return None


def get_cuda_version() -> Optional[str]:
    """Return CUDA version used by PyTorch if available."""
    try:
        import torch

        return None if torch.version.cuda is None else str(torch.version.cuda)
    except ImportError:
        return None


def select_device(
    preference: str = "cuda",
    gpu_index: int = 0,
    allow_cpu_fallback: bool = True,
    verbose: bool = True,
) -> DeviceInfo:
    """
    Select compute device.

    Args:
        preference:
            "cuda", "cpu", or "auto".
        gpu_index:
            CUDA device index, usually 0.
        allow_cpu_fallback:
            If True, use CPU when CUDA is requested but unavailable.
        verbose:
            If True, print device summary.

    Returns:
        DeviceInfo object.
    """
    preference = str(preference).lower().strip()
    gpu_index = int(gpu_index)

    if preference not in {"cuda", "cpu", "auto"}:
        raise ValueError(
            f"Invalid device preference='{preference}'. Expected 'cuda', 'cpu', or 'auto'."
        )

    try:
        import torch
    except ImportError as exc:
        if allow_cpu_fallback:
            if verbose:
                print("=" * 80)
                print("DEVICE SETUP")
                print("=" * 80)
                print("PyTorch not installed. Device set to CPU-like mode.")
                print("=" * 80)

            return DeviceInfo(
                device="cpu",
                device_type="cpu",
                use_gpu=False,
                cuda_available=False,
                gpu_name="N/A",
                gpu_count=0,
                gpu_index=0,
                torch_version=None,
                cuda_version=None,
            )

        raise ImportError("PyTorch is required for device selection.") from exc

    has_cuda = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if has_cuda else 0

    if preference == "cpu":
        device = torch.device("cpu")
        use_gpu = False

    elif preference in {"cuda", "auto"}:
        if has_cuda:
            if gpu_index < 0 or gpu_index >= gpu_count:
                raise ValueError(
                    f"Invalid gpu_index={gpu_index}. Available CUDA device count: {gpu_count}"
                )

            device = torch.device(f"cuda:{gpu_index}")
            use_gpu = True
            torch.cuda.set_device(device)

        else:
            if allow_cpu_fallback:
                device = torch.device("cpu")
                use_gpu = False
            else:
                raise RuntimeError("CUDA was requested but is not available.")

    else:
        device = torch.device("cpu")
        use_gpu = False

    gpu_name = get_gpu_name(gpu_index) if use_gpu else "N/A"

    info = DeviceInfo(
        device=device,
        device_type=str(device).split(":")[0],
        use_gpu=use_gpu,
        cuda_available=has_cuda,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        gpu_index=gpu_index,
        torch_version=get_torch_version(),
        cuda_version=get_cuda_version(),
    )

    if verbose:
        print_device_info(info)

    return info


def setup_device_from_config(
    config: Mapping[str, Any],
    verbose: bool = True,
) -> DeviceInfo:
    """
    Select device using config.

    Expected config:

        device:
          preference: "cuda"
          gpu_index: 0
          allow_cpu_fallback: true
          require_gpu: false
    """
    preference = _get_nested(config, "device.preference", "cuda")
    gpu_index = int(_get_nested(config, "device.gpu_index", 0))
    allow_cpu_fallback = bool(_get_nested(config, "device.allow_cpu_fallback", True))

    info = select_device(
        preference=preference,
        gpu_index=gpu_index,
        allow_cpu_fallback=allow_cpu_fallback,
        verbose=verbose,
    )

    assert_gpu_if_required(config)

    return info


def print_device_info(info: DeviceInfo) -> None:
    """Print a clear device summary."""
    print("=" * 80)
    print("DEVICE SETUP")
    print("=" * 80)
    print(f"Selected device              : {info.device}")
    print(f"Device type                  : {info.device_type}")
    print(f"Use GPU                      : {info.use_gpu}")
    print(f"CUDA available               : {info.cuda_available}")
    print(f"GPU count                    : {info.gpu_count}")
    print(f"GPU index                    : {info.gpu_index}")
    print(f"GPU name                     : {info.gpu_name}")
    print(f"PyTorch version              : {info.torch_version}")
    print(f"CUDA version                 : {info.cuda_version}")
    print("=" * 80)


def device_to_string(device: Any) -> str:
    """Return a safe string representation of device."""
    return str(device)


def is_cuda_device(device: Any) -> bool:
    """Return True if device is CUDA."""
    return str(device).startswith("cuda")


def move_to_device(obj: Any, device: Any, non_blocking: bool = True) -> Any:
    """
    Recursively move PyTorch tensors/modules inside an object to device.

    Supports:
    - tensors/modules with .to(device),
    - dict,
    - list,
    - tuple.

    Non-PyTorch objects are returned unchanged.
    """
    if isinstance(obj, Mapping):
        return {
            key: move_to_device(value, device=device, non_blocking=non_blocking)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(item, device=device, non_blocking=non_blocking)
            for item in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(item, device=device, non_blocking=non_blocking)
            for item in obj
        )

    if hasattr(obj, "to"):
        try:
            return obj.to(device=device, non_blocking=non_blocking)
        except TypeError:
            try:
                return obj.to(device)
            except Exception:
                return obj
        except Exception:
            return obj

    return obj


def clear_cuda_cache() -> None:
    """Clear CUDA cache if available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def synchronize_cuda(device: Optional[Any] = None) -> None:
    """Synchronize CUDA if available."""
    try:
        import torch

        if torch.cuda.is_available():
            if device is None:
                torch.cuda.synchronize()
            else:
                torch.cuda.synchronize(device)
    except ImportError:
        pass


def get_cuda_memory_stats(device: Optional[Any] = None) -> dict:
    """Return JSON-safe CUDA memory stats."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "max_allocated_mb": 0.0,
                "max_reserved_mb": 0.0,
            }

        if device is None:
            device = torch.device("cuda:0")

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)

        mb = 1024.0 * 1024.0

        return {
            "cuda_available": True,
            "device": str(device),
            "allocated_mb": float(allocated / mb),
            "reserved_mb": float(reserved / mb),
            "max_allocated_mb": float(max_allocated / mb),
            "max_reserved_mb": float(max_reserved / mb),
        }

    except ImportError:
        return {
            "cuda_available": False,
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
        }


def reset_cuda_peak_memory_stats(device: Optional[Any] = None) -> None:
    """Reset CUDA peak memory stats."""
    try:
        import torch

        if torch.cuda.is_available():
            if device is None:
                device = torch.device("cuda:0")
            torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        pass


def get_cuda_memory_summary(device: Optional[Any] = None) -> str:
    """
    Return CUDA memory summary if CUDA is available.

    Useful for debugging GPU memory usage.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA is not available."

        if device is None:
            device = torch.device("cuda:0")

        return torch.cuda.memory_summary(device=device)

    except ImportError:
        return "PyTorch is not installed."


def print_cuda_memory_summary(device: Optional[Any] = None) -> None:
    """Print CUDA memory summary if available."""
    print("=" * 80)
    print("CUDA MEMORY SUMMARY")
    print("=" * 80)
    print(get_cuda_memory_summary(device=device))
    print("=" * 80)


def get_mixed_precision_config(config: Mapping[str, Any]) -> MixedPrecisionConfig:
    """
    Read mixed precision settings from config.

    Expected config:

        training:
          mixed_precision:
            enabled: false
            dtype: "float16"
            use_grad_scaler: true
    """
    return MixedPrecisionConfig(
        enabled=bool(_get_nested(config, "training.mixed_precision.enabled", False)),
        dtype=str(_get_nested(config, "training.mixed_precision.dtype", "float16")),
        use_grad_scaler=bool(
            _get_nested(config, "training.mixed_precision.use_grad_scaler", True)
        ),
    )


def get_autocast_dtype(dtype_name: str) -> Any:
    """Convert dtype name to torch dtype for autocast."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for mixed precision.") from exc

    dtype_name = str(dtype_name).lower().strip()

    if dtype_name in {"float16", "fp16", "half"}:
        return torch.float16

    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16

    if dtype_name in {"float32", "fp32"}:
        return torch.float32

    raise ValueError(
        f"Unknown mixed precision dtype='{dtype_name}'. "
        "Supported: float16, bfloat16, float32."
    )


def create_grad_scaler(config: Mapping[str, Any], device: Any) -> Any:
    """
    Create GradScaler for AMP training.

    Returns:
        torch.cuda.amp.GradScaler or None.
    """
    mp_config = get_mixed_precision_config(config)

    if not mp_config.enabled:
        return None

    if not is_cuda_device(device):
        return None

    if not mp_config.use_grad_scaler:
        return None

    try:
        import torch

        return torch.cuda.amp.GradScaler(enabled=True)
    except ImportError:
        return None


def autocast_context(config: Mapping[str, Any], device: Any) -> Any:
    """
    Return an autocast context manager.

    Usage:
        with autocast_context(config, device):
            output = model(batch)
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for autocast.") from exc

    mp_config = get_mixed_precision_config(config)

    enabled = bool(mp_config.enabled and is_cuda_device(device))
    dtype = get_autocast_dtype(mp_config.dtype)

    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def require_gpu(config: Mapping[str, Any]) -> bool:
    """
    Check whether config requires GPU strictly.

    Expected config:
        device:
          require_gpu: false
    """
    return bool(_get_nested(config, "device.require_gpu", False))


def assert_gpu_if_required(config: Mapping[str, Any]) -> None:
    """Raise an error if config requires GPU but CUDA is unavailable."""
    if require_gpu(config) and not cuda_available():
        raise RuntimeError("device.require_gpu=true, but CUDA is not available.")


def device_environment_summary(info: Optional[DeviceInfo] = None) -> dict:
    """Return JSON-safe device environment summary."""
    summary = {
        "torch_available": torch_available(),
        "cuda_available": cuda_available(),
        "gpu_count": get_gpu_count(),
        "gpu_name": get_gpu_name(0) if cuda_available() else "N/A",
        "torch_version": get_torch_version(),
        "cuda_version": get_cuda_version(),
    }

    if info is not None:
        summary["selected_device"] = info.as_dict()

    return summary


__all__ = [
    "DeviceInfo",
    "MixedPrecisionConfig",
    "torch_available",
    "cuda_available",
    "get_gpu_name",
    "get_gpu_count",
    "get_torch_version",
    "get_cuda_version",
    "select_device",
    "setup_device_from_config",
    "print_device_info",
    "device_to_string",
    "is_cuda_device",
    "move_to_device",
    "clear_cuda_cache",
    "synchronize_cuda",
    "get_cuda_memory_stats",
    "reset_cuda_peak_memory_stats",
    "get_cuda_memory_summary",
    "print_cuda_memory_summary",
    "get_mixed_precision_config",
    "get_autocast_dtype",
    "create_grad_scaler",
    "autocast_context",
    "require_gpu",
    "assert_gpu_if_required",
    "device_environment_summary",
]