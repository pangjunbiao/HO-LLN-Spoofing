"""
Logging utilities for the AV-GPS causal spoofing detection project.

This file is responsible for:
- creating a console + file logger,
- appending run information to logs/run_log.txt,
- appending one-row summaries to logs/experiment_history.csv,
- saving errors/exceptions to logs/errors.log,
- keeping every run traceable with date/time, mode, seed, device, and status.

Step 1 goal:
Make sure every project run leaves a clear trace in the logs directory.
"""

from __future__ import annotations

import csv
import logging
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Union

from src.utils.io import ensure_dir, ensure_parent_dir


PathLike = Union[str, Path]


def _get_nested(config: Mapping[str, Any], key_path: str, default: Any = None) -> Any:
    """Small nested config getter used only inside logging utilities."""
    current: Any = config

    for key in key_path.split("."):
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default

    return current


def now_string(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current date/time string."""
    return datetime.now().strftime(fmt)


def make_run_id(prefix: str = "run") -> str:
    """Create unique run id from timestamp."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def setup_logger(
    name: str,
    log_file: PathLike,
    level: int = logging.INFO,
    console: bool = True,
    reset_handlers: bool = True,
) -> logging.Logger:
    """
    Set up a Python logger writing to console and a file.

    Args:
        name:
            Logger name.
        log_file:
            File where log messages are appended.
        level:
            Logging level.
        console:
            If True, also print logs to console.
        reset_handlers:
            If True, remove old handlers to avoid duplicate logs.

    Returns:
        Configured logger.
    """
    log_file = ensure_parent_dir(log_file)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if reset_handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def append_run_log_header(
    run_log_path: PathLike,
    run_id: str,
    mode: str,
    seed_mode: str,
    active_seed: Optional[int],
    device_info: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str] = None,
) -> None:
    """
    Append a readable run-start block to logs/run_log.txt.
    """
    run_log_path = ensure_parent_dir(run_log_path)
    device_info = device_info or {}

    lines = [
        "",
        "=" * 100,
        f"Run started        : {now_string()}",
        f"Run ID             : {run_id}",
        f"Mode               : {mode}",
        f"Config             : {config_path or 'N/A'}",
        f"Seed mode          : {seed_mode}",
        f"Active seed        : {active_seed if active_seed is not None else 'N/A'}",
        f"Device             : {device_info.get('device', 'N/A')}",
        f"Use GPU            : {device_info.get('use_gpu', 'N/A')}",
        f"CUDA available     : {device_info.get('cuda_available', 'N/A')}",
        f"GPU name           : {device_info.get('gpu_name', 'N/A')}",
        "-" * 100,
    ]

    with run_log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def append_run_log_footer(
    run_log_path: PathLike,
    status: str,
    message: str = "",
) -> None:
    """
    Append a readable run-end block to logs/run_log.txt.
    """
    run_log_path = ensure_parent_dir(run_log_path)

    lines = [
        "-" * 100,
        f"Run ended          : {now_string()}",
        f"Status             : {status}",
    ]

    if message:
        lines.append(f"Message            : {message}")

    lines.append("=" * 100)

    with run_log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def append_experiment_history(
    history_path: PathLike,
    row: Mapping[str, Any],
) -> None:
    """
    Append one row to logs/experiment_history.csv.

    If file does not exist, header is created automatically.
    """
    history_path = ensure_parent_dir(history_path)

    fieldnames = [
        "datetime",
        "run_id",
        "mode",
        "seed_mode",
        "active_seed",
        "device",
        "use_gpu",
        "cuda_available",
        "gpu_name",
        "status",
        "message",
    ]

    file_exists = history_path.exists()

    clean_row = {field: row.get(field, "") for field in fieldnames}

    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(clean_row)


def append_error_log(
    error_log_path: PathLike,
    run_id: str,
    error: BaseException,
) -> None:
    """
    Append exception traceback to logs/errors.log.
    """
    error_log_path = ensure_parent_dir(error_log_path)

    lines = [
        "",
        "=" * 100,
        f"Error time         : {now_string()}",
        f"Run ID             : {run_id}",
        f"Error type         : {type(error).__name__}",
        f"Error message      : {str(error)}",
        "-" * 100,
        traceback.format_exc(),
        "=" * 100,
    ]

    with error_log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def get_log_paths_from_config(config: Mapping[str, Any]) -> Dict[str, Path]:
    """
    Resolve log paths from config.

    Expected config keys:
        paths.logs_dir
        logging.run_log_file
        logging.history_file
        logging.error_log_file
    """
    project_root = Path(_get_nested(config, "project.root", ".")).resolve()

    logs_dir = Path(_get_nested(config, "paths.logs_dir", "logs"))
    if not logs_dir.is_absolute():
        logs_dir = project_root / logs_dir

    ensure_dir(logs_dir)

    run_log_file = _get_nested(config, "logging.run_log_file", "run_log.txt")
    history_file = _get_nested(config, "logging.history_file", "experiment_history.csv")
    error_log_file = _get_nested(config, "logging.error_log_file", "errors.log")

    return {
        "logs_dir": logs_dir.resolve(),
        "run_log": (logs_dir / run_log_file).resolve(),
        "history": (logs_dir / history_file).resolve(),
        "errors": (logs_dir / error_log_file).resolve(),
    }


class RunLogger:
    """
    Small run-level logger manager.

    This is used by main.py to make sure every run has:
    - console/file logger,
    - run_log header/footer,
    - experiment_history row,
    - error log on failure.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        run_id: str,
        mode: str,
        seed_mode: str,
        active_seed: Optional[int],
        device_info: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.run_id = run_id
        self.mode = mode
        self.seed_mode = seed_mode
        self.active_seed = active_seed
        self.device_info = device_info or {}
        self.config_path = config_path

        self.log_paths = get_log_paths_from_config(config)
        self.logger = setup_logger(
            name="av_gps_project",
            log_file=self.log_paths["run_log"],
            level=logging.INFO,
            console=bool(_get_nested(config, "logging.console", True)),
        )

    def start(self) -> None:
        append_run_log_header(
            run_log_path=self.log_paths["run_log"],
            run_id=self.run_id,
            mode=self.mode,
            seed_mode=self.seed_mode,
            active_seed=self.active_seed,
            device_info=self.device_info,
            config_path=self.config_path,
        )

        self.logger.info("Run started.")
        self.logger.info("Run ID: %s", self.run_id)
        self.logger.info("Mode: %s", self.mode)
        self.logger.info("Seed mode: %s", self.seed_mode)
        self.logger.info("Active seed: %s", self.active_seed)
        self.logger.info("Device: %s", self.device_info.get("device", "N/A"))
        self.logger.info("Use GPU: %s", self.device_info.get("use_gpu", "N/A"))
        self.logger.info("GPU name: %s", self.device_info.get("gpu_name", "N/A"))

    def finish(self, status: str, message: str = "") -> None:
        self.logger.info("Run finished with status: %s", status)

        if message:
            self.logger.info("Message: %s", message)

        append_run_log_footer(
            run_log_path=self.log_paths["run_log"],
            status=status,
            message=message,
        )

        append_experiment_history(
            history_path=self.log_paths["history"],
            row={
                "datetime": now_string(),
                "run_id": self.run_id,
                "mode": self.mode,
                "seed_mode": self.seed_mode,
                "active_seed": self.active_seed,
                "device": self.device_info.get("device", ""),
                "use_gpu": self.device_info.get("use_gpu", ""),
                "cuda_available": self.device_info.get("cuda_available", ""),
                "gpu_name": self.device_info.get("gpu_name", ""),
                "status": status,
                "message": message,
            },
        )

    def log_error(self, error: BaseException) -> None:
        self.logger.exception("Run failed due to an exception.")
        append_error_log(
            error_log_path=self.log_paths["errors"],
            run_id=self.run_id,
            error=error,
        )


@contextmanager
def managed_run_logger(
    config: Mapping[str, Any],
    run_id: str,
    mode: str,
    seed_mode: str,
    active_seed: Optional[int],
    device_info: Optional[Mapping[str, Any]] = None,
    config_path: Optional[str] = None,
) -> Iterator[RunLogger]:
    """
    Context manager around RunLogger.

    Usage:
        with managed_run_logger(...) as run_logger:
            run_logger.logger.info("Something")
    """
    run_logger = RunLogger(
        config=config,
        run_id=run_id,
        mode=mode,
        seed_mode=seed_mode,
        active_seed=active_seed,
        device_info=device_info,
        config_path=config_path,
    )

    run_logger.start()

    try:
        yield run_logger
    except Exception as error:
        run_logger.log_error(error)
        run_logger.finish(status="FAILED", message=str(error))
        raise
    else:
        run_logger.finish(status="SUCCESS", message="Step completed successfully.")