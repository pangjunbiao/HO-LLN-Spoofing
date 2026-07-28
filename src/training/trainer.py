"""
General PyTorch trainer for causal AV-GPS spoofing detection.

Step 12:
This file implements the common training protocol for:
- full proposed model,
- official ablations,
- later PyTorch baselines.

Supported:
- weighted BCE from logits,
- class weights computed from TRAIN split only,
- valid-row and padding masks,
- GPU training,
- optional mixed precision,
- optimizer/scheduler,
- early stopping,
- best checkpoint saving,
- train/validation history saving,
- same protocol for full model and ablations.

Important:
- This trainer does not select theta or N_p.
- The synthetic Step-10 theta=0.55 is never used here.
- After training, validation probabilities are saved so the Step-10 selector can
  later select the real theta and N_p from validation predictions.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import compute_ranking_metrics
from src.models.model_factory import build_model
from src.training.early_stopping import EarlyStopping, build_early_stopping_config
from src.training.losses import (
    ClassWeightSummary,
    WeightedBCEWithLogitsLoss,
    build_class_weight_config,
    compute_balanced_class_weights,
    labels_to_binary_tensor,
)
from src.training.optimizer import (
    GradientClipResult,
    build_gradient_clipping_config,
    build_optimizer_config,
    clip_gradients,
    create_optimizer,
    optimizer_state_summary,
)
from src.training.scheduler import SchedulerController, build_scheduler_config
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import (
    autocast_context,
    clear_cuda_cache,
    create_grad_scaler,
    get_cuda_memory_stats,
    move_to_device,
    reset_cuda_peak_memory_stats,
    setup_device_from_config,
    synchronize_cuda,
)
from src.utils.io import ensure_dir, save_json
from src.utils.seed import make_torch_generator, seed_worker


DEFAULT_STEP12_FEATURE_COLUMNS = [
    "xi_eta_east_scaled",
    "xi_eta_north_scaled",
    "xi_eta_dot_east_scaled",
    "xi_eta_dot_north_scaled",
    "xi_eta_ddot_east_scaled",
    "xi_eta_ddot_north_scaled",
    "xi_q_scaled",
    "xi_accum_log_scaled",
    "xi_nu",
]


@dataclass
class SequenceWindowSpec:
    """Sequence window specification."""

    window_length: int = 256
    stride: int = 128
    include_last_partial: bool = True
    min_valid_rows: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step12TrainerConfig:
    """Top-level Step-12 trainer config."""

    model_name: str = "proposed"
    variant_name: str = "full"

    max_epochs: int = 30
    batch_size_train: int = 16
    batch_size_eval: int = 8
    num_workers: int = 0
    pin_memory: bool = True

    train_window: SequenceWindowSpec = field(default_factory=SequenceWindowSpec)
    eval_window: SequenceWindowSpec = field(
        default_factory=lambda: SequenceWindowSpec(
            window_length=512,
            stride=512,
            include_last_partial=True,
            min_valid_rows=1,
        )
    )

    checkpoint_dir: str = "results/models"
    best_checkpoint_name: str = "proposed_best.pt"
    last_checkpoint_name: str = "proposed_last.pt"

    history_csv: str = "results/tables/step12_training_history.csv"
    history_json: str = "results/tables/step12_training_history.json"
    summary_json: str = "results/tables/step12_training_summary.json"
    validation_predictions_npz: str = "results/tables/step12_validation_predictions.npz"

    monitor: str = "val_loss"

    independent_windows: bool = True
    carry_hidden_state_across_windows: bool = False

    save_best: bool = True
    save_last: bool = True
    save_validation_predictions: bool = True

    log_every_n_batches: int = 25

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["train_window"] = self.train_window.to_dict()
        payload["eval_window"] = self.eval_window.to_dict()
        return payload


@dataclass
class EpochResult:
    """Metrics for one epoch and split."""

    split: str
    epoch: int

    loss: float
    unweighted_loss: float

    valid_count: int
    positive_count: int
    negative_count: int

    auprc: Optional[float]
    auroc: Optional[float]

    runtime_seconds: float

    probability_mean: float
    probability_min: float
    probability_max: float

    learning_rate: Optional[float] = None
    grad_norm_mean: Optional[float] = None
    grad_norm_max: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingRunSummary:
    """JSON-safe final training run summary."""

    model_name: str
    variant_name: str
    active_seed: int

    final_status: str
    best_epoch: Optional[int]
    best_monitor: str
    best_monitor_value: Optional[float]

    epochs_completed: int
    stopped_early: bool
    stop_reason: Optional[str]

    best_checkpoint_path: Optional[str]
    last_checkpoint_path: Optional[str]

    class_weights: Dict[str, Any]
    optimizer: Dict[str, Any]
    scheduler: Dict[str, Any]
    early_stopping: Dict[str, Any]
    trainer_config: Dict[str, Any]

    train_dataset_summary: Dict[str, Any]
    val_dataset_summary: Dict[str, Any]

    final_cuda_memory: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to JSON-safe float or None."""
    if value is None:
        return None

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _get_project_path(config: Mapping[str, Any], value: str) -> Path:
    """Resolve path relative to project root."""
    return resolve_project_path(config, value)


def build_step12_trainer_config(config: Mapping[str, Any]) -> Step12TrainerConfig:
    """Build Step12TrainerConfig from project config."""
    train_window = SequenceWindowSpec(
        window_length=int(get_by_path(config, "training.dataset.sequence.window_length_train", 256)),
        stride=int(get_by_path(config, "training.dataset.sequence.stride_train", 128)),
        include_last_partial=bool(
            get_by_path(config, "training.dataset.sequence.include_last_partial_train", True)
        ),
        min_valid_rows=int(get_by_path(config, "training.dataset.sequence.min_valid_rows", 1)),
    )

    eval_window = SequenceWindowSpec(
        window_length=int(get_by_path(config, "training.dataset.sequence.window_length_eval", 512)),
        stride=int(get_by_path(config, "training.dataset.sequence.stride_eval", 512)),
        include_last_partial=bool(
            get_by_path(config, "training.dataset.sequence.include_last_partial_eval", True)
        ),
        min_valid_rows=int(get_by_path(config, "training.dataset.sequence.min_valid_rows", 1)),
    )

    model_name = str(get_by_path(config, "training.step12.model_name", "proposed"))
    variant_name = str(get_by_path(config, "training.step12.variant_name", "full"))

    return Step12TrainerConfig(
        model_name=model_name,
        variant_name=variant_name,
        max_epochs=int(get_by_path(config, "training.trainer.max_epochs", 30)),
        batch_size_train=int(get_by_path(config, "training.dataloader.batch_size_train", 16)),
        batch_size_eval=int(get_by_path(config, "training.dataloader.batch_size_eval", 8)),
        num_workers=int(get_by_path(config, "training.dataloader.num_workers", 0)),
        pin_memory=bool(get_by_path(config, "training.dataloader.pin_memory", True)),
        train_window=train_window,
        eval_window=eval_window,
        checkpoint_dir=str(get_by_path(config, "paths.models_dir", "results/models")),
        best_checkpoint_name=str(
            get_by_path(config, "training.checkpointing.best_checkpoint_name", "proposed_best.pt")
        ),
        last_checkpoint_name=str(
            get_by_path(config, "training.checkpointing.last_checkpoint_name", "proposed_last.pt")
        ),
        history_csv=str(
            get_by_path(config, "paths.step12_training_history_csv", "results/tables/step12_training_history.csv")
        ),
        history_json=str(
            get_by_path(config, "paths.step12_training_history_json", "results/tables/step12_training_history.json")
        ),
        summary_json=str(
            get_by_path(config, "paths.step12_training_summary_json", "results/tables/step12_training_summary.json")
        ),
        validation_predictions_npz=str(
            get_by_path(
                config,
                "paths.step12_validation_predictions_npz",
                "results/tables/step12_validation_predictions.npz",
            )
        ),
        monitor=str(get_by_path(config, "training.early_stopping.monitor", "val_loss")),
        independent_windows=bool(
            get_by_path(config, "training.state_handling.independent_windows", True)
        ),
        carry_hidden_state_across_windows=bool(
            get_by_path(config, "training.state_handling.carry_hidden_state_across_windows", False)
        ),
        save_best=bool(get_by_path(config, "training.checkpointing.save_best", True)),
        save_last=bool(get_by_path(config, "training.checkpointing.save_last", True)),
        save_validation_predictions=bool(
            get_by_path(config, "training.outputs.save_validation_predictions", True)
        ),
        log_every_n_batches=int(get_by_path(config, "training.logging.log_every_n_batches", 25)),
    )


def _label_series_to_binary(labels: pd.Series) -> np.ndarray:
    """Convert labels to binary numpy array."""
    values = labels.to_numpy()

    if values.dtype.kind in {"U", "S", "O"}:
        out = []
        for item in values:
            text = str(item).strip().lower()
            out.append(1 if text in {"attack", "attacked", "spoof", "spoofing", "1", "true"} else 0)
        return np.asarray(out, dtype=np.int64)

    return (values.astype(np.float32) >= 0.5).astype(np.int64)


def _sort_split_dataframe(df: pd.DataFrame, segment_column: str, order_column: str) -> pd.DataFrame:
    """Sort split dataframe by segment and causal order."""
    if segment_column not in df.columns:
        raise KeyError(f"Missing segment column: {segment_column}")

    if order_column in df.columns:
        return df.sort_values([segment_column, order_column], kind="mergesort").reset_index(drop=True)

    return df.sort_values([segment_column], kind="mergesort").reset_index(drop=True)


def _resolve_split_csv_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve processed xi split CSV path."""
    default_map = {
        "train": "data/processed/train_xi.csv",
        "val": "data/processed/val_xi.csv",
        "test": "data/processed/test_xi.csv",
        "external": "data/processed/external_xi.csv",
        "online": "data/processed/online_xi.csv",
    }

    path_value = get_by_path(
        config,
        f"training.dataset.xi_split_files.{split_name}",
        default_map[split_name],
    )

    return _get_project_path(config, str(path_value))


class XiWindowDataset(Dataset):
    """
    Sequence-window dataset loaded from processed xi CSV.

    Each item is a fixed-length padded window:
        x:            [T,9]
        y:            [T]
        loss_mask:    [T] xi_nu valid rows
        padding_mask: [T] real rows
        delta_t:      [T]
        reset_state:  scalar 1, because windows are independent by default
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        split_name: str,
        feature_columns: Sequence[str],
        label_column: str,
        validity_column: str,
        segment_column: str,
        order_column: str,
        delta_t_column: str,
        window_spec: SequenceWindowSpec,
        independent_windows: bool = True,
    ) -> None:
        super().__init__()

        self.df = _sort_split_dataframe(
            dataframe.copy(),
            segment_column=segment_column,
            order_column=order_column,
        )

        self.split_name = str(split_name)
        self.feature_columns = list(feature_columns)
        self.label_column = str(label_column)
        self.validity_column = str(validity_column)
        self.segment_column = str(segment_column)
        self.order_column = str(order_column)
        self.delta_t_column = str(delta_t_column)
        self.window_spec = window_spec
        self.independent_windows = bool(independent_windows)

        self._validate_columns()
        self.windows = self._build_windows()

        if len(self.windows) == 0:
            raise ValueError(f"No windows created for split={self.split_name}.")

    def _validate_columns(self) -> None:
        required = (
            list(self.feature_columns)
            + [
                self.label_column,
                self.validity_column,
                self.segment_column,
                self.delta_t_column,
            ]
        )

        missing = [column for column in required if column not in self.df.columns]

        if missing:
            raise KeyError(f"Missing required columns for {self.split_name}: {missing}")

        if len(self.feature_columns) != 9:
            raise ValueError(
                f"Expected exactly 9 feature columns, got {len(self.feature_columns)}."
            )

    def _build_windows(self) -> List[Tuple[int, int, str, int]]:
        """
        Build windows.

        Returns list of:
            (start_global_index, end_global_index_exclusive, segment_id, local_window_id)
        """
        windows: List[Tuple[int, int, str, int]] = []

        window_length = int(self.window_spec.window_length)
        stride = int(self.window_spec.stride)

        if window_length <= 0 or stride <= 0:
            raise ValueError("window_length and stride must be positive.")

        grouped = self.df.groupby(self.segment_column, sort=False)

        for segment_id, segment_df in grouped:
            indices = segment_df.index.to_numpy()
            length = int(len(indices))

            local_window_id = 0
            start_local = 0

            while start_local < length:
                end_local = min(start_local + window_length, length)

                if end_local <= start_local:
                    break

                real_count = end_local - start_local

                if real_count < window_length and not self.window_spec.include_last_partial:
                    break

                global_start = int(indices[start_local])
                global_end = int(indices[end_local - 1]) + 1

                valid_values = self.df.iloc[global_start:global_end][self.validity_column].to_numpy()
                valid_count = int(np.nansum(valid_values.astype(np.float32) > 0.5))

                if valid_count >= int(self.window_spec.min_valid_rows):
                    windows.append(
                        (
                            global_start,
                            global_end,
                            str(segment_id),
                            int(local_window_id),
                        )
                    )
                    local_window_id += 1

                if end_local >= length:
                    break

                start_local += stride

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        start, end, segment_id, local_window_id = self.windows[index]

        window_df = self.df.iloc[start:end]

        real_length = int(len(window_df))
        max_length = int(self.window_spec.window_length)

        x = np.zeros((max_length, len(self.feature_columns)), dtype=np.float32)
        y = np.zeros((max_length,), dtype=np.int64)
        loss_mask = np.zeros((max_length,), dtype=np.float32)
        padding_mask = np.zeros((max_length,), dtype=np.float32)
        delta_t = np.ones((max_length,), dtype=np.float32)

        x_real = window_df[self.feature_columns].to_numpy(dtype=np.float32)
        y_real = _label_series_to_binary(window_df[self.label_column])
        loss_mask_real = (
            window_df[self.validity_column].to_numpy(dtype=np.float32) > 0.5
        ).astype(np.float32)

        if self.delta_t_column in window_df.columns:
            delta_real = window_df[self.delta_t_column].to_numpy(dtype=np.float32)
            delta_real = np.nan_to_num(delta_real, nan=0.0, posinf=5.0, neginf=0.0)
        else:
            delta_real = np.ones((real_length,), dtype=np.float32)

        x[:real_length] = x_real
        y[:real_length] = y_real
        loss_mask[:real_length] = loss_mask_real
        padding_mask[:real_length] = 1.0
        delta_t[:real_length] = delta_real

        reset_state = 1.0 if self.independent_windows or local_window_id == 0 else 0.0

        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "loss_mask": torch.from_numpy(loss_mask),
            "padding_mask": torch.from_numpy(padding_mask),
            "delta_t": torch.from_numpy(delta_t),
            "reset_state": torch.tensor(reset_state, dtype=torch.float32),
            "segment_id": segment_id,
            "window_id": int(local_window_id),
            "start_index": int(start),
            "end_index": int(end),
            "real_length": int(real_length),
            "split": self.split_name,
        }

    def summary(self) -> Dict[str, Any]:
        labels = _label_series_to_binary(self.df[self.label_column])
        valid = (self.df[self.validity_column].to_numpy(dtype=np.float32) > 0.5)

        return {
            "split": self.split_name,
            "rows": int(len(self.df)),
            "segments": int(self.df[self.segment_column].nunique()),
            "windows": int(len(self.windows)),
            "feature_dim": int(len(self.feature_columns)),
            "normal_rows": int((labels == 0).sum()),
            "attack_rows": int((labels == 1).sum()),
            "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "window_length": int(self.window_spec.window_length),
            "stride": int(self.window_spec.stride),
            "include_last_partial": bool(self.window_spec.include_last_partial),
            "independent_windows": bool(self.independent_windows),
        }


def collate_xi_windows(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate fixed-size xi windows."""
    tensor_keys = ["x", "y", "loss_mask", "padding_mask", "delta_t", "reset_state"]

    output: Dict[str, Any] = {}

    for key in tensor_keys:
        output[key] = torch.stack([item[key] for item in batch], dim=0)

    output["segment_id"] = [item["segment_id"] for item in batch]
    output["window_id"] = torch.tensor([item["window_id"] for item in batch], dtype=torch.long)
    output["start_index"] = torch.tensor([item["start_index"] for item in batch], dtype=torch.long)
    output["end_index"] = torch.tensor([item["end_index"] for item in batch], dtype=torch.long)
    output["real_length"] = torch.tensor([item["real_length"] for item in batch], dtype=torch.long)
    output["split"] = [item["split"] for item in batch]

    return output


def build_step12_datasets(config: Mapping[str, Any], trainer_config: Step12TrainerConfig) -> Tuple[XiWindowDataset, XiWindowDataset]:
    """Build train and validation datasets from processed xi CSVs."""
    feature_columns = list(
        get_by_path(
            config,
            "training.dataset.feature_columns",
            get_by_path(config, "model.input.recommended_model_input_columns", DEFAULT_STEP12_FEATURE_COLUMNS),
        )
    )

    label_column = str(get_by_path(config, "training.dataset.label_column", "Data Type"))
    validity_column = str(get_by_path(config, "training.dataset.validity_column", "xi_nu"))
    segment_column = str(get_by_path(config, "training.dataset.segment_column", "segment_id"))
    order_column = str(get_by_path(config, "training.dataset.order_column", "within_segment_index"))
    delta_t_column = str(get_by_path(config, "training.dataset.delta_t_column", "delta_t_seconds"))

    train_path = _resolve_split_csv_path(config, "train")
    val_path = _resolve_split_csv_path(config, "val")

    if not train_path.exists():
        raise FileNotFoundError(f"Train xi CSV not found: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation xi CSV not found: {val_path}")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    train_dataset = XiWindowDataset(
        dataframe=train_df,
        split_name="train",
        feature_columns=feature_columns,
        label_column=label_column,
        validity_column=validity_column,
        segment_column=segment_column,
        order_column=order_column,
        delta_t_column=delta_t_column,
        window_spec=trainer_config.train_window,
        independent_windows=trainer_config.independent_windows,
    )

    val_dataset = XiWindowDataset(
        dataframe=val_df,
        split_name="val",
        feature_columns=feature_columns,
        label_column=label_column,
        validity_column=validity_column,
        segment_column=segment_column,
        order_column=order_column,
        delta_t_column=delta_t_column,
        window_spec=trainer_config.eval_window,
        independent_windows=True,
    )

    return train_dataset, val_dataset


def build_step12_dataloaders(
    config: Mapping[str, Any],
    trainer_config: Step12TrainerConfig,
    active_seed: int,
) -> Tuple[DataLoader, DataLoader, XiWindowDataset, XiWindowDataset]:
    """Build train/validation dataloaders."""
    train_dataset, val_dataset = build_step12_datasets(config, trainer_config)

    generator = make_torch_generator(active_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(trainer_config.batch_size_train),
        shuffle=True,
        num_workers=int(trainer_config.num_workers),
        pin_memory=bool(trainer_config.pin_memory),
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=collate_xi_windows,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(trainer_config.batch_size_eval),
        shuffle=False,
        num_workers=int(trainer_config.num_workers),
        pin_memory=bool(trainer_config.pin_memory),
        worker_init_fn=seed_worker,
        generator=make_torch_generator(active_seed + 1),
        collate_fn=collate_xi_windows,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset, val_dataset


def compute_class_weights_from_dataset(
    dataset: XiWindowDataset,
    class_weight_config: Any,
) -> ClassWeightSummary:
    """Compute class weights from TRAIN dataset only."""
    labels = _label_series_to_binary(dataset.df[dataset.label_column])
    valid_mask = (
        dataset.df[dataset.validity_column].to_numpy(dtype=np.float32) > 0.5
    ).astype(np.float32)

    return compute_balanced_class_weights(
        labels=labels,
        mask=valid_mask,
        config=class_weight_config,
    )


def _flatten_valid_predictions(
    probabilities: List[np.ndarray],
    labels: List[np.ndarray],
    masks: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten prediction/label/mask lists."""
    if not probabilities:
        return (
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
        )

    p = np.concatenate([arr.reshape(-1) for arr in probabilities], axis=0).astype(np.float32)
    y = np.concatenate([arr.reshape(-1) for arr in labels], axis=0).astype(np.int64)
    m = np.concatenate([arr.reshape(-1) for arr in masks], axis=0).astype(np.float32)

    return p, y, m


def _compute_ranking_from_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute AUPRC/AUROC from valid rows."""
    if len(y_true) == 0:
        return {"auprc": None, "auroc": None}

    keep = valid_mask > 0.5
    if keep.sum() == 0:
        return {"auprc": None, "auroc": None}

    y = y_true[keep]
    p = y_score[keep]

    try:
        ranking = compute_ranking_metrics(y_true=y, y_score=p)
        return {
            "auprc": _safe_float(getattr(ranking, "auprc", None)),
            "auroc": _safe_float(getattr(ranking, "auroc", None)),
        }
    except Exception:
        return {"auprc": None, "auroc": None}


class TorchModelTrainer:
    """General trainer for proposed model and PyTorch ablations."""

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        trainer_config: Step12TrainerConfig,
        active_seed: int,
        device: torch.device,
        class_weight_summary: ClassWeightSummary,
    ) -> None:
        self.model = model
        self.config = config
        self.trainer_config = trainer_config
        self.active_seed = int(active_seed)
        self.device = device

        self.class_weight_summary = class_weight_summary

        self.criterion = WeightedBCEWithLogitsLoss.from_project_config(
            config=config,
            class_weight_summary=class_weight_summary,
        )

        optimizer_config = build_optimizer_config(config)
        self.optimizer, self.optimizer_summary = create_optimizer(
            model=self.model,
            config=optimizer_config,
        )

        self.gradient_clipping_config = build_gradient_clipping_config(config)

        scheduler_config = build_scheduler_config(config)
        self.scheduler = SchedulerController(
            optimizer=self.optimizer,
            config=scheduler_config,
        )

        early_config = build_early_stopping_config(config)
        self.early_stopping = EarlyStopping(early_config)

        self.grad_scaler = create_grad_scaler(config, device)

        self.history: List[Dict[str, Any]] = []

        self.best_checkpoint_path = self._checkpoint_path(
            self.trainer_config.best_checkpoint_name
        )
        self.last_checkpoint_path = self._checkpoint_path(
            self.trainer_config.last_checkpoint_name
        )

        ensure_dir(self.best_checkpoint_path.parent)

    def _checkpoint_path(self, filename: str) -> Path:
        checkpoint_dir = _get_project_path(self.config, self.trainer_config.checkpoint_dir)
        ensure_dir(checkpoint_dir)
        return checkpoint_dir / filename

    def _batch_forward_loss(self, batch: Mapping[str, Any]) -> Tuple[Any, Any]:
        """Forward pass and loss computation."""
        with autocast_context(self.config, self.device):
            output = self.model(batch)
            loss_result = self.criterion(
                predictions=output.logits,
                labels=batch["y"],
                loss_mask=batch.get("loss_mask"),
                padding_mask=batch.get("padding_mask"),
            )

        return output, loss_result

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> EpochResult:
        """Train for one epoch."""
        self.model.train()

        start_time = time.perf_counter()

        loss_sum = 0.0
        unweighted_loss_sum = 0.0
        valid_count_sum = 0
        positive_count_sum = 0
        negative_count_sum = 0

        grad_norms: List[float] = []

        probabilities_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_to_device(batch, self.device)

            self.optimizer.zero_grad(set_to_none=True)

            output, loss_result = self._batch_forward_loss(batch)
            loss = loss_result.loss

            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
                self.grad_scaler.unscale_(self.optimizer)
                clip_result = clip_gradients(self.model, self.gradient_clipping_config)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                loss.backward()
                clip_result = clip_gradients(self.model, self.gradient_clipping_config)
                self.optimizer.step()

            if clip_result.total_norm_before_clip is not None:
                grad_norms.append(float(clip_result.total_norm_before_clip))

            valid_count = max(int(loss_result.valid_count), 1)
            loss_sum += float(loss_result.loss.detach().cpu().item()) * valid_count
            unweighted_loss_sum += float(loss_result.unweighted_loss.detach().cpu().item()) * valid_count
            valid_count_sum += int(loss_result.valid_count)
            positive_count_sum += int(loss_result.positive_count)
            negative_count_sum += int(loss_result.negative_count)

            probabilities_list.append(output.probabilities.detach().cpu().numpy())
            labels_list.append(batch["y"].detach().cpu().numpy())
            masks_list.append(
                (batch["loss_mask"] * batch["padding_mask"]).detach().cpu().numpy()
            )

            if (
                self.trainer_config.log_every_n_batches > 0
                and batch_index % self.trainer_config.log_every_n_batches == 0
            ):
                print(
                    f"Epoch {epoch:03d} | train batch {batch_index:04d}/{len(train_loader):04d} | "
                    f"loss={float(loss.detach().cpu().item()):.6f} | "
                    f"valid={loss_result.valid_count} | "
                    f"grad_norm={clip_result.total_norm_before_clip}"
                )

        y_score, y_true, valid_mask = _flatten_valid_predictions(
            probabilities_list,
            labels_list,
            masks_list,
        )
        ranking = _compute_ranking_from_arrays(
            y_true=y_true,
            y_score=y_score,
            valid_mask=valid_mask,
        )

        runtime = time.perf_counter() - start_time
        denom = max(valid_count_sum, 1)

        return EpochResult(
            split="train",
            epoch=int(epoch),
            loss=float(loss_sum / denom),
            unweighted_loss=float(unweighted_loss_sum / denom),
            valid_count=int(valid_count_sum),
            positive_count=int(positive_count_sum),
            negative_count=int(negative_count_sum),
            auprc=ranking["auprc"],
            auroc=ranking["auroc"],
            runtime_seconds=float(runtime),
            probability_mean=float(np.mean(y_score)) if y_score.size else 0.0,
            probability_min=float(np.min(y_score)) if y_score.size else 0.0,
            probability_max=float(np.max(y_score)) if y_score.size else 0.0,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm_mean=float(np.mean(grad_norms)) if grad_norms else None,
            grad_norm_max=float(np.max(grad_norms)) if grad_norms else None,
        )

    @torch.no_grad()
    def evaluate_one_epoch(
        self,
        val_loader: DataLoader,
        epoch: int,
        collect_predictions: bool = False,
    ) -> Tuple[EpochResult, Optional[Dict[str, np.ndarray]]]:
        """Evaluate for one epoch."""
        self.model.eval()

        start_time = time.perf_counter()

        loss_sum = 0.0
        unweighted_loss_sum = 0.0
        valid_count_sum = 0
        positive_count_sum = 0
        negative_count_sum = 0

        probabilities_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        masks_list: List[np.ndarray] = []

        segment_ids: List[str] = []
        window_ids: List[int] = []
        start_indices: List[int] = []
        end_indices: List[int] = []

        for batch in val_loader:
            batch = move_to_device(batch, self.device)

            output, loss_result = self._batch_forward_loss(batch)

            valid_count = max(int(loss_result.valid_count), 1)
            loss_sum += float(loss_result.loss.detach().cpu().item()) * valid_count
            unweighted_loss_sum += float(loss_result.unweighted_loss.detach().cpu().item()) * valid_count
            valid_count_sum += int(loss_result.valid_count)
            positive_count_sum += int(loss_result.positive_count)
            negative_count_sum += int(loss_result.negative_count)

            probabilities_list.append(output.probabilities.detach().cpu().numpy())
            labels_list.append(batch["y"].detach().cpu().numpy())
            masks_list.append(
                (batch["loss_mask"] * batch["padding_mask"]).detach().cpu().numpy()
            )

            if collect_predictions:
                segment_ids.extend([str(item) for item in batch["segment_id"]])
                window_ids.extend(batch["window_id"].detach().cpu().numpy().astype(int).tolist())
                start_indices.extend(batch["start_index"].detach().cpu().numpy().astype(int).tolist())
                end_indices.extend(batch["end_index"].detach().cpu().numpy().astype(int).tolist())

        y_score, y_true, valid_mask = _flatten_valid_predictions(
            probabilities_list,
            labels_list,
            masks_list,
        )
        ranking = _compute_ranking_from_arrays(
            y_true=y_true,
            y_score=y_score,
            valid_mask=valid_mask,
        )

        runtime = time.perf_counter() - start_time
        denom = max(valid_count_sum, 1)

        result = EpochResult(
            split="val",
            epoch=int(epoch),
            loss=float(loss_sum / denom),
            unweighted_loss=float(unweighted_loss_sum / denom),
            valid_count=int(valid_count_sum),
            positive_count=int(positive_count_sum),
            negative_count=int(negative_count_sum),
            auprc=ranking["auprc"],
            auroc=ranking["auroc"],
            runtime_seconds=float(runtime),
            probability_mean=float(np.mean(y_score)) if y_score.size else 0.0,
            probability_min=float(np.min(y_score)) if y_score.size else 0.0,
            probability_max=float(np.max(y_score)) if y_score.size else 0.0,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            grad_norm_mean=None,
            grad_norm_max=None,
        )

        prediction_payload = None

        if collect_predictions:
            prediction_payload = {
                "y_score_flat": y_score,
                "y_true_flat": y_true,
                "valid_mask_flat": valid_mask,
                "probabilities_windows": np.concatenate(probabilities_list, axis=0)
                if probabilities_list
                else np.empty((0, 0), dtype=np.float32),
                "labels_windows": np.concatenate(labels_list, axis=0)
                if labels_list
                else np.empty((0, 0), dtype=np.int64),
                "valid_mask_windows": np.concatenate(masks_list, axis=0)
                if masks_list
                else np.empty((0, 0), dtype=np.float32),
                "segment_id": np.asarray(segment_ids, dtype=object),
                "window_id": np.asarray(window_ids, dtype=np.int64),
                "start_index": np.asarray(start_indices, dtype=np.int64),
                "end_index": np.asarray(end_indices, dtype=np.int64),
            }

        return result, prediction_payload

    def _metrics_for_monitor(self, train_result: EpochResult, val_result: EpochResult) -> Dict[str, Any]:
        """Build monitor metric dict."""
        return {
            "train_loss": train_result.loss,
            "val_loss": val_result.loss,
            "train_unweighted_loss": train_result.unweighted_loss,
            "val_unweighted_loss": val_result.unweighted_loss,
            "train_auprc": train_result.auprc,
            "val_auprc": val_result.auprc,
            "train_auroc": train_result.auroc,
            "val_auroc": val_result.auroc,
        }

    def _checkpoint_payload(
        self,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult,
    ) -> Dict[str, Any]:
        """Create checkpoint payload."""
        return {
            "epoch": int(epoch),
            "active_seed": int(self.active_seed),
            "model_name": self.trainer_config.model_name,
            "variant_name": self.trainer_config.variant_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "early_stopping_state_dict": self.early_stopping.state_dict(),
            "class_weights": self.class_weight_summary.to_dict(),
            "optimizer_summary": self.optimizer_summary.to_dict(),
            "trainer_config": self.trainer_config.to_dict(),
            "train_result": train_result.to_dict(),
            "val_result": val_result.to_dict(),
            "history": list(self.history),
            "model_module_summary": self.model.module_summary()
            if hasattr(self.model, "module_summary")
            else str(type(self.model)),
        }

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult,
    ) -> None:
        """Save checkpoint."""
        ensure_dir(path.parent)
        payload = self._checkpoint_payload(
            epoch=epoch,
            train_result=train_result,
            val_result=val_result,
        )
        torch.save(payload, path)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_dataset: XiWindowDataset,
        val_dataset: XiWindowDataset,
    ) -> TrainingRunSummary:
        """Run full training."""
        print("=" * 100)
        print("STEP 12 TRAINING START")
        print("=" * 100)
        print(f"Model name        : {self.trainer_config.model_name}")
        print(f"Variant           : {self.trainer_config.variant_name}")
        print(f"Seed              : {self.active_seed}")
        print(f"Device            : {self.device}")
        print(f"Max epochs        : {self.trainer_config.max_epochs}")
        print(f"Train windows     : {len(train_dataset)}")
        print(f"Val windows       : {len(val_dataset)}")
        print(f"Class weights     : alpha_0={self.class_weight_summary.alpha_0:.6f}, "
              f"alpha_1={self.class_weight_summary.alpha_1:.6f}")
        print(f"Best checkpoint   : {self.best_checkpoint_path}")
        print("=" * 100)

        reset_cuda_peak_memory_stats(self.device)

        best_epoch: Optional[int] = None
        stopped_early = False

        for epoch in range(1, int(self.trainer_config.max_epochs) + 1):
            train_result = self.train_one_epoch(train_loader, epoch=epoch)
            val_result, _ = self.evaluate_one_epoch(
                val_loader,
                epoch=epoch,
                collect_predictions=False,
            )

            monitor_metrics = self._metrics_for_monitor(train_result, val_result)
            early_result = self.early_stopping.step(
                epoch=epoch,
                metrics=monitor_metrics,
            )

            scheduler_result = self.scheduler.step(
                epoch=epoch,
                metrics=monitor_metrics,
            )

            epoch_payload = {
                "epoch": int(epoch),
                "train": train_result.to_dict(),
                "val": val_result.to_dict(),
                "early_stopping": early_result.to_dict(),
                "scheduler": scheduler_result.to_dict(),
                "cuda_memory": get_cuda_memory_stats(self.device),
            }
            self.history.append(epoch_payload)

            if early_result.improved:
                best_epoch = int(epoch)

                if self.trainer_config.save_best:
                    self.save_checkpoint(
                        path=self.best_checkpoint_path,
                        epoch=epoch,
                        train_result=train_result,
                        val_result=val_result,
                    )

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_result.loss:.6f} | "
                f"val_loss={val_result.loss:.6f} | "
                f"val_auprc={val_result.auprc} | "
                f"val_auroc={val_result.auroc} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.8f} | "
                f"best={early_result.best_value} @ epoch {early_result.best_epoch} | "
                f"bad_epochs={early_result.bad_epochs}"
            )

            if early_result.should_stop:
                stopped_early = True
                print(f"Early stopping triggered: {early_result.stop_reason}")
                break

        epochs_completed = len(self.history)

        if self.trainer_config.save_last and self.history:
            last_train = EpochResult(**self.history[-1]["train"])
            last_val = EpochResult(**self.history[-1]["val"])
            self.save_checkpoint(
                path=self.last_checkpoint_path,
                epoch=epochs_completed,
                train_result=last_train,
                val_result=last_val,
            )

        if self.trainer_config.save_validation_predictions:
            self.save_validation_predictions(val_loader)

        self.save_history()

        final_status = "PASSED" if self.best_checkpoint_path.exists() else "FAILED"

        summary = TrainingRunSummary(
            model_name=self.trainer_config.model_name,
            variant_name=self.trainer_config.variant_name,
            active_seed=int(self.active_seed),
            final_status=final_status,
            best_epoch=self.early_stopping.best_epoch,
            best_monitor=self.early_stopping.monitor,
            best_monitor_value=self.early_stopping.best_value,
            epochs_completed=int(epochs_completed),
            stopped_early=bool(stopped_early),
            stop_reason=self.early_stopping.state.stop_reason,
            best_checkpoint_path=str(self.best_checkpoint_path) if self.best_checkpoint_path.exists() else None,
            last_checkpoint_path=str(self.last_checkpoint_path) if self.last_checkpoint_path.exists() else None,
            class_weights=self.class_weight_summary.to_dict(),
            optimizer=optimizer_state_summary(self.optimizer, self.optimizer_summary),
            scheduler=self.scheduler.summary(),
            early_stopping=self.early_stopping.best_summary(),
            trainer_config=self.trainer_config.to_dict(),
            train_dataset_summary=train_dataset.summary(),
            val_dataset_summary=val_dataset.summary(),
            final_cuda_memory=get_cuda_memory_stats(self.device),
        )

        self.save_summary(summary)

        print("=" * 100)
        print("STEP 12 TRAINING SUMMARY")
        print("=" * 100)
        print(f"Final status       : {summary.final_status}")
        print(f"Best epoch         : {summary.best_epoch}")
        print(f"Best {summary.best_monitor:<12}: {summary.best_monitor_value}")
        print(f"Epochs completed   : {summary.epochs_completed}")
        print(f"Stopped early      : {summary.stopped_early}")
        print(f"Best checkpoint    : {summary.best_checkpoint_path}")
        print(f"Last checkpoint    : {summary.last_checkpoint_path}")
        print(f"History CSV        : {_get_project_path(self.config, self.trainer_config.history_csv)}")
        print(f"Summary JSON       : {_get_project_path(self.config, self.trainer_config.summary_json)}")
        print("=" * 100)

        if final_status != "PASSED":
            raise RuntimeError("Step 12 training failed: best checkpoint was not saved.")

        clear_cuda_cache()
        return summary

    @torch.no_grad()
    def save_validation_predictions(self, val_loader: DataLoader) -> None:
        """Save validation probabilities for later threshold selection."""
        if self.best_checkpoint_path.exists():
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        _val_result, payload = self.evaluate_one_epoch(
            val_loader,
            epoch=int(self.early_stopping.best_epoch or 0),
            collect_predictions=True,
        )

        if payload is None:
            return

        output_path = _get_project_path(
            self.config,
            self.trainer_config.validation_predictions_npz,
        )
        ensure_dir(output_path.parent)

        np.savez_compressed(output_path, **payload)

    def save_history(self) -> None:
        """Save training history JSON and CSV."""
        json_path = _get_project_path(self.config, self.trainer_config.history_json)
        csv_path = _get_project_path(self.config, self.trainer_config.history_csv)

        ensure_dir(json_path.parent)
        ensure_dir(csv_path.parent)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

        flat_rows = []

        for item in self.history:
            row = {
                "epoch": item["epoch"],
                "train_loss": item["train"]["loss"],
                "val_loss": item["val"]["loss"],
                "train_unweighted_loss": item["train"]["unweighted_loss"],
                "val_unweighted_loss": item["val"]["unweighted_loss"],
                "train_auprc": item["train"]["auprc"],
                "val_auprc": item["val"]["auprc"],
                "train_auroc": item["train"]["auroc"],
                "val_auroc": item["val"]["auroc"],
                "learning_rate": item["train"]["learning_rate"],
                "grad_norm_mean": item["train"]["grad_norm_mean"],
                "grad_norm_max": item["train"]["grad_norm_max"],
                "best_value": item["early_stopping"]["best_value"],
                "best_epoch": item["early_stopping"]["best_epoch"],
                "bad_epochs": item["early_stopping"]["bad_epochs"],
            }
            flat_rows.append(row)

        pd.DataFrame(flat_rows).to_csv(csv_path, index=False)

    def save_summary(self, summary: TrainingRunSummary) -> None:
        """Save final summary JSON."""
        summary_path = _get_project_path(self.config, self.trainer_config.summary_json)
        save_json(summary.to_dict(), summary_path, indent=2)


def run_step12_training_protocol(
    config: Mapping[str, Any],
    active_seed: int,
) -> TrainingRunSummary:
    """Build everything and run Step-12 training."""
    device_info = setup_device_from_config(config, verbose=True)
    device = device_info.device

    trainer_config = build_step12_trainer_config(config)

    train_loader, val_loader, train_dataset, val_dataset = build_step12_dataloaders(
        config=config,
        trainer_config=trainer_config,
        active_seed=active_seed,
    )

    class_weight_config = build_class_weight_config(config)
    class_weight_summary = compute_class_weights_from_dataset(
        dataset=train_dataset,
        class_weight_config=class_weight_config,
    )

    model, build_info, _variant_config = build_model(
        config=config,
        variant_name=trainer_config.variant_name,
        device=device,
    )

    print("=" * 100)
    print("STEP 12 MODEL BUILD SUMMARY")
    print("=" * 100)
    print(f"Variant               : {build_info.variant_name}")
    print(f"Variant group         : {build_info.variant_group}")
    print(f"Trainable parameters  : {build_info.trainable_parameters}")
    print(f"Input dim             : {build_info.input_dim}")
    print(f"Temporal block        : {build_info.temporal_block}")
    print(f"use_residual_evolution: {build_info.use_residual_evolution}")
    print(f"use_weak_accumulation : {build_info.use_weak_accumulation}")
    print(f"use_kirchhoff_exchange: {build_info.use_kirchhoff_exchange}")
    print(f"use_third_order       : {build_info.use_third_order}")
    print("=" * 100)

    trainer = TorchModelTrainer(
        model=model,
        config=config,
        trainer_config=trainer_config,
        active_seed=active_seed,
        device=device,
        class_weight_summary=class_weight_summary,
    )

    return trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

def proposed_best_checkpoint_path(config: Mapping[str, Any]) -> Path:
    """
    Resolve the full proposed-model best checkpoint path.

    Step 13 uses this to decide whether it can reuse an existing Step-12
    checkpoint or whether it must train the full model first.
    """
    models_dir = str(get_by_path(config, "paths.models_dir", "results/models"))
    best_name = str(
        get_by_path(
            config,
            "training.checkpointing.best_checkpoint_name",
            "proposed_best.pt",
        )
    )

    return _get_project_path(config, str(Path(models_dir) / best_name))


def proposed_best_checkpoint_exists(config: Mapping[str, Any]) -> bool:
    """
    Return True if the full proposed-model best checkpoint exists.

    This is only a convenience helper for Step 13 orchestration.
    It does not change Step 12 training behavior.
    """
    return proposed_best_checkpoint_path(config).exists()

__all__ = [
    "DEFAULT_STEP12_FEATURE_COLUMNS",
    "SequenceWindowSpec",
    "Step12TrainerConfig",
    "EpochResult",
    "TrainingRunSummary",
    "XiWindowDataset",
    "collate_xi_windows",
    "build_step12_trainer_config",
    "build_step12_datasets",
    "build_step12_dataloaders",
    "compute_class_weights_from_dataset",
    "TorchModelTrainer",
    "run_step12_training_protocol",
    "proposed_best_checkpoint_path",
    "proposed_best_checkpoint_exists",
]
