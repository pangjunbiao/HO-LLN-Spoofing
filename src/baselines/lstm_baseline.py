"""
LSTM-xi baseline for causal GNSS spoofing detection.

Step 15 purpose:
- Train a standard sequence baseline on the same reconstructed xi_t features.
- Use Dataset-1 train only for fitting.
- Use Dataset-1 validation only for early stopping and threshold selection later.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online later.
- Do not use raw shortcut columns.
- Reset hidden state at independent trajectory/window start.

Important:
- This is a fair standard LSTM baseline, not an artificially weakened baseline.
- It uses the same xi_t input columns as the proposed model.
- It does not use Kirchhoff exchange, third-order fusion, weak-accumulation modules,
  or liquid second-order dynamics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from src.baselines.xgboost_baseline import (
    BaselineSplitData,
    compute_scale_pos_weight,
    get_baseline_delta_t_column,
    get_baseline_feature_columns,
    get_baseline_label_column,
    get_baseline_order_column,
    get_baseline_segment_column,
    get_baseline_split_csv_path,
    get_baseline_validity_column,
    labels_to_binary,
)
from src.evaluation.evaluate_dataset1 import EvaluationPredictionBundle
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed


@dataclass
class SequenceWindowSpec:
    """Windowing settings for sequence baselines."""

    train_window_length: int = 256
    train_stride: int = 128
    eval_window_length: int = 512
    eval_stride: int = 512
    online_full_sequence: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceWindowSummary:
    """Dataset/window summary."""

    split_name: str
    csv_path: str
    rows: int
    valid_rows: int
    invalid_rows: int
    attack_valid_rows: int
    normal_valid_rows: int
    segments: int
    windows: int
    window_length: int
    stride: int
    feature_dim: int
    full_sequence: bool
    feature_columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LSTMBaselineConfig:
    """Configuration for LSTM-xi baseline."""

    model_name: str = "LSTM-xi"

    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.10
    bidirectional: bool = False

    batch_size: int = 16
    eval_batch_size: int = 16
    max_epochs: int = 80
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 1.0

    patience: int = 12
    min_delta: float = 1.0e-5

    use_train_class_weight: bool = True
    num_workers: int = 0

    output_model_path: str = "results/models/lstm_xi.pt"
    output_history_csv: str = "results/tables/lstm_xi_training_history.csv"
    output_summary_json: str = "results/tables/lstm_xi_training_summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SequenceEpochResult:
    """One sequence-baseline epoch result."""

    epoch: int
    train_loss: float
    val_loss: float
    val_auprc: Optional[float]
    val_accuracy_05: float
    learning_rate: float
    runtime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LSTMBaselineArtifact:
    """Trained LSTM baseline artifact."""

    model_name: str
    model: nn.Module
    config: LSTMBaselineConfig
    model_path: Path
    feature_columns: List[str]
    window_spec: SequenceWindowSpec

    best_epoch: int
    best_val_loss: float
    train_summary: Dict[str, Any]
    val_summary: Dict[str, Any]
    history: List[SequenceEpochResult]
    fit_runtime_seconds: float
    active_seed: int
    device: str

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "config": self.config.to_dict(),
            "feature_columns": list(self.feature_columns),
            "window_spec": self.window_spec.to_dict(),
            "best_epoch": int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss),
            "train_summary": self.train_summary,
            "val_summary": self.val_summary,
            "history": [item.to_dict() for item in self.history],
            "fit_runtime_seconds": float(self.fit_runtime_seconds),
            "active_seed": int(self.active_seed),
            "device": str(self.device),
        }


class SequenceXiWindowDataset(Dataset):
    """
    Sequence-window dataset for LSTM/GRU/TCN baselines.

    Each item returns:
    - x: [T,F]
    - y: [T]
    - loss_mask: [T] from xi_nu
    - padding_mask: [T]
    - row_indices: [T]
    - delta_t: [T]
    - segment_id
    - real_length
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        split_name: str,
        feature_columns: Sequence[str],
        window_length: int,
        stride: int,
        full_sequence: bool = False,
    ) -> None:
        super().__init__()

        self.config = config
        self.split_name = str(split_name)
        self.feature_columns = list(feature_columns)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.full_sequence = bool(full_sequence)

        self.label_column = get_baseline_label_column(config)
        self.validity_column = get_baseline_validity_column(config)
        self.segment_column = get_baseline_segment_column(config)
        self.order_column = get_baseline_order_column(config)
        self.delta_t_column = get_baseline_delta_t_column(config)
        self.csv_path = get_baseline_split_csv_path(config, split_name)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Sequence baseline split file not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path, low_memory=False)
        df["_sequence_original_index"] = np.arange(len(df), dtype=np.int64)

        if self.segment_column in df.columns and self.order_column in df.columns:
            df = df.sort_values(
                [self.segment_column, self.order_column],
                kind="mergesort",
            ).reset_index(drop=True)
        elif self.segment_column in df.columns:
            df = df.sort_values(
                [self.segment_column, "_sequence_original_index"],
                kind="mergesort",
            ).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        missing = [column for column in self.feature_columns if column not in df.columns]
        if missing:
            raise KeyError(f"Missing sequence baseline feature columns: {missing}")

        if self.label_column not in df.columns:
            raise KeyError(f"Missing label column '{self.label_column}' in {self.csv_path}")

        self.df = df

        x = df[self.feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.x_all = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        self.y_all = labels_to_binary(df[self.label_column].to_numpy()).astype(np.int64)

        if self.validity_column in df.columns:
            valid = pd.to_numeric(df[self.validity_column], errors="coerce").fillna(0.0)
            self.valid_mask_all = (valid.to_numpy(dtype=np.float32) > 0.5).astype(np.float32)
        else:
            self.valid_mask_all = np.ones(len(df), dtype=np.float32)

        if self.segment_column in df.columns:
            self.segment_ids_all = df[self.segment_column].astype(str).to_numpy(dtype=object)
        else:
            self.segment_ids_all = np.asarray([f"{split_name}_segment_0"] * len(df), dtype=object)

        if self.delta_t_column in df.columns:
            self.delta_t_all = pd.to_numeric(
                df[self.delta_t_column],
                errors="coerce",
            ).fillna(1.0).to_numpy(dtype=np.float32)
        else:
            self.delta_t_all = np.ones(len(df), dtype=np.float32)

        self.row_indices_all = np.arange(len(df), dtype=np.int64)
        self.windows: List[np.ndarray] = self._build_windows()

        if self.full_sequence and self.windows:
            self.window_length = max(len(indices) for indices in self.windows)

    def _build_windows(self) -> List[np.ndarray]:
        """Build window row-index arrays without crossing segment boundaries."""
        windows: List[np.ndarray] = []

        if len(self.df) == 0:
            return windows

        grouped = pd.DataFrame(
            {
                "segment_id": self.segment_ids_all,
                "row_index": self.row_indices_all,
            }
        ).groupby("segment_id", sort=False)

        for _segment_id, group in grouped:
            indices = group["row_index"].to_numpy(dtype=np.int64)
            n = len(indices)

            if n == 0:
                continue

            if self.full_sequence:
                windows.append(indices)
                continue

            if n <= self.window_length:
                windows.append(indices)
                continue

            starts = list(range(0, n - self.window_length + 1, max(self.stride, 1)))
            last_start = n - self.window_length

            if starts[-1] != last_start:
                starts.append(last_start)

            for start in starts:
                windows.append(indices[start : start + self.window_length])

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row_indices = self.windows[int(index)]
        real_length = int(len(row_indices))
        pad_length = int(self.window_length - real_length)

        x = np.zeros((self.window_length, len(self.feature_columns)), dtype=np.float32)
        y = np.zeros((self.window_length,), dtype=np.float32)
        loss_mask = np.zeros((self.window_length,), dtype=np.float32)
        padding_mask = np.zeros((self.window_length,), dtype=np.float32)
        delta_t = np.ones((self.window_length,), dtype=np.float32)
        padded_row_indices = -np.ones((self.window_length,), dtype=np.int64)

        x[:real_length] = self.x_all[row_indices]
        y[:real_length] = self.y_all[row_indices].astype(np.float32)
        loss_mask[:real_length] = self.valid_mask_all[row_indices]
        padding_mask[:real_length] = 1.0
        delta_t[:real_length] = self.delta_t_all[row_indices]
        padded_row_indices[:real_length] = row_indices

        segment_id = str(self.segment_ids_all[row_indices[0]]) if real_length > 0 else "unknown"

        return {
            "x": x,
            "y": y,
            "loss_mask": loss_mask,
            "padding_mask": padding_mask,
            "delta_t": delta_t,
            "row_indices": padded_row_indices,
            "segment_id": segment_id,
            "real_length": real_length,
        }

    def summary(self) -> Dict[str, Any]:
        valid = self.valid_mask_all > 0.5

        summary = SequenceWindowSummary(
            split_name=self.split_name,
            csv_path=str(self.csv_path),
            rows=int(len(self.df)),
            valid_rows=int(valid.sum()),
            invalid_rows=int(len(valid) - valid.sum()),
            attack_valid_rows=int(((self.y_all == 1) & valid).sum()),
            normal_valid_rows=int(((self.y_all == 0) & valid).sum()),
            segments=int(pd.Series(self.segment_ids_all).nunique()),
            windows=int(len(self.windows)),
            window_length=int(self.window_length),
            stride=int(self.stride),
            feature_dim=int(len(self.feature_columns)),
            full_sequence=bool(self.full_sequence),
            feature_columns=list(self.feature_columns),
        )

        return summary.to_dict()


def collate_sequence_windows(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate sequence-window dicts."""
    return {
        "x": torch.tensor(np.stack([item["x"] for item in items]), dtype=torch.float32),
        "y": torch.tensor(np.stack([item["y"] for item in items]), dtype=torch.float32),
        "loss_mask": torch.tensor(
            np.stack([item["loss_mask"] for item in items]),
            dtype=torch.float32,
        ),
        "padding_mask": torch.tensor(
            np.stack([item["padding_mask"] for item in items]),
            dtype=torch.float32,
        ),
        "delta_t": torch.tensor(
            np.stack([item["delta_t"] for item in items]),
            dtype=torch.float32,
        ),
        "row_indices": torch.tensor(
            np.stack([item["row_indices"] for item in items]),
            dtype=torch.long,
        ),
        "segment_id": [str(item["segment_id"]) for item in items],
        "real_length": torch.tensor(
            [int(item["real_length"]) for item in items],
            dtype=torch.long,
        ),
    }


class LSTMXiClassifier(nn.Module):
    """Standard LSTM time-step classifier."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.10,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bool(bidirectional),
        )

        output_dim = int(hidden_dim) * (2 if bool(bidirectional) else 1)

        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(output_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits with shape [B,T]."""
        sequence_output, _hidden = self.lstm(x)
        sequence_output = self.dropout(sequence_output)
        logits = self.output(sequence_output).squeeze(-1)
        return logits


def _project_path(config: Mapping[str, Any], value: str | Path) -> Path:
    return resolve_project_path(config, str(value))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None

    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass

    try:
        value = float(value)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def save_json_safe(payload: Mapping[str, Any], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(dict(payload)), file, indent=2)

    return output_path


def build_sequence_window_spec(config: Mapping[str, Any]) -> SequenceWindowSpec:
    """Build shared sequence-window spec."""
    return SequenceWindowSpec(
        train_window_length=int(
            get_by_path(
                config,
                "baselines.sequence.train_window_length",
                get_by_path(
                    config,
                    "training.dataset.sequence.window_length_train",
                    get_by_path(config, "training.dataset.train_window_size", 256),
                ),
            )
        ),
        train_stride=int(
            get_by_path(
                config,
                "baselines.sequence.train_stride",
                get_by_path(
                    config,
                    "training.dataset.sequence.stride_train",
                    get_by_path(config, "training.dataset.train_stride", 128),
                ),
            )
        ),
        eval_window_length=int(
            get_by_path(
                config,
                "baselines.sequence.eval_window_length",
                get_by_path(
                    config,
                    "training.dataset.sequence.window_length_eval",
                    get_by_path(config, "training.dataset.eval_window_size", 512),
                ),
            )
        ),
        eval_stride=int(
            get_by_path(
                config,
                "baselines.sequence.eval_stride",
                get_by_path(
                    config,
                    "training.dataset.sequence.stride_eval",
                    get_by_path(config, "training.dataset.eval_stride", 512),
                ),
            )
        ),
        online_full_sequence=bool(
            get_by_path(config, "baselines.sequence.online_full_sequence", True)
        ),
    )


def build_lstm_baseline_config(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> LSTMBaselineConfig:
    """Build LSTM baseline config."""
    base = "baselines.lstm_xi"

    return LSTMBaselineConfig(
        model_name=str(get_by_path(config, f"{base}.model_name", "LSTM-xi")),
        hidden_dim=int(get_by_path(config, f"{base}.hidden_dim", 64)),
        num_layers=int(get_by_path(config, f"{base}.num_layers", 1)),
        dropout=float(get_by_path(config, f"{base}.dropout", 0.10)),
        bidirectional=bool(get_by_path(config, f"{base}.bidirectional", False)),
        batch_size=int(get_by_path(config, f"{base}.batch_size", 16)),
        eval_batch_size=int(get_by_path(config, f"{base}.eval_batch_size", 16)),
        max_epochs=int(get_by_path(config, f"{base}.max_epochs", 80)),
        learning_rate=float(get_by_path(config, f"{base}.learning_rate", 1.0e-3)),
        weight_decay=float(get_by_path(config, f"{base}.weight_decay", 1.0e-4)),
        gradient_clip_norm=float(get_by_path(config, f"{base}.gradient_clip_norm", 1.0)),
        patience=int(get_by_path(config, f"{base}.patience", 12)),
        min_delta=float(get_by_path(config, f"{base}.min_delta", 1.0e-5)),
        use_train_class_weight=bool(get_by_path(config, f"{base}.use_train_class_weight", True)),
        num_workers=int(get_by_path(config, f"{base}.num_workers", 0)),
        output_model_path=str(
            get_by_path(config, f"{base}.output_model_path", "results/models/lstm_xi.pt")
        ),
        output_history_csv=str(
            get_by_path(
                config,
                f"{base}.output_history_csv",
                "results/tables/lstm_xi_training_history.csv",
            )
        ),
        output_summary_json=str(
            get_by_path(
                config,
                f"{base}.output_summary_json",
                "results/tables/lstm_xi_training_summary.json",
            )
        ),
    )


def build_sequence_dataset(
    config: Mapping[str, Any],
    split_name: str,
    feature_columns: Sequence[str],
    window_spec: SequenceWindowSpec,
    train: bool = False,
) -> SequenceXiWindowDataset:
    """Build sequence dataset for one split."""
    split_name = str(split_name)
    full_sequence = bool(split_name == "online" and window_spec.online_full_sequence)

    if train:
        window_length = window_spec.train_window_length
        stride = window_spec.train_stride
    elif full_sequence:
        window_length = window_spec.eval_window_length
        stride = window_spec.eval_stride
    else:
        window_length = window_spec.eval_window_length
        stride = window_spec.eval_stride

    return SequenceXiWindowDataset(
        config=config,
        split_name=split_name,
        feature_columns=feature_columns,
        window_length=window_length,
        stride=stride,
        full_sequence=full_sequence,
    )


def build_sequence_dataloader(
    dataset: SequenceXiWindowDataset,
    batch_size: int,
    shuffle: bool,
    active_seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build sequence DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(int(active_seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        generator=generator,
        collate_fn=collate_sequence_windows,
        drop_last=False,
    )


def compute_binary_auprc(y_true: np.ndarray, probabilities: np.ndarray) -> Optional[float]:
    """Compute AUPRC if sklearn is available and both classes exist."""
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return None

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities).astype(float)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None

    return float(average_precision_score(y_true, probabilities))


def compute_accuracy(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    """Compute simple 0.5 validation accuracy for monitoring only."""
    y_true = np.asarray(y_true).astype(int)

    if len(y_true) == 0:
        return 0.0

    predictions = (np.asarray(probabilities) >= float(threshold)).astype(int)
    return float((predictions == y_true).mean())


def masked_bce_with_logits(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    pos_weight: Optional[Tensor],
) -> Tensor:
    """Masked BCEWithLogits loss."""
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )

    mask = mask.float()
    denominator = torch.clamp(mask.sum(), min=1.0)
    return (loss * mask).sum() / denominator


def move_sequence_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Move tensor batch fields to device."""
    moved: Dict[str, Any] = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value

    return moved


def _make_pos_weight_from_dataset(
    dataset: SequenceXiWindowDataset,
    use_train_class_weight: bool,
    device: torch.device,
) -> Optional[Tensor]:
    """Make train-only positive class weight."""
    if not use_train_class_weight:
        return None

    y_valid = dataset.y_all[dataset.valid_mask_all > 0.5]
    scale_pos_weight = compute_scale_pos_weight(y_valid)

    return torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)


def _train_one_epoch_sequence(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[Tensor],
    gradient_clip_norm: float,
) -> float:
    """Train one epoch for sequence model."""
    model.train()
    losses: List[float] = []

    for batch in loader:
        batch = move_sequence_batch_to_device(batch, device)

        x = batch["x"]
        y = batch["y"]
        mask = batch["loss_mask"] * batch["padding_mask"]

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = masked_bce_with_logits(
            logits=logits,
            labels=y,
            mask=mask,
            pos_weight=pos_weight,
        )

        loss.backward()

        if gradient_clip_norm is not None and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(gradient_clip_norm),
            )

        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    if not losses:
        return float("nan")

    return float(np.mean(losses))


@torch.no_grad()
def _evaluate_sequence_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: Optional[Tensor],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate sequence model and return val loss + valid probabilities/labels."""
    model.eval()

    losses: List[float] = []
    probabilities: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for batch in loader:
        batch = move_sequence_batch_to_device(batch, device)

        x = batch["x"]
        y = batch["y"]
        mask = batch["loss_mask"] * batch["padding_mask"]

        logits = model(x)
        loss = masked_bce_with_logits(
            logits=logits,
            labels=y,
            mask=mask,
            pos_weight=pos_weight,
        )

        probs = torch.sigmoid(logits)

        valid = mask.detach().cpu().numpy().reshape(-1) > 0.5

        probabilities.append(probs.detach().cpu().numpy().reshape(-1)[valid])
        labels.append(y.detach().cpu().numpy().reshape(-1)[valid])
        losses.append(float(loss.detach().cpu().item()))

    if probabilities:
        p = np.concatenate(probabilities).astype(np.float32)
        y_np = np.concatenate(labels).astype(np.int64)
    else:
        p = np.asarray([], dtype=np.float32)
        y_np = np.asarray([], dtype=np.int64)

    if not losses:
        return float("nan"), p, y_np

    return float(np.mean(losses)), p, y_np


def save_lstm_checkpoint(
    model: nn.Module,
    model_path: Path | str,
    cfg: LSTMBaselineConfig,
    window_spec: SequenceWindowSpec,
    feature_columns: Sequence[str],
    epoch: int,
    val_loss: float,
    active_seed: int,
) -> Path:
    """Save LSTM baseline checkpoint."""
    model_path = Path(model_path)
    ensure_dir(model_path.parent)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "window_spec": window_spec.to_dict(),
            "feature_columns": list(feature_columns),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "active_seed": int(active_seed),
            "model_name": cfg.model_name,
        },
        model_path,
    )

    return model_path


def _load_torch_checkpoint(path: Path | str, device: torch.device) -> Dict[str, Any]:
    """Load torch checkpoint with compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_lstm_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> LSTMBaselineArtifact:
    """Train LSTM-xi baseline."""
    start_time = time.perf_counter()

    set_global_seed(int(active_seed))

    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_lstm_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)
    window_spec = build_sequence_window_spec(config)

    train_dataset = build_sequence_dataset(
        config=config,
        split_name="train",
        feature_columns=feature_columns,
        window_spec=window_spec,
        train=True,
    )
    val_dataset = build_sequence_dataset(
        config=config,
        split_name="val",
        feature_columns=feature_columns,
        window_spec=window_spec,
        train=False,
    )

    train_loader = build_sequence_dataloader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        active_seed=active_seed,
        num_workers=cfg.num_workers,
    )
    val_loader = build_sequence_dataloader(
        dataset=val_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        active_seed=active_seed,
        num_workers=cfg.num_workers,
    )

    model = LSTMXiClassifier(
        input_dim=len(feature_columns),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    pos_weight = _make_pos_weight_from_dataset(
        dataset=train_dataset,
        use_train_class_weight=cfg.use_train_class_weight,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    model_path = _project_path(config, cfg.output_model_path)
    history_csv = _project_path(config, cfg.output_history_csv)
    summary_json = _project_path(config, cfg.output_summary_json)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[SequenceEpochResult] = []

    print("=" * 100)
    print("STEP 15 BASELINE TRAINING: LSTM-xi")
    print("=" * 100)
    print(f"Device              : {device}")
    print(f"Train rows/windows  : {train_dataset.summary()['rows']} / {train_dataset.summary()['windows']}")
    print(f"Val rows/windows    : {val_dataset.summary()['rows']} / {val_dataset.summary()['windows']}")
    print(f"Feature dim         : {len(feature_columns)}")
    print(f"Hidden dim/layers   : {cfg.hidden_dim} / {cfg.num_layers}")
    print(f"Bidirectional       : {cfg.bidirectional}")
    print(f"Dropout             : {cfg.dropout}")
    print(f"Max epochs/patience : {cfg.max_epochs} / {cfg.patience}")
    print("Uses same xi_t features only. No raw shortcut columns.")
    print("=" * 100)

    for epoch in range(1, int(cfg.max_epochs) + 1):
        epoch_start = time.perf_counter()

        train_loss = _train_one_epoch_sequence(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )

        val_loss, val_probs, val_labels = _evaluate_sequence_model(
            model=model,
            loader=val_loader,
            device=device,
            pos_weight=pos_weight,
        )

        val_auprc = compute_binary_auprc(val_labels, val_probs)
        val_accuracy = compute_accuracy(val_labels, val_probs, threshold=0.5)

        epoch_result = SequenceEpochResult(
            epoch=int(epoch),
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            val_auprc=_safe_float(val_auprc),
            val_accuracy_05=float(val_accuracy),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            runtime_seconds=float(time.perf_counter() - epoch_start),
        )
        history.append(epoch_result)

        improved = val_loss < (best_val_loss - float(cfg.min_delta))

        if improved:
            best_val_loss = float(val_loss)
            best_epoch = int(epoch)
            epochs_without_improvement = 0

            save_lstm_checkpoint(
                model=model,
                model_path=model_path,
                cfg=cfg,
                window_spec=window_spec,
                feature_columns=feature_columns,
                epoch=epoch,
                val_loss=val_loss,
                active_seed=active_seed,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_auprc={val_auprc} | "
            f"val_acc={val_accuracy:.6f} | "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= int(cfg.patience):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if not model_path.exists():
        save_lstm_checkpoint(
            model=model,
            model_path=model_path,
            cfg=cfg,
            window_spec=window_spec,
            feature_columns=feature_columns,
            epoch=int(history[-1].epoch if history else 0),
            val_loss=float(history[-1].val_loss if history else float("nan")),
            active_seed=active_seed,
        )

    checkpoint = _load_torch_checkpoint(model_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ensure_dir(history_csv.parent)
    pd.DataFrame([item.to_dict() for item in history]).to_csv(history_csv, index=False)

    fit_runtime = float(time.perf_counter() - start_time)

    artifact = LSTMBaselineArtifact(
        model_name=cfg.model_name,
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        window_spec=window_spec,
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
        train_summary=train_dataset.summary(),
        val_summary=val_dataset.summary(),
        history=history,
        fit_runtime_seconds=fit_runtime,
        active_seed=int(active_seed),
        device=str(device),
    )

    save_json_safe(artifact.summary(), summary_json)

    print("LSTM-xi training completed.")
    print(f"Model path          : {model_path}")
    print(f"History CSV         : {history_csv}")
    print(f"Summary JSON        : {summary_json}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best val loss       : {best_val_loss}")
    print(f"Runtime seconds     : {fit_runtime:.3f}")
    print("=" * 100)

    return artifact


def load_lstm_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> LSTMBaselineArtifact:
    """Load saved LSTM baseline."""
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_lstm_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)
    window_spec = build_sequence_window_spec(config)
    model_path = _project_path(config, cfg.output_model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved LSTM baseline not found: {model_path}")

    checkpoint = _load_torch_checkpoint(model_path, device)

    feature_columns = list(checkpoint.get("feature_columns", feature_columns))

    model = LSTMXiClassifier(
        input_dim=len(feature_columns),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return LSTMBaselineArtifact(
        model_name=cfg.model_name,
        model=model,
        config=cfg,
        model_path=model_path,
        feature_columns=feature_columns,
        window_spec=window_spec,
        best_epoch=int(checkpoint.get("epoch", -1)),
        best_val_loss=float(checkpoint.get("val_loss", float("nan"))),
        train_summary={},
        val_summary={},
        history=[],
        fit_runtime_seconds=0.0,
        active_seed=int(active_seed),
        device=str(device),
    )


def _sort_and_deduplicate_bundle(bundle: EvaluationPredictionBundle) -> EvaluationPredictionBundle:
    """Sort and average duplicate row predictions from overlapping windows."""
    if len(bundle.row_indices) == 0:
        return bundle

    df = pd.DataFrame(
        {
            "row_index": bundle.row_indices,
            "probability": bundle.probabilities,
            "logit": bundle.logits,
            "label": bundle.labels,
            "valid_mask": bundle.valid_mask,
            "segment_id": bundle.segment_ids.astype(str),
            "delta_t": bundle.delta_t,
        }
    )

    grouped = (
        df.groupby("row_index", sort=True)
        .agg(
            probability=("probability", "mean"),
            logit=("logit", "mean"),
            label=("label", "first"),
            valid_mask=("valid_mask", "max"),
            segment_id=("segment_id", "first"),
            delta_t=("delta_t", "first"),
        )
        .reset_index()
    )

    return EvaluationPredictionBundle(
        split_name=bundle.split_name,
        probabilities=grouped["probability"].to_numpy(dtype=np.float32),
        logits=grouped["logit"].to_numpy(dtype=np.float32),
        labels=grouped["label"].to_numpy(dtype=np.int64),
        valid_mask=grouped["valid_mask"].to_numpy(dtype=np.float32),
        segment_ids=grouped["segment_id"].to_numpy(dtype=object),
        row_indices=grouped["row_index"].to_numpy(dtype=np.int64),
        delta_t=grouped["delta_t"].to_numpy(dtype=np.float32),
        checkpoint_path=bundle.checkpoint_path,
        model_name=bundle.model_name,
    )


@torch.no_grad()
def collect_lstm_predictions(
    config: Mapping[str, Any],
    artifact: LSTMBaselineArtifact,
    split_name: str,
    device: Optional[torch.device] = None,
) -> EvaluationPredictionBundle:
    """Collect LSTM predictions for one split."""
    if device is None:
        device = next(artifact.model.parameters()).device

    dataset = build_sequence_dataset(
        config=config,
        split_name=split_name,
        feature_columns=artifact.feature_columns,
        window_spec=artifact.window_spec,
        train=False,
    )

    loader = build_sequence_dataloader(
        dataset=dataset,
        batch_size=artifact.config.eval_batch_size,
        shuffle=False,
        active_seed=artifact.active_seed,
        num_workers=artifact.config.num_workers,
    )

    model = artifact.model.to(device)
    model.eval()

    probabilities: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    segment_ids: List[str] = []
    row_indices: List[int] = []
    delta_t_values: List[float] = []

    for batch in loader:
        batch = move_sequence_batch_to_device(batch, device)

        batch_logits = model(batch["x"])
        batch_probs = torch.sigmoid(batch_logits)

        probs_np = batch_probs.detach().cpu().numpy()
        logits_np = batch_logits.detach().cpu().numpy()
        labels_np = batch["y"].detach().cpu().numpy()
        valid_np = (batch["loss_mask"] * batch["padding_mask"]).detach().cpu().numpy()
        rows_np = batch["row_indices"].detach().cpu().numpy()
        delta_np = batch["delta_t"].detach().cpu().numpy()
        real_lengths = batch["real_length"].detach().cpu().numpy().astype(int)
        batch_segment_ids = [str(item) for item in batch["segment_id"]]

        for i in range(probs_np.shape[0]):
            real_len = int(real_lengths[i])

            probabilities.append(probs_np[i, :real_len].reshape(-1))
            logits.append(logits_np[i, :real_len].reshape(-1))
            labels.append(labels_np[i, :real_len].reshape(-1))
            valid_masks.append(valid_np[i, :real_len].reshape(-1))

            row_indices.extend(rows_np[i, :real_len].astype(int).tolist())
            delta_t_values.extend(delta_np[i, :real_len].astype(float).tolist())
            segment_ids.extend([batch_segment_ids[i]] * real_len)

    if probabilities:
        p = np.concatenate(probabilities).astype(np.float32)
        z = np.concatenate(logits).astype(np.float32)
        y = np.concatenate(labels).astype(np.int64)
        m = np.concatenate(valid_masks).astype(np.float32)
    else:
        p = np.asarray([], dtype=np.float32)
        z = np.asarray([], dtype=np.float32)
        y = np.asarray([], dtype=np.int64)
        m = np.asarray([], dtype=np.float32)

    bundle = EvaluationPredictionBundle(
        split_name=str(split_name),
        probabilities=p,
        logits=z,
        labels=y,
        valid_mask=m,
        segment_ids=np.asarray(segment_ids, dtype=object),
        row_indices=np.asarray(row_indices, dtype=np.int64),
        delta_t=np.asarray(delta_t_values, dtype=np.float32),
        checkpoint_path=str(artifact.model_path),
        model_name=artifact.model_name,
    )

    return _sort_and_deduplicate_bundle(bundle)


__all__ = [
    "SequenceWindowSpec",
    "SequenceWindowSummary",
    "LSTMBaselineConfig",
    "SequenceEpochResult",
    "LSTMBaselineArtifact",
    "SequenceXiWindowDataset",
    "collate_sequence_windows",
    "LSTMXiClassifier",
    "build_sequence_window_spec",
    "build_lstm_baseline_config",
    "build_sequence_dataset",
    "build_sequence_dataloader",
    "compute_binary_auprc",
    "compute_accuracy",
    "masked_bce_with_logits",
    "move_sequence_batch_to_device",
    "save_lstm_checkpoint",
    "train_lstm_baseline",
    "load_lstm_baseline",
    "collect_lstm_predictions",
]