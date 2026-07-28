"""
GRU-xi baseline for causal GNSS spoofing detection.

Step 15 purpose:
- Train a standard GRU sequence baseline on the same reconstructed xi_t features.
- Use Dataset-1 train only for fitting.
- Use Dataset-1 validation only for early stopping and threshold selection later.
- Evaluate Dataset-1 test, Dataset-2 external, and Dataset-3 online later.
- Do not use raw shortcut columns.

Important:
- This is a fair standard GRU baseline, not an artificially weakened baseline.
- It uses the same xi_t input columns as the proposed model.
- It does not use Kirchhoff exchange, third-order fusion, weak-accumulation modules,
  or liquid second-order dynamics.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.baselines.lstm_baseline import (
    SequenceEpochResult,
    SequenceWindowSpec,
    build_sequence_dataloader,
    build_sequence_dataset,
    build_sequence_window_spec,
    collect_lstm_predictions,
    compute_accuracy,
    compute_binary_auprc,
    masked_bce_with_logits,
    move_sequence_batch_to_device,
)
from src.baselines.xgboost_baseline import (
    compute_scale_pos_weight,
    get_baseline_feature_columns,
)
from src.evaluation.evaluate_dataset1 import EvaluationPredictionBundle
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import setup_device_from_config
from src.utils.io import ensure_dir
from src.utils.seed import set_global_seed


@dataclass
class GRUBaselineConfig:
    """Configuration for GRU-xi baseline."""

    model_name: str = "GRU-xi"

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

    output_model_path: str = "results/models/gru_xi.pt"
    output_history_csv: str = "results/tables/gru_xi_training_history.csv"
    output_summary_json: str = "results/tables/gru_xi_training_summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GRUBaselineArtifact:
    """Trained GRU baseline artifact."""

    model_name: str
    model: nn.Module
    config: GRUBaselineConfig
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


class GRUXiClassifier(nn.Module):
    """Standard GRU time-step classifier."""

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

        self.gru = nn.GRU(
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
        sequence_output, _hidden = self.gru(x)
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


def build_gru_baseline_config(
    config: Mapping[str, Any],
    active_seed: int = 42,
) -> GRUBaselineConfig:
    """Build GRU baseline config."""
    base = "baselines.gru_xi"

    return GRUBaselineConfig(
        model_name=str(get_by_path(config, f"{base}.model_name", "GRU-xi")),
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
            get_by_path(config, f"{base}.output_model_path", "results/models/gru_xi.pt")
        ),
        output_history_csv=str(
            get_by_path(
                config,
                f"{base}.output_history_csv",
                "results/tables/gru_xi_training_history.csv",
            )
        ),
        output_summary_json=str(
            get_by_path(
                config,
                f"{base}.output_summary_json",
                "results/tables/gru_xi_training_summary.json",
            )
        ),
    )


def _make_pos_weight_from_dataset(
    dataset: Any,
    use_train_class_weight: bool,
    device: torch.device,
) -> Optional[Tensor]:
    """Make train-only positive class weight."""
    if not use_train_class_weight:
        return None

    y_valid = dataset.y_all[dataset.valid_mask_all > 0.5]
    scale_pos_weight = compute_scale_pos_weight(y_valid)

    return torch.tensor([scale_pos_weight], dtype=torch.float32, device=device)


def _train_one_epoch_gru(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[Tensor],
    gradient_clip_norm: float,
) -> float:
    """Train one epoch for GRU model."""
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
def _evaluate_gru_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: Optional[Tensor],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate GRU model and return val loss + valid probabilities/labels."""
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


def save_gru_checkpoint(
    model: nn.Module,
    model_path: Path | str,
    cfg: GRUBaselineConfig,
    window_spec: SequenceWindowSpec,
    feature_columns: Sequence[str],
    epoch: int,
    val_loss: float,
    active_seed: int,
) -> Path:
    """Save GRU baseline checkpoint."""
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


def train_gru_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> GRUBaselineArtifact:
    """Train GRU-xi baseline."""
    start_time = time.perf_counter()

    set_global_seed(int(active_seed))

    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_gru_baseline_config(config=config, active_seed=active_seed)
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

    model = GRUXiClassifier(
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
    print("STEP 15 BASELINE TRAINING: GRU-xi")
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

        train_loss = _train_one_epoch_gru(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            pos_weight=pos_weight,
            gradient_clip_norm=cfg.gradient_clip_norm,
        )

        val_loss, val_probs, val_labels = _evaluate_gru_model(
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

            save_gru_checkpoint(
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
        save_gru_checkpoint(
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

    artifact = GRUBaselineArtifact(
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

    print("GRU-xi training completed.")
    print(f"Model path          : {model_path}")
    print(f"History CSV         : {history_csv}")
    print(f"Summary JSON        : {summary_json}")
    print(f"Best epoch          : {best_epoch}")
    print(f"Best val loss       : {best_val_loss}")
    print(f"Runtime seconds     : {fit_runtime:.3f}")
    print("=" * 100)

    return artifact


def load_gru_baseline(
    config: Mapping[str, Any],
    active_seed: int = 42,
    device: Optional[torch.device] = None,
) -> GRUBaselineArtifact:
    """Load saved GRU baseline."""
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    cfg = build_gru_baseline_config(config=config, active_seed=active_seed)
    feature_columns = get_baseline_feature_columns(config)
    window_spec = build_sequence_window_spec(config)
    model_path = _project_path(config, cfg.output_model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Saved GRU baseline not found: {model_path}")

    checkpoint = _load_torch_checkpoint(model_path, device)

    feature_columns = list(checkpoint.get("feature_columns", feature_columns))

    model = GRUXiClassifier(
        input_dim=len(feature_columns),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return GRUBaselineArtifact(
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


@torch.no_grad()
def collect_gru_predictions(
    config: Mapping[str, Any],
    artifact: GRUBaselineArtifact,
    split_name: str,
    device: Optional[torch.device] = None,
) -> EvaluationPredictionBundle:
    """
    Collect GRU predictions for one split.

    Uses the same sequence-window prediction logic as LSTM, but with the GRU artifact.
    """
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

    # Reuse the exact LSTM deduplication behavior through collect_lstm_predictions?
    # No: build directly here to avoid misleading model name. Use local equivalent.
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


__all__ = [
    "GRUBaselineConfig",
    "GRUBaselineArtifact",
    "GRUXiClassifier",
    "build_gru_baseline_config",
    "save_gru_checkpoint",
    "train_gru_baseline",
    "load_gru_baseline",
    "collect_gru_predictions",
]