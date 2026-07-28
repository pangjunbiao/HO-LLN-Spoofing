"""
Dataset-1 evaluation for the full proposed model.

Step 13 responsibilities:
- load trained full proposed checkpoint,
- collect validation probabilities,
- select real threshold theta and persistence N_p on Dataset-1 validation only,
- evaluate Dataset-1 internal test,
- save Dataset-1 main comparison table,
- save threshold-selection artifacts,
- save prediction artifacts for diagnostics.

Important:
- This file does not use the synthetic Step-10 theta=0.55 as the final threshold.
- theta and N_p are selected from real validation predictions from the trained model.
- Dataset-1 internal test is evaluated only after validation selection.
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
from torch import nn
from torch.utils.data import DataLoader

from src.models.model_factory import build_model
from src.training.trainer import (
    SequenceWindowSpec,
    XiWindowDataset,
    collate_xi_windows,
)
from src.utils.config import get_by_path, resolve_project_path
from src.utils.device import move_to_device, setup_device_from_config
from src.utils.io import ensure_dir, save_json
from src.utils.seed import make_torch_generator, seed_worker

from src.evaluation.result_tables import (
    extract_primary_metrics,
    metrics_to_dataset1_row,
    print_primary_metric_table,
    save_dataset1_main_comparison_row,
)


DEFAULT_STEP13_FEATURE_COLUMNS = [
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
class EvaluationPredictionBundle:
    """Flattened prediction bundle for one split."""

    split_name: str
    probabilities: np.ndarray
    logits: np.ndarray
    labels: np.ndarray
    valid_mask: np.ndarray
    segment_ids: np.ndarray
    row_indices: np.ndarray
    delta_t: np.ndarray

    checkpoint_path: Optional[str] = None
    model_name: str = "Proposed"

    def to_dict_summary(self) -> Dict[str, Any]:
        valid = self.valid_mask > 0.5
        labels_valid = self.labels[valid] if valid.any() else np.asarray([], dtype=np.int64)

        return {
            "split_name": self.split_name,
            "rows": int(len(self.labels)),
            "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "normal_valid_rows": int((labels_valid == 0).sum()) if labels_valid.size else 0,
            "attack_valid_rows": int((labels_valid == 1).sum()) if labels_valid.size else 0,
            "segments": int(len(set(str(x) for x in self.segment_ids))),
            "probability_min": float(np.min(self.probabilities)) if len(self.probabilities) else None,
            "probability_max": float(np.max(self.probabilities)) if len(self.probabilities) else None,
            "probability_mean": float(np.mean(self.probabilities)) if len(self.probabilities) else None,
            "checkpoint_path": self.checkpoint_path,
            "model_name": self.model_name,
        }

    def save_npz(self, output_path: Path | str) -> Path:
        """Save prediction bundle as NPZ."""
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        np.savez_compressed(
            output_path,
            split_name=np.asarray([self.split_name], dtype=object),
            probabilities=self.probabilities.astype(np.float32),
            logits=self.logits.astype(np.float32),
            labels=self.labels.astype(np.int64),
            valid_mask=self.valid_mask.astype(np.float32),
            segment_ids=self.segment_ids.astype(object),
            row_indices=self.row_indices.astype(np.int64),
            delta_t=self.delta_t.astype(np.float32),
            checkpoint_path=np.asarray([self.checkpoint_path or ""], dtype=object),
            model_name=np.asarray([self.model_name], dtype=object),
        )

        return output_path


@dataclass
class AttackEvent:
    """Contiguous valid attack block."""

    event_id: int
    segment_id: str
    start_index: int
    end_index: int
    start_position: int
    end_position: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdSelectionResult:
    """Selected validation threshold/persistence payload."""

    theta: float
    persistence: int
    objective: str
    monitor_split: str
    selected_metric_value: Optional[float]
    selected_candidate: Dict[str, Any]
    candidate_count: int
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetEvaluationResult:
    """Evaluation result for one dataset split."""

    model_name: str
    split_name: str
    metrics: Dict[str, Any]
    threshold: float
    persistence: int
    checkpoint_path: Optional[str]
    prediction_summary: Dict[str, Any]
    artifact_paths: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any) -> Optional[float]:
    """Convert to finite float or None."""
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


def _project_path(config: Mapping[str, Any], path_value: str) -> Path:
    """Resolve project-relative path."""
    return resolve_project_path(config, path_value)


def _label_series_to_binary(labels: pd.Series) -> np.ndarray:
    """Convert label series to binary labels."""
    values = labels.to_numpy()

    if values.dtype.kind in {"U", "S", "O"}:
        attack_words = {
            "attack",
            "attacked",
            "spoof",
            "spoofing",
            "malicious",
            "anomaly",
            "1",
            "true",
        }
        return np.asarray(
            [1 if str(item).strip().lower() in attack_words else 0 for item in values],
            dtype=np.int64,
        )

    return (values.astype(np.float32) >= 0.5).astype(np.int64)


def _resolve_xi_split_path(config: Mapping[str, Any], split_name: str) -> Path:
    """Resolve processed xi split path."""
    defaults = {
        "train": "data/processed/train_xi.csv",
        "val": "data/processed/val_xi.csv",
        "test": "data/processed/test_xi.csv",
        "external": "data/processed/external_xi.csv",
        "online": "data/processed/online_xi.csv",
    }

    if split_name not in defaults:
        raise ValueError(f"Unknown split_name={split_name!r}.")

    configured = get_by_path(
        config,
        f"training.dataset.xi_split_files.{split_name}",
        defaults[split_name],
    )

    return _project_path(config, str(configured))


def get_step13_paths(config: Mapping[str, Any]) -> Dict[str, Path]:
    """Resolve Step-13 output paths."""
    return {
        "dataset1_table": _project_path(
            config,
            str(get_by_path(config, "paths.dataset1_main_comparison_csv", "results/tables/dataset1_main_comparison.csv")),
        ),
        "dataset1_summary_json": _project_path(
            config,
            str(get_by_path(config, "paths.dataset1_proposed_summary_json", "results/tables/dataset1_proposed_summary.json")),
        ),
        "threshold_selection_json": _project_path(
            config,
            str(get_by_path(config, "paths.proposed_threshold_selection_json", "results/tables/proposed_threshold_selection.json")),
        ),
        "threshold_candidates_csv": _project_path(
            config,
            str(get_by_path(config, "paths.proposed_threshold_candidates_csv", "results/tables/proposed_threshold_candidates.csv")),
        ),
        "val_predictions_npz": _project_path(
            config,
            str(get_by_path(config, "paths.dataset1_val_predictions_npz", "results/tables/dataset1_val_predictions.npz")),
        ),
        "test_predictions_npz": _project_path(
            config,
            str(get_by_path(config, "paths.dataset1_test_predictions_npz", "results/tables/dataset1_test_predictions.npz")),
        ),
    }


def get_checkpoint_path(config: Mapping[str, Any], checkpoint_path: Optional[str] = None) -> Path:
    """Resolve proposed best checkpoint path."""
    if checkpoint_path is not None:
        return _project_path(config, checkpoint_path)

    models_dir = str(get_by_path(config, "paths.models_dir", "results/models"))
    best_name = str(get_by_path(config, "training.checkpointing.best_checkpoint_name", "proposed_best.pt"))

    return _project_path(config, str(Path(models_dir) / best_name))


def build_evaluation_dataset(
    config: Mapping[str, Any],
    split_name: str,
    full_sequence: bool = False,
) -> XiWindowDataset:
    """Build an evaluation dataset for any processed xi split."""
    csv_path = _resolve_xi_split_path(config, split_name)

    if not csv_path.exists():
        raise FileNotFoundError(f"Processed xi split not found for {split_name}: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    feature_columns = list(
        get_by_path(
            config,
            "training.dataset.feature_columns",
            get_by_path(
                config,
                "model.input.recommended_model_input_columns",
                DEFAULT_STEP13_FEATURE_COLUMNS,
            ),
        )
    )

    label_column = str(get_by_path(config, "training.dataset.label_column", "Data Type"))
    validity_column = str(get_by_path(config, "training.dataset.validity_column", "xi_nu"))
    segment_column = str(get_by_path(config, "training.dataset.segment_column", "segment_id"))
    order_column = str(get_by_path(config, "training.dataset.order_column", "within_segment_index"))
    delta_t_column = str(get_by_path(config, "training.dataset.delta_t_column", "delta_t_seconds"))

    if full_sequence:
        window_length = max(int(len(df)), 1)
        stride = window_length
    else:
        window_length = int(get_by_path(config, "training.dataset.sequence.window_length_eval", 512))
        stride = int(get_by_path(config, "training.dataset.sequence.stride_eval", 512))

    window_spec = SequenceWindowSpec(
        window_length=window_length,
        stride=stride,
        include_last_partial=True,
        min_valid_rows=int(get_by_path(config, "training.dataset.sequence.min_valid_rows", 1)),
    )

    return XiWindowDataset(
        dataframe=df,
        split_name=split_name,
        feature_columns=feature_columns,
        label_column=label_column,
        validity_column=validity_column,
        segment_column=segment_column,
        order_column=order_column,
        delta_t_column=delta_t_column,
        window_spec=window_spec,
        independent_windows=True,
    )


def build_evaluation_dataloader(
    config: Mapping[str, Any],
    split_name: str,
    active_seed: int,
    full_sequence: bool = False,
) -> Tuple[DataLoader, XiWindowDataset]:
    """Build evaluation dataloader."""
    dataset = build_evaluation_dataset(
        config=config,
        split_name=split_name,
        full_sequence=full_sequence,
    )

    batch_size = int(get_by_path(config, "training.dataloader.batch_size_eval", 8))
    num_workers = int(get_by_path(config, "training.dataloader.num_workers", 0))
    pin_memory = bool(get_by_path(config, "training.dataloader.pin_memory", True))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(active_seed + 1000),
        collate_fn=collate_xi_windows,
        drop_last=False,
    )

    return loader, dataset


def load_trained_model_for_evaluation(
    config: Mapping[str, Any],
    checkpoint_path: Optional[str] = None,
    device: Optional[Any] = None,
    variant_name: Optional[str] = None,
) -> Tuple[nn.Module, Path, Dict[str, Any]]:
    """Build model and load trained checkpoint."""
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    if variant_name is None:
        variant_name = str(
            get_by_path(
                config,
                "training.step12.variant_name",
                get_by_path(config, "model.proposed.variant_name", "full"),
            )
        )

    checkpoint = get_checkpoint_path(config, checkpoint_path)

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run Step 12 first."
        )

    model, build_info, _variant_config = build_model(
        config=config,
        variant_name=variant_name,
        device=device,
    )

    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload.get("model_state_dict", payload)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "active_seed": payload.get("active_seed"),
        "variant_name": variant_name,
        "model_build_info": build_info.to_dict() if hasattr(build_info, "to_dict") else str(build_info),
    }

    return model, checkpoint, metadata


@torch.no_grad()
def collect_model_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Any,
    split_name: str,
    checkpoint_path: Optional[str] = None,
    model_name: str = "Proposed",
) -> EvaluationPredictionBundle:
    """Collect flattened model predictions from an evaluation dataloader."""
    model.eval()

    probabilities: List[np.ndarray] = []
    logits: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []
    segment_ids: List[str] = []
    row_indices: List[int] = []
    delta_t_values: List[float] = []

    for batch in dataloader:
        batch = move_to_device(batch, device)
        output = model(batch)

        batch_probs = output.probabilities.detach().cpu().numpy()
        batch_logits = output.logits.detach().cpu().numpy()
        batch_labels = batch["y"].detach().cpu().numpy()
        batch_valid = (batch["loss_mask"] * batch["padding_mask"]).detach().cpu().numpy()
        batch_delta = batch["delta_t"].detach().cpu().numpy()
        real_lengths = batch["real_length"].detach().cpu().numpy().astype(int)
        start_indices = batch["start_index"].detach().cpu().numpy().astype(int)
        batch_segment_ids = [str(item) for item in batch["segment_id"]]

        for i in range(batch_probs.shape[0]):
            real_len = int(real_lengths[i])
            start = int(start_indices[i])
            seg_id = batch_segment_ids[i]

            probabilities.append(batch_probs[i, :real_len].reshape(-1))
            logits.append(batch_logits[i, :real_len].reshape(-1))
            labels.append(batch_labels[i, :real_len].reshape(-1))
            valid_masks.append(batch_valid[i, :real_len].reshape(-1))

            segment_ids.extend([seg_id] * real_len)
            row_indices.extend(list(range(start, start + real_len)))
            delta_t_values.extend(batch_delta[i, :real_len].reshape(-1).astype(float).tolist())

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
        split_name=split_name,
        probabilities=p,
        logits=z,
        labels=y,
        valid_mask=m,
        segment_ids=np.asarray(segment_ids, dtype=object),
        row_indices=np.asarray(row_indices, dtype=np.int64),
        delta_t=np.asarray(delta_t_values, dtype=np.float32),
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )

    return sort_and_deduplicate_bundle(bundle)


def sort_and_deduplicate_bundle(bundle: EvaluationPredictionBundle) -> EvaluationPredictionBundle:
    """
    Sort by row index and average duplicate probabilities if duplicates exist.

    Evaluation windows are normally non-overlapping, but this protects against
    accidental overlap in config.
    """
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


def _average_precision_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy average precision fallback."""
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    positives = int((y_true == 1).sum())
    if positives == 0:
        return None

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)
    recall_step = (y_sorted == 1) / positives

    return float(np.sum(precision * recall_step))


def _roc_auc_score_numpy(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Pure NumPy AUROC fallback using rank statistic."""
    y_true = y_true.astype(int)
    y_score = y_score.astype(float)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


def compute_probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Compute AUPRC/AUROC on valid rows only."""
    keep = valid_mask > 0.5

    if keep.sum() == 0:
        return {"auprc": None, "auroc": None}

    y = labels[keep].astype(int)
    p = probabilities[keep].astype(float)

    auprc = None
    auroc = None

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y)) >= 2:
            auprc = float(average_precision_score(y, p))
            auroc = float(roc_auc_score(y, p))
        elif int(y.sum()) > 0:
            auprc = float(average_precision_score(y, p))
    except Exception:
        auprc = _average_precision_score_numpy(y, p)
        auroc = _roc_auc_score_numpy(y, p)

    return {
        "auprc": _safe_float(auprc),
        "auroc": _safe_float(auroc),
    }


def apply_persistence_alarm(
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    theta: float,
    persistence: int,
) -> np.ndarray:
    """
    Apply c_t = I(p_t >= theta) and confirmed alarm after N_p positives.

    Invalid rows break the positive run.
    Segment changes reset the positive run.
    """
    theta = float(theta)
    persistence = int(persistence)

    if persistence <= 0:
        raise ValueError("persistence must be >= 1.")

    confirmed = np.zeros_like(probabilities, dtype=np.int64)
    run_length = 0
    previous_segment: Optional[str] = None

    for i in range(len(probabilities)):
        segment = str(segment_ids[i])

        if previous_segment is None or segment != previous_segment:
            run_length = 0
            previous_segment = segment

        if valid_mask[i] <= 0.5:
            run_length = 0
            confirmed[i] = 0
            continue

        raw_positive = int(probabilities[i] >= theta)

        if raw_positive:
            run_length += 1
        else:
            run_length = 0

        confirmed[i] = 1 if run_length >= persistence else 0

    return confirmed


def find_attack_events(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
) -> List[AttackEvent]:
    """Find contiguous valid attack events per segment."""
    events: List[AttackEvent] = []
    event_id = 0

    in_event = False
    start_pos = -1
    start_index = -1
    current_segment = None

    for i in range(len(labels)):
        segment = str(segment_ids[i])
        is_attack = bool(labels[i] == 1 and valid_mask[i] > 0.5)

        if current_segment is None:
            current_segment = segment

        segment_changed = segment != current_segment

        if segment_changed and in_event:
            events.append(
                AttackEvent(
                    event_id=event_id,
                    segment_id=str(current_segment),
                    start_index=start_index,
                    end_index=i - 1,
                    start_position=start_pos,
                    end_position=i - 1,
                )
            )
            event_id += 1
            in_event = False

        if segment_changed:
            current_segment = segment

        if is_attack and not in_event:
            in_event = True
            start_pos = i
            start_index = i

        if in_event and (not is_attack):
            events.append(
                AttackEvent(
                    event_id=event_id,
                    segment_id=str(current_segment),
                    start_index=start_index,
                    end_index=i - 1,
                    start_position=start_pos,
                    end_position=i - 1,
                )
            )
            event_id += 1
            in_event = False

    if in_event:
        events.append(
            AttackEvent(
                event_id=event_id,
                segment_id=str(current_segment),
                start_index=start_index,
                end_index=len(labels) - 1,
                start_position=start_pos,
                end_position=len(labels) - 1,
            )
        )

    return events


def _cumulative_time_seconds(delta_t: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Build within-segment cumulative time in seconds."""
    cumulative = np.zeros(len(delta_t), dtype=np.float64)

    previous_segment = None
    running = 0.0

    for i in range(len(delta_t)):
        segment = str(segment_ids[i])

        if previous_segment is None or segment != previous_segment:
            running = 0.0
            previous_segment = segment
        else:
            dt = float(delta_t[i])
            if math.isfinite(dt) and dt > 0.0:
                running += dt

        cumulative[i] = running

    return cumulative


def compute_event_detection_metrics(
    labels: np.ndarray,
    confirmed_alarm: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    delta_t: np.ndarray,
) -> Dict[str, Any]:
    """Compute attack-event detection rate and detection delays."""
    events = find_attack_events(
        labels=labels,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
    )

    cumulative_time = _cumulative_time_seconds(delta_t, segment_ids)

    detected_count = 0
    delays: List[float] = []
    event_payloads: List[Dict[str, Any]] = []

    for event in events:
        alarm_positions = np.where(
            confirmed_alarm[event.start_position : event.end_position + 1] == 1
        )[0]

        detected = len(alarm_positions) > 0
        delay = None

        if detected:
            detected_count += 1
            first_alarm_pos = event.start_position + int(alarm_positions[0])
            delay = float(
                cumulative_time[first_alarm_pos] - cumulative_time[event.start_position]
            )
            delays.append(delay)

        payload = event.to_dict()
        payload["detected"] = bool(detected)
        payload["delay_seconds"] = delay
        event_payloads.append(payload)

    total_events = len(events)

    return {
        "attack_event_count": int(total_events),
        "detected_attack_event_count": int(detected_count),
        "attack_detection_rate": float(detected_count / total_events) if total_events else None,
        "mean_detection_delay": float(np.mean(delays)) if delays else None,
        "detection_delays": delays,
        "attack_events": event_payloads,
    }


def compute_threshold_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    valid_mask: np.ndarray,
    segment_ids: np.ndarray,
    delta_t: np.ndarray,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """Compute all threshold/alarm metrics."""
    confirmed_alarm = apply_persistence_alarm(
        probabilities=probabilities,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        theta=theta,
        persistence=persistence,
    )

    keep = valid_mask > 0.5

    y = labels[keep].astype(int)
    pred = confirmed_alarm[keep].astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    event_metrics = compute_event_detection_metrics(
        labels=labels,
        confirmed_alarm=confirmed_alarm,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        delta_t=delta_t,
    )

    return {
        "theta": float(theta),
        "persistence": int(persistence),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "confirmed_positive_count": int(confirmed_alarm.sum()),
        "false_alarms": fp,
        **event_metrics,
    }


def evaluate_bundle_with_threshold(
    bundle: EvaluationPredictionBundle,
    theta: float,
    persistence: int,
) -> Dict[str, Any]:
    """Evaluate one prediction bundle with selected threshold/persistence."""
    start_time = time.perf_counter()

    probability_metrics = compute_probability_metrics(
        labels=bundle.labels,
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
    )

    threshold_metrics = compute_threshold_metrics(
        labels=bundle.labels,
        probabilities=bundle.probabilities,
        valid_mask=bundle.valid_mask,
        segment_ids=bundle.segment_ids,
        delta_t=bundle.delta_t,
        theta=theta,
        persistence=persistence,
    )

    metrics = {
        **probability_metrics,
        **threshold_metrics,
        "runtime_seconds": float(time.perf_counter() - start_time),
    }

    return metrics


def _metric_for_objective(metrics: Mapping[str, Any], objective: str) -> Optional[float]:
    """Return scalar objective value."""
    objective = str(objective).lower().strip()

    if objective in {"maximize_f1", "f1"}:
        return _safe_float(metrics.get("f1"))

    if objective in {"maximize_auprc", "auprc"}:
        return _safe_float(metrics.get("auprc"))

    if objective in {"maximize_recall", "recall"}:
        return _safe_float(metrics.get("recall"))

    if objective in {"maximize_attack_detection_rate", "attack_detection_rate", "adr"}:
        return _safe_float(metrics.get("attack_detection_rate"))

    raise ValueError(f"Unsupported threshold-selection objective: {objective}")


def _candidate_sort_key(candidate: Mapping[str, Any], objective: str) -> Tuple[Any, ...]:
    """Candidate sort key. Higher is better."""
    score = _metric_for_objective(candidate, objective)

    if score is None:
        score = -1.0e18

    adr = candidate.get("attack_detection_rate")
    delay = candidate.get("mean_detection_delay")
    fpr = candidate.get("fpr")
    recall = candidate.get("recall")

    return (
        float(score),
        float(adr) if adr is not None else -1.0,
        -float(delay) if delay is not None else -1.0e9,
        -float(fpr) if fpr is not None else -1.0e9,
        float(recall) if recall is not None else -1.0,
        -int(candidate.get("persistence", 999)),
        float(candidate.get("theta", 0.0)),
    )


def select_threshold_and_persistence(
    validation_bundle: EvaluationPredictionBundle,
    config: Mapping[str, Any],
) -> ThresholdSelectionResult:
    """
    Select theta and N_p from validation predictions only.

    This is the real model threshold-selection step.
    Synthetic Step-10 theta=0.55 is not reused.
    """
    theta_grid = list(
        get_by_path(
            config,
            "training.threshold_selection.threshold_grid",
            [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        )
    )
    persistence_grid = list(
        get_by_path(config, "training.threshold_selection.persistence_grid", [1, 2, 3, 4, 5])
    )

    objective = str(
        get_by_path(config, "training.threshold_selection.objective", "maximize_f1")
    )

    max_fpr = get_by_path(config, "training.threshold_selection.max_fpr_constraint", None)
    min_recall = get_by_path(config, "training.threshold_selection.min_recall_constraint", None)
    min_adr = get_by_path(config, "training.threshold_selection.min_attack_detection_rate_constraint", None)

    candidates: List[Dict[str, Any]] = []

    for theta in theta_grid:
        for persistence in persistence_grid:
            metrics = evaluate_bundle_with_threshold(
                bundle=validation_bundle,
                theta=float(theta),
                persistence=int(persistence),
            )

            candidate = {
                "split": "val",
                "theta": float(theta),
                "persistence": int(persistence),
                **metrics,
            }

            constraints_ok = True

            if max_fpr is not None and candidate["fpr"] > float(max_fpr):
                constraints_ok = False

            if min_recall is not None and candidate["recall"] < float(min_recall):
                constraints_ok = False

            if min_adr is not None:
                adr_value = candidate.get("attack_detection_rate")
                if adr_value is None or adr_value < float(min_adr):
                    constraints_ok = False

            candidate["constraints_ok"] = bool(constraints_ok)
            candidates.append(candidate)

    valid_candidates = [candidate for candidate in candidates if candidate["constraints_ok"]]

    if not valid_candidates:
        raise RuntimeError("Threshold selection failed: no candidates satisfied constraints.")

    selected = sorted(
        valid_candidates,
        key=lambda item: _candidate_sort_key(item, objective),
        reverse=True,
    )[0]

    selected_value = _metric_for_objective(selected, objective)

    return ThresholdSelectionResult(
        theta=float(selected["theta"]),
        persistence=int(selected["persistence"]),
        objective=objective,
        monitor_split="val",
        selected_metric_value=selected_value,
        selected_candidate=selected,
        candidate_count=len(candidates),
        candidates=candidates,
    )


def save_threshold_selection_artifacts(
    selection: ThresholdSelectionResult,
    config: Mapping[str, Any],
) -> Dict[str, str]:
    """Save threshold-selection JSON and candidate CSV."""
    paths = get_step13_paths(config)

    ensure_dir(paths["threshold_selection_json"].parent)
    ensure_dir(paths["threshold_candidates_csv"].parent)

    save_json(selection.to_dict(), paths["threshold_selection_json"], indent=2)

    candidate_df = pd.DataFrame(selection.candidates)
    candidate_df.to_csv(paths["threshold_candidates_csv"], index=False)

    return {
        "threshold_selection_json": str(paths["threshold_selection_json"]),
        "threshold_candidates_csv": str(paths["threshold_candidates_csv"]),
    }


def run_dataset1_evaluation(
    config: Mapping[str, Any],
    active_seed: int = 42,
    checkpoint_path: Optional[str] = None,
    model_name: str = "Proposed",
    device: Optional[Any] = None,
) -> DatasetEvaluationResult:
    """
    Run Dataset-1 validation threshold selection and internal-test evaluation.

    Returns:
        DatasetEvaluationResult for Dataset-1 internal test.
    """
    if device is None:
        device_info = setup_device_from_config(config, verbose=True)
        device = device_info.device

    start_time = time.perf_counter()

    model, checkpoint, checkpoint_metadata = load_trained_model_for_evaluation(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        variant_name="full",
    )

    val_loader, val_dataset = build_evaluation_dataloader(
        config=config,
        split_name="val",
        active_seed=active_seed,
        full_sequence=False,
    )
    test_loader, test_dataset = build_evaluation_dataloader(
        config=config,
        split_name="test",
        active_seed=active_seed,
        full_sequence=False,
    )

    print("=" * 100)
    print("STEP 13 DATASET-1 EVALUATION")
    print("=" * 100)
    print(f"Model                  : {model_name}")
    print(f"Checkpoint             : {checkpoint}")
    print(f"Validation rows/windows: {val_dataset.summary()['rows']} / {val_dataset.summary()['windows']}")
    print(f"Test rows/windows      : {test_dataset.summary()['rows']} / {test_dataset.summary()['windows']}")
    print("Selecting theta and N_p on validation only.")
    print("=" * 100)

    val_bundle = collect_model_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        split_name="val",
        checkpoint_path=str(checkpoint),
        model_name=model_name,
    )

    selection = select_threshold_and_persistence(
        validation_bundle=val_bundle,
        config=config,
    )

    threshold_artifacts = save_threshold_selection_artifacts(
        selection=selection,
        config=config,
    )

    test_bundle = collect_model_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        split_name="test",
        checkpoint_path=str(checkpoint),
        model_name=model_name,
    )

    test_metrics = evaluate_bundle_with_threshold(
        bundle=test_bundle,
        theta=selection.theta,
        persistence=selection.persistence,
    )
    test_metrics["runtime_seconds"] = float(time.perf_counter() - start_time)

    paths = get_step13_paths(config)
    val_npz = val_bundle.save_npz(paths["val_predictions_npz"])
    test_npz = test_bundle.save_npz(paths["test_predictions_npz"])

    table_row = metrics_to_dataset1_row(
        model_name=model_name,
        split="Dataset-1 Test",
        metrics=test_metrics,
        threshold=selection.theta,
        persistence=selection.persistence,
        checkpoint_path=str(checkpoint),
        notes="Full proposed model; theta and N_p selected on Dataset-1 validation only.",
    )

    save_dataset1_main_comparison_row(
        output_path=paths["dataset1_table"],
        row=table_row,
    )

    result = DatasetEvaluationResult(
        model_name=model_name,
        split_name="Dataset-1 Test",
        metrics=test_metrics,
        threshold=selection.theta,
        persistence=selection.persistence,
        checkpoint_path=str(checkpoint),
        prediction_summary=test_bundle.to_dict_summary(),
        artifact_paths={
            "dataset1_table": str(paths["dataset1_table"]),
            "dataset1_summary_json": str(paths["dataset1_summary_json"]),
            "validation_predictions_npz": str(val_npz),
            "test_predictions_npz": str(test_npz),
            **threshold_artifacts,
        },
    )

    summary_payload = {
        "result": result.to_dict(),
        "validation_prediction_summary": val_bundle.to_dict_summary(),
        "test_prediction_summary": test_bundle.to_dict_summary(),
        "threshold_selection": {
            key: value
            for key, value in selection.to_dict().items()
            if key != "candidates"
        },
        "checkpoint_metadata": checkpoint_metadata,
        "val_dataset_summary": val_dataset.summary(),
        "test_dataset_summary": test_dataset.summary(),
        "leakage_rules": {
            "threshold_selected_on_validation_only": True,
            "test_not_used_for_threshold_selection": True,
            "dataset2_not_used_for_threshold_selection": True,
            "dataset3_not_used_for_threshold_selection": True,
            "synthetic_step10_theta_not_used": True,
        },
    }

    save_json(summary_payload, paths["dataset1_summary_json"], indent=2)

    primary_row = {
        "Model": model_name,
        **extract_primary_metrics(test_metrics),
    }
    print_primary_metric_table(
        title="STEP 13 DATASET-1 INTERNAL TEST PRIMARY METRICS",
        rows=[primary_row],
        model_key="Model",
    )

    print("Selected validation threshold/persistence:")
    print(f"  theta       : {selection.theta}")
    print(f"  persistence : {selection.persistence}")
    print(f"  objective   : {selection.objective}")
    print(f"  selected validation objective value: {selection.selected_metric_value}")
    print("Saved artifacts:")
    for key, value in result.artifact_paths.items():
        print(f"  {key}: {value}")
    print("=" * 100)

    return result


__all__ = [
    "EvaluationPredictionBundle",
    "AttackEvent",
    "ThresholdSelectionResult",
    "DatasetEvaluationResult",
    "get_step13_paths",
    "get_checkpoint_path",
    "build_evaluation_dataset",
    "build_evaluation_dataloader",
    "load_trained_model_for_evaluation",
    "collect_model_predictions",
    "sort_and_deduplicate_bundle",
    "compute_probability_metrics",
    "apply_persistence_alarm",
    "find_attack_events",
    "compute_event_detection_metrics",
    "compute_threshold_metrics",
    "evaluate_bundle_with_threshold",
    "select_threshold_and_persistence",
    "save_threshold_selection_artifacts",
    "run_dataset1_evaluation",
]