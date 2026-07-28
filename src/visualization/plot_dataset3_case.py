"""
Dataset-3 online case-study visualization.

Step 19 purpose:
- show the Dataset-3 online sequence,
- compare Dataset-3 EKF Detector with the Proposed detector,
- show Proposed probability and confirmed alarms,
- show physically meaningful evidence traces.

This file is visualization-only.
It does not train, retune, or modify predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def _pick_first_existing_column(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    """Return the first candidate column found in the dataframe."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _finite_numeric(values: Sequence[Any], default: float = 0.0) -> np.ndarray:
    """Convert values to finite float array."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = np.where(np.isfinite(arr), arr, default)
    return arr.astype(float)


def _to_binary_array(values: Sequence[Any]) -> np.ndarray:
    """Convert numeric/string values to binary 0/1."""
    series = pd.Series(values)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == len(series):
        return (numeric.to_numpy(dtype=float) >= 0.5).astype(int)

    positive_tokens = {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "alarm",
        "attack",
        "spoof",
        "spoofing",
        "detected",
        "positive",
        "warning",
    }
    negative_tokens = {
        "0",
        "false",
        "f",
        "no",
        "n",
        "normal",
        "none",
        "clear",
        "negative",
        "safe",
    }

    out = np.zeros(len(series), dtype=int)

    for i, value in enumerate(series.astype(str).str.strip().str.lower()):
        if value in positive_tokens:
            out[i] = 1
        elif value in negative_tokens or value == "" or value == "nan":
            out[i] = 0
        else:
            maybe_number = pd.to_numeric(value, errors="coerce")
            if pd.notna(maybe_number):
                out[i] = int(float(maybe_number) >= 0.5)
            else:
                raise ValueError(
                    f"Cannot convert value '{value}' to binary alarm. "
                    "Please normalize the alarm column to 0/1."
                )

    return out


def _ordered_case_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe ordered as one online sequence."""
    out = df.copy()

    if "segment_id" not in out.columns:
        out["segment_id"] = "dataset3_online"

    if "row_index" in out.columns:
        plot_order = pd.to_numeric(out["row_index"], errors="coerce")
    else:
        plot_order = pd.Series(
            np.arange(len(out), dtype=int),
            index=out.index,
        )

    fallback_order = pd.Series(
        np.arange(len(out), dtype=int),
        index=out.index,
    )

    out["_plot_order"] = (
        plot_order.where(plot_order.notna(), fallback_order)
        .astype(int)
    )

    return (
        out.sort_values(["segment_id", "_plot_order"], kind="mergesort")
        .reset_index(drop=True)
        .copy()
    )


def _make_time_axis(df: pd.DataFrame) -> np.ndarray:
    """
    Build causal online time axis.

    If delta_t is available, use cumulative time in seconds.
    Otherwise, use row position.
    """
    if "delta_t" not in df.columns:
        return np.arange(len(df), dtype=float)

    dt = _finite_numeric(df["delta_t"].to_numpy(), default=1.0)
    dt = np.where(dt > 0.0, dt, 1.0)

    x = np.zeros(len(df), dtype=float)

    for i in range(1, len(df)):
        x[i] = x[i - 1] + float(dt[i])

    return x


def _attack_spans(labels: Sequence[Any]) -> List[Tuple[int, int]]:
    """Extract contiguous attack spans over row positions."""
    y = _to_binary_array(labels)
    spans: List[Tuple[int, int]] = []

    i = 0
    while i < len(y):
        if y[i] != 1:
            i += 1
            continue

        start = i
        while i + 1 < len(y) and y[i + 1] == 1:
            i += 1
        end = i

        spans.append((int(start), int(end)))
        i += 1

    return spans


def _span_end_x(x: np.ndarray, start: int, end: int) -> float:
    """Return a visually correct x-end for shaded intervals."""
    if end + 1 < len(x):
        return float(x[end + 1])

    if end > start:
        return float(x[end] + (x[end] - x[end - 1]))

    return float(x[end] + 1.0)


def _shade_attack_intervals(
    axes: Sequence[plt.Axes],
    spans: Sequence[Tuple[int, int]],
    x: np.ndarray,
) -> None:
    """Shade ground-truth attack intervals on all panels."""
    for ax in axes:
        for start, end in spans:
            ax.axvspan(
                float(x[start]),
                _span_end_x(x, start, end),
                facecolor="0.90",
                edgecolor="none",
                zorder=0,
            )


def _format_axis(ax: plt.Axes) -> None:
    """Apply IEEE-style axis formatting."""
    ax.grid(True, axis="y", color="0.86", linewidth=0.5)
    ax.grid(True, axis="x", color="0.92", linewidth=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, width=0.7, length=3)


def _format_binary_axis(ax: plt.Axes) -> None:
    """Format binary-alarm axes."""
    ax.set_ylim(-0.12, 1.18)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"])
    _format_axis(ax)


def _plot_binary_panel(
    ax: plt.Axes,
    x: np.ndarray,
    values: Sequence[Any],
    ylabel: str,
) -> None:
    """Plot a binary sequence panel."""
    y = _to_binary_array(values)
    ax.step(
        x,
        y,
        where="post",
        color="black",
        linewidth=1.05,
        zorder=2,
    )
    ax.set_ylabel(ylabel)
    _format_binary_axis(ax)


def _plot_numeric_panel(
    ax: plt.Axes,
    x: np.ndarray,
    values: Sequence[Any],
    ylabel: str,
) -> None:
    """Plot a numeric sequence panel."""
    y = _finite_numeric(values)
    ax.plot(
        x,
        y,
        color="black",
        linewidth=1.0,
        zorder=2,
    )
    ax.set_ylabel(ylabel)
    _format_axis(ax)


def _prepare_residual_energy_for_display(
    values: Sequence[Any],
    column_name: str,
) -> Tuple[np.ndarray, str]:
    """
    Prepare residual energy for paper plotting.

    Raw xi_q can have very large physical units, so use log10(1 + q_t).
    Scaled xi_q_scaled is plotted directly.
    """
    y = _finite_numeric(values)

    if "scaled" in column_name.lower():
        return y, r"Scaled $q_t$"

    y = np.maximum(y, 0.0)
    y = np.log10(1.0 + y)

    return y, r"$\log_{10}(1+q_t)$"


def _prepare_accumulated_evidence_for_display(
    values: Sequence[Any],
    column_name: str,
) -> Tuple[np.ndarray, str]:
    """
    Prepare accumulated evidence for paper plotting.

    xi_accum_log is already the log accumulated evidence feature.
    """
    y = _finite_numeric(values)

    if "scaled" in column_name.lower():
        return y, r"Scaled $\tilde{a}_t$"

    return y, r"Accum. evidence $\tilde{a}_t$"


def _first_alarm_positions_inside_spans(
    labels: Sequence[Any],
    alarm_values: Sequence[Any],
    spans: Sequence[Tuple[int, int]],
) -> List[Optional[int]]:
    """Return first alarm position inside each attack interval."""
    _ = _to_binary_array(labels)
    alarm = _to_binary_array(alarm_values)

    first_positions: List[Optional[int]] = []

    for start, end in spans:
        offsets = np.where(alarm[start : end + 1] == 1)[0]

        if len(offsets) == 0:
            first_positions.append(None)
            continue

        first_positions.append(int(start + int(offsets[0])))

    return first_positions


def _annotate_attack_names(
    ax: plt.Axes,
    spans: Sequence[Tuple[int, int]],
    x: np.ndarray,
) -> None:
    """Add Attack-1 / Attack-2 text on the top panel."""
    for idx, (start, end) in enumerate(spans, start=1):
        center = 0.5 * (float(x[start]) + _span_end_x(x, start, end))
        ax.text(
            center,
            1.08,
            f"Attack {idx}",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )


def _annotate_alarm_delays(
    ax: plt.Axes,
    x: np.ndarray,
    labels: Sequence[Any],
    alarms: Sequence[Any],
    spans: Sequence[Tuple[int, int]],
    prefix: str,
) -> None:
    """
    Mark first alarm inside each attack and annotate causal delay.

    This makes the figure self-explaining without changing metrics.
    """
    first_positions = _first_alarm_positions_inside_spans(
        labels=labels,
        alarm_values=alarms,
        spans=spans,
    )

    for attack_idx, ((start, _end), first_pos) in enumerate(
        zip(spans, first_positions),
        start=1,
    ):
        if first_pos is None:
            continue

        delay = float(x[first_pos] - x[start])

        ax.axvline(
            float(x[first_pos]),
            color="0.25",
            linestyle=":",
            linewidth=0.8,
            zorder=1,
        )
        ax.plot(
            float(x[first_pos]),
            1.07,
            marker="v",
            markersize=3.8,
            color="black",
            clip_on=False,
            zorder=3,
        )
        ax.text(
            float(x[first_pos]),
            1.12,
            f"{prefix}{attack_idx}: {delay:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7,
            clip_on=False,
        )


def _write_missing_panel(
    ax: plt.Axes,
    ylabel: str,
    message: str,
) -> None:
    """Write placeholder text if a diagnostic column is missing."""
    ax.set_ylabel(ylabel)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    _format_axis(ax)


def plot_dataset3_case_study(
    df: pd.DataFrame,
    output_dir: str | Path,
    # title: str = "Dataset-3 online case study",
    title: str = "Dataset-3 causal sequential case study",
    probability_column: str = "probability",
    proposed_alarm_column: str = "confirmed_alarm",
    ekf_alarm_column: str = "ekf_alarm",
    label_column: str = "label",
    residual_energy_candidates: Optional[Sequence[str]] = None,
    accumulated_evidence_candidates: Optional[Sequence[str]] = None,
    theta: Optional[float] = None,
    save_png: bool = True,
    save_pdf: bool = True,
    dpi: int = 600,
) -> Dict[str, Any]:
    """
    Create IEEE-style Dataset-3 online case-study figure.

    Required dataframe columns:
    - label
    - probability
    - confirmed_alarm
    - ekf_alarm

    Optional dataframe columns:
    - xi_q or xi_q_scaled
    - xi_accum_log or xi_accum_log_scaled
    """
    if residual_energy_candidates is None:
        residual_energy_candidates = ["xi_q", "xi_q_scaled"]

    if accumulated_evidence_candidates is None:
        accumulated_evidence_candidates = ["xi_accum_log", "xi_accum_log_scaled"]

    required = [
        label_column,
        probability_column,
        proposed_alarm_column,
        ekf_alarm_column,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset-3 case-study dataframe missing columns: {missing}")

    ordered = _ordered_case_frame(df)
    x = _make_time_axis(ordered)

    attack_intervals = _attack_spans(ordered[label_column].to_numpy())

    residual_energy_column = _pick_first_existing_column(
        ordered,
        residual_energy_candidates,
    )
    accumulated_evidence_column = _pick_first_existing_column(
        ordered,
        accumulated_evidence_candidates,
    )

    rc_params = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
    }

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            nrows=6,
            ncols=1,
            figsize=(7.2, 7.4),
            sharex=True,
            constrained_layout=False,
        )

        _shade_attack_intervals(axes, attack_intervals, x)

        _plot_binary_panel(
            ax=axes[0],
            x=x,
            values=ordered[label_column].to_numpy(),
            ylabel="Ground truth\n$y_t$",
        )
        _annotate_attack_names(axes[0], attack_intervals, x)

        _plot_binary_panel(
            ax=axes[1],
            x=x,
            values=ordered[ekf_alarm_column].to_numpy(),
            ylabel="EKF alarm",
        )
        _annotate_alarm_delays(
            ax=axes[1],
            x=x,
            labels=ordered[label_column].to_numpy(),
            alarms=ordered[ekf_alarm_column].to_numpy(),
            spans=attack_intervals,
            prefix="E",
        )

        probability = _finite_numeric(ordered[probability_column].to_numpy())
        axes[2].plot(
            x,
            probability,
            color="black",
            linewidth=1.0,
            zorder=2,
        )

        if theta is not None:
            theta_value = float(theta)
            axes[2].axhline(
                theta_value,
                color="0.25",
                linestyle="--",
                linewidth=0.8,
                zorder=1,
            )
            axes[2].text(
                float(x[-1]),
                min(theta_value + 0.06, 0.96),
                rf"$\theta={theta_value:.2f}$",
                ha="right",
                va="bottom",
                fontsize=8,
            )

        axes[2].set_ylabel("Proposed\n$\\hat{p}_t$")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_yticks([0.0, 0.5, 1.0])
        _format_axis(axes[2])

        _plot_binary_panel(
            ax=axes[3],
            x=x,
            values=ordered[proposed_alarm_column].to_numpy(),
            ylabel="Proposed alarm\n$\\bar{c}_t$",
        )
        _annotate_alarm_delays(
            ax=axes[3],
            x=x,
            labels=ordered[label_column].to_numpy(),
            alarms=ordered[proposed_alarm_column].to_numpy(),
            spans=attack_intervals,
            prefix="P",
        )

        if residual_energy_column is not None:
            residual_y, residual_ylabel = _prepare_residual_energy_for_display(
                values=ordered[residual_energy_column].to_numpy(),
                column_name=residual_energy_column,
            )
            axes[4].plot(
                x,
                residual_y,
                color="black",
                linewidth=0.95,
                zorder=2,
            )
            axes[4].set_ylabel(residual_ylabel)
            _format_axis(axes[4])
        else:
            _write_missing_panel(
                ax=axes[4],
                ylabel=r"$q_t$",
                message="Residual-energy column not found",
            )

        if accumulated_evidence_column is not None:
            accum_y, accum_ylabel = _prepare_accumulated_evidence_for_display(
                values=ordered[accumulated_evidence_column].to_numpy(),
                column_name=accumulated_evidence_column,
            )
            axes[5].plot(
                x,
                accum_y,
                color="black",
                linewidth=0.95,
                zorder=2,
            )
            axes[5].set_ylabel(accum_ylabel)
            _format_axis(axes[5])
        else:
            _write_missing_panel(
                ax=axes[5],
                ylabel=r"$\tilde{a}_t$",
                message="Accumulated-evidence column not found",
            )

        # axes[-1].set_xlabel("Online time (s)")
        axes[-1].set_xlabel("Elapsed time (s)")

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        fig.suptitle(title, y=0.992, fontsize=9)
        fig.align_ylabels(axes)
        fig.subplots_adjust(
            left=0.115,
            right=0.995,
            top=0.955,
            bottom=0.075,
            hspace=0.22,
        )

        output_dir = Path(output_dir)
        _ensure_dir(output_dir)

        output_paths: Dict[str, Optional[str]] = {
            "png_path": None,
            "pdf_path": None,
        }

        def _safe_save_figure(
                fig: plt.Figure,
                target_path: Path,
                dpi: Optional[int] = None,
        ) -> Path:
            """
            Save figure safely.

            If Windows blocks overwriting an open PDF/PNG, save with a new suffix
            instead of failing the whole Step 19 run.
            """
            candidates = [
                target_path,
                target_path.with_name(f"{target_path.stem}_v2{target_path.suffix}"),
                target_path.with_name(f"{target_path.stem}_v3{target_path.suffix}"),
                target_path.with_name(f"{target_path.stem}_paper{target_path.suffix}"),
            ]

            last_error: Optional[Exception] = None

            for candidate in candidates:
                try:
                    if dpi is None:
                        fig.savefig(candidate, bbox_inches="tight")
                    else:
                        fig.savefig(candidate, dpi=dpi, bbox_inches="tight")
                    return candidate
                except PermissionError as exc:
                    last_error = exc
                    continue

            raise PermissionError(
                f"Could not save figure. Close any open PDF/PNG viewer for: {target_path}"
            ) from last_error

        if save_png:
            png_path = _safe_save_figure(
                fig=fig,
                target_path=output_dir / "dataset3_case_study.png",
                dpi=dpi,
            )
            output_paths["png_path"] = str(png_path)

        if save_pdf:
            pdf_path = _safe_save_figure(
                fig=fig,
                target_path=output_dir / "dataset3_case_study.pdf",
                dpi=None,
            )
            output_paths["pdf_path"] = str(pdf_path)

    return {
        "status": "PASSED",
        "figure_dir": str(output_dir),
        "rows_used": int(len(ordered)),
        "attack_interval_count": int(len(attack_intervals)),
        "residual_energy_column": residual_energy_column,
        "accumulated_evidence_column": accumulated_evidence_column,
        **output_paths,
    }


__all__ = [
    "plot_dataset3_case_study",
]