"""
Synthetic validation runner for the Step-2 unified event engine.

Run from the project root:

    python -m src.experiments.run_unified_evaluator_validation

No YAML changes and no model checkpoints are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.evaluation.unified_alarm_rules import build_alarm_sequence
from src.evaluation.unified_evaluator import (
    UnifiedPredictionBundle,
    evaluate_prediction_bundle,
)
from src.evaluation.unified_operating_point import (
    DEFAULT_PERSISTENCE_GRID,
    DEFAULT_THRESHOLD_GRID,
    select_operating_point,
)


def _bundle(
    probabilities,
    labels,
    valid_mask=None,
    segment_ids=None,
    delta_t=None,
    split_name="validation",
    model_name="synthetic",
) -> UnifiedPredictionBundle:
    n = len(probabilities)
    return UnifiedPredictionBundle(
        split_name=split_name,
        model_name=model_name,
        probabilities=np.asarray(probabilities, dtype=float),
        labels=np.asarray(labels, dtype=int),
        valid_mask=np.ones(n, dtype=int)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=int),
        segment_ids=np.asarray(["s1"] * n, dtype=object)
        if segment_ids is None
        else np.asarray(segment_ids, dtype=object),
        delta_t=np.ones(n, dtype=float)
        if delta_t is None
        else np.asarray(delta_t, dtype=float),
        row_indices=np.arange(n, dtype=int),
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_synthetic_validation() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    # 1. One attack with one alarm.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.1, 0.1, 0.9, 0.9, 0.9, 0.1],
            labels=[0, 0, 1, 1, 1, 0],
        ),
        theta=0.5,
        persistence=2,
    )
    _check(result.event_metrics.attack_events_total == 1, "Expected one attack.")
    _check(result.event_metrics.attack_events_detected == 1, "Attack should be detected.")
    _check(result.attack_detections[0].delay_rows == 1, "Expected one-row delay.")
    results.append({"name": "one_attack_one_alarm", "status": "PASSED"})

    # 2. Missed attack.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.1, 0.2, 0.3, 0.4],
            labels=[0, 1, 1, 0],
        ),
        theta=0.8,
        persistence=1,
    )
    _check(result.event_metrics.attack_events_missed == 1, "Attack should be missed.")
    _check(result.attack_detections[0].delay_seconds is None, "Miss delay must be None.")
    results.append({"name": "missed_attack", "status": "PASSED"})

    # 3. Invalid row inside an attack does not split truth event.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.1, 0.9, 0.9, 0.9, 0.9, 0.1],
            labels=[0, 1, 1, 1, 1, 0],
            valid_mask=[1, 1, 0, 1, 1, 1],
        ),
        theta=0.5,
        persistence=2,
    )
    _check(len(result.attack_events) == 1, "Invalid row split the attack event.")
    _check(result.attack_events[0].duration_rows == 4, "Attack duration should include invalid row.")
    _check(result.attack_detections[0].first_eligible_alarm_onset_position == 4, "Wrong post-invalid onset.")
    results.append({"name": "invalid_row_inside_attack", "status": "PASSED"})

    # 4. Alarm active before attack does not receive zero-delay credit;
    #    later new onset inside the same attack is eligible.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.9, 0.9, 0.1, 0.9, 0.9],
            labels=[0, 1, 1, 1, 0],
        ),
        theta=0.5,
        persistence=1,
    )
    detection = result.attack_detections[0]
    _check(detection.alarm_active_before_attack, "Pre-existing alarm was not recorded.")
    _check(detection.detected, "Later new onset should detect the attack.")
    _check(detection.first_eligible_alarm_onset_position == 3, "Wrong eligible onset.")
    _check(detection.delay_rows == 2, "Pre-existing alarm incorrectly got zero delay.")
    results.append({"name": "preexisting_alarm_no_zero_credit", "status": "PASSED"})

    # 5. Alarm crossing normal-to-attack boundary without re-onset is a miss.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.9, 0.9, 0.9, 0.9],
            labels=[0, 1, 1, 0],
        ),
        theta=0.5,
        persistence=1,
    )
    _check(result.attack_detections[0].alarm_active_before_attack, "Crossing alarm not marked.")
    _check(not result.attack_detections[0].detected, "Continuing alarm must not count as new detection.")
    results.append({"name": "alarm_crossing_boundary", "status": "PASSED"})

    # 6. Multiple attacks with the first missed; event numbering must stay intact.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
            labels=[0, 1, 1, 0, 1, 0],
        ),
        theta=0.5,
        persistence=1,
    )
    _check(len(result.attack_detections) == 2, "Expected two attack records.")
    _check(result.attack_detections[0].event_id == "attack_0001", "First ID changed.")
    _check(not result.attack_detections[0].detected, "First attack should be missed.")
    _check(result.attack_detections[0].delay_rows is None, "First miss needs None delay.")
    _check(result.attack_detections[1].event_id == "attack_0002", "Second ID changed.")
    _check(result.attack_detections[1].detected, "Second attack should be detected.")
    results.append({"name": "multiple_attacks_first_missed", "status": "PASSED"})

    # 7. Segment boundary resets persistence.
    alarm = build_alarm_sequence(
        probabilities=[0.9, 0.9],
        segment_ids=["a", "b"],
        valid_mask=[1, 1],
        theta=0.5,
        persistence=2,
    )
    _check(alarm.confirmed_alarm.tolist() == [0, 0], "Persistence crossed segment boundary.")
    results.append({"name": "segment_boundary_reset", "status": "PASSED"})

    # 8. Irregular delta_t delay.
    result = evaluate_prediction_bundle(
        _bundle(
            probabilities=[0.1, 0.1, 0.1, 0.9],
            labels=[0, 1, 1, 1],
            delta_t=[0.0, 1.0, 2.5, 4.0],
        ),
        theta=0.5,
        persistence=1,
    )
    _check(abs(result.attack_detections[0].delay_seconds - 6.5) < 1e-12, "Irregular delay is wrong.")
    results.append({"name": "irregular_delta_t", "status": "PASSED"})

    # 9. Invalid observation breaks persistence.
    alarm = build_alarm_sequence(
        probabilities=[0.9, 0.9, 0.9, 0.9],
        segment_ids=["s1"] * 4,
        valid_mask=[1, 0, 1, 1],
        theta=0.5,
        persistence=2,
    )
    _check(alarm.confirmed_alarm.tolist() == [0, 0, 0, 1], "Invalid row did not break persistence.")
    results.append({"name": "persistence_broken_by_invalid", "status": "PASSED"})

    # 10. Locked validation grid and validation-only guard.
    selection_bundle = _bundle(
        probabilities=[0.1, 0.2, 0.8, 0.9, 0.1, 0.95],
        labels=[0, 0, 1, 1, 0, 1],
        split_name="validation",
    )
    selection = select_operating_point(selection_bundle, include_candidates=False)
    expected_count = len(DEFAULT_THRESHOLD_GRID) * len(DEFAULT_PERSISTENCE_GRID)
    _check(selection.candidate_count == expected_count == 190, "Locked grid is not 19x10.")

    guard_triggered = False
    try:
        select_operating_point(
            _bundle(
                probabilities=[0.1, 0.9],
                labels=[0, 1],
                split_name="test",
            ),
            include_candidates=False,
        )
    except ValueError:
        guard_triggered = True
    _check(guard_triggered, "Test split was incorrectly allowed for selection.")
    results.append({"name": "operating_point_grid_and_guard", "status": "PASSED"})

    return {
        "status": "PASSED",
        "tests_passed": len(results),
        "tests": results,
        "locked_rules": {
            "attack_events_ignore_validity": True,
            "invalid_rows_break_persistence": True,
            "segment_boundaries_reset_persistence": True,
            "preexisting_alarm_requires_new_onset": True,
            "miss_delay_is_none": True,
            "threshold_grid": list(DEFAULT_THRESHOLD_GRID),
            "persistence_grid": list(DEFAULT_PERSISTENCE_GRID),
            "candidate_order": [
                "higher_f1",
                "higher_attack_detection_rate",
                "lower_mean_detection_delay",
                "lower_fpr",
                "higher_recall",
                "lower_persistence",
                "higher_threshold",
            ],
        },
    }


def main() -> None:
    report = run_synthetic_validation()
    output = Path(
        "results/extended_comparison/unified_event_engine/"
        "step2_synthetic_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)

    print("=" * 100)
    print("STEP 2 UNIFIED EVENT-EVALUATION ENGINE")
    print("=" * 100)
    print(f"Status       : {report['status']}")
    print(f"Tests passed : {report['tests_passed']}")
    print(f"Report       : {output.resolve()}")
    print("=" * 100)


if __name__ == "__main__":
    main()
