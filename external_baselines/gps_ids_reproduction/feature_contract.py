"""
Locked GPS-IDS behavior-feature contract.

Scientific scope
----------------
This is a protocol-controlled reimplementation of the published GPS-IDS
behavior-feature representation. It is not claimed to be an exact reproduction
because the paper does not publish executable feature-extraction code or a
complete classifier specification.

The contract follows the feature blocks identified by the GPS-IDS paper:
- Eq. (15): dynamic bicycle / vehicle-model block
- Eq. (16): GPS localization block
- Eq. (17): state-estimation block
- Eq. (18): motion-planning block
- Eq. (20): PID-controller block

Only contemporaneous raw AV-GPS measurements or onboard planner/controller
outputs are included. No future differences, attack labels, EKF decisions,
source identifiers, split identifiers, attack-derived validity flags, or
post-event summaries are model features.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd


GPS_IDS_CONTRACT_VERSION = "gps_ids_behavior_features_v2"

GPS_IDS_PAPER = {
    "title": (
        "GPS-IDS: An Anomaly-based GPS Spoofing Attack Detection "
        "Framework for Autonomous Vehicles"
    ),
    "arxiv_id": "2405.08359",
    "method_name": "GPS-IDS classifier-suite reimplementation",
    "reproduction_status": "protocol-controlled reimplementation_not_exact_reproduction",
}

# Locked canonical model-input order.
GPS_IDS_MODEL_FEATURES: Tuple[str, ...] = (
    # Eq. (15) / Eq. (17): vehicle state and control input.
    "longitudinal_velocity_mps",
    "lateral_velocity_mps",
    "yaw_rate_deg_s",
    "steering_angle_deg",
    # Eq. (16): GPS localization and signal-quality state.
    "gps_latitude_deg",
    "gps_longitude_deg",
    "gps_hdop",
    "gps_vdop",
    "satellite_count",
    "satellite_locks",
    # Eq. (18): motion-planning state.
    "target_yaw_deg",
    "current_yaw_deg",
    "cross_track_error_m",
    # Eq. (20): controller input/output behavior.
    "throttle_percent",
    "speed_mps",
)

# Canonical metadata/target fields written to every output CSV.
GPS_IDS_OUTPUT_METADATA_COLUMNS: Tuple[str, ...] = (
    "segment_id",
    "row_index",
    "within_segment_index",
    "delta_t",
    "split",
    "label",
    "valid_mask",
    "feature_complete_mask",
    "target_yaw_missing",
)

# Explicit aliases are locked here to prevent silent fuzzy matching.
GPS_IDS_RAW_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "longitudinal_velocity_mps": (
        "Longitudinal Velocity (m/s)",
    ),
    "lateral_velocity_mps": (
        "Lateral Velocity (m/s)",
    ),
    "yaw_rate_deg_s": (
        "Yaw Rate (deg/s)",
    ),
    "steering_angle_deg": (
        "Steering Angle (deg)",
    ),
    "gps_latitude_deg": (
        "GPS Latitude",
    ),
    "gps_longitude_deg": (
        "GPS Longitude",
    ),
    "gps_hdop": (
        "GPS HDOP",
    ),
    "gps_vdop": (
        "GPS VDOP",
    ),
    "satellite_count": (
        "Satellite Count",
    ),
    "satellite_locks": (
        "Satellite Locks",
    ),
    "target_yaw_deg": (
        "Heading To Next WP (deg)",
        "Heading To Next WP",
    ),
    "current_yaw_deg": (
        "Yaw (deg)",
        "Yaw",
    ),
    "cross_track_error_m": (
        "X-Track Error (m)",
        "X-Track Error",
    ),
    "throttle_percent": (
        "Throttle (%)",
        "Throttle",
    ),
    "speed_mps": (
        "Velocity (m/s)",
        "Velocity",
    ),
}

# These columns may be present in segmented raw files but must never enter X.
GPS_IDS_EXCLUDED_MODEL_COLUMNS: Tuple[str, ...] = (
    # Canonical exported metadata/target columns.
    "label",
    "valid_mask",
    "feature_complete_mask",
    "target_yaw_missing",
    "split",
    "row_index",
    "delta_t",
    # Raw source columns.
    "Data Type",
    "EKF Detector",
    "source_file",
    "source_key",
    "dataset_role",
    "raw_row_index",
    "row_order_in_source",
    "row_index_original",
    "segment_id",
    "segment_index",
    "within_segment_index",
    "xi_split",
    "xi_source_dataset",
    "valid_transition_prelim",
    "nu_prelim",
    "nu",
    "xi_nu",
    "normal_to_attack_transition",
    "attack_to_normal_transition",
    "is_segment_start",
    "segment_boundary",
    "segment_boundary_reason",
    "Clock Date",
    "Clock Time",
    "Run Time",
    "Hobbs",
    "Travelled Distance (m)",
    "Distance To Home (m)",
    "Distance To GCS (m)",
    "Mission Index",
    "GPS MGRS",
)

FORBIDDEN_XI_INPUT_BASENAMES: Tuple[str, ...] = (
    "train_xi.csv",
    "validation_xi.csv",
    "val_xi.csv",
    "test_xi.csv",
    "external_xi.csv",
    "online_xi.csv",
    "dataset1_xi.csv",
    "dataset2_xi.csv",
    "dataset3_xi.csv",
)


@dataclass(frozen=True)
class FeatureMappingRow:
    published_variable: str
    original_equation_component: str
    raw_av_gps_column: str
    canonical_feature_name: str
    transformation: str
    units: str
    availability: str
    leakage_status: str
    final_inclusion_decision: str
    rationale: str

    def to_machine_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> Dict[str, Any]:
        return {
            "Published GPS-IDS variable": self.published_variable,
            "Original equation/component": self.original_equation_component,
            "Raw AV-GPS column": self.raw_av_gps_column,
            "Canonical feature name": self.canonical_feature_name,
            "Transformation": self.transformation,
            "Units": self.units,
            "Availability": self.availability,
            "Leakage status": self.leakage_status,
            "Final inclusion decision": self.final_inclusion_decision,
            "Rationale": self.rationale,
        }


@dataclass(frozen=True)
class GPSIDSFeatureContract:
    contract_version: str
    source_paper: Dict[str, Any]
    representation_name: str
    comparison_role: str
    source_data_rule: str
    final_model_feature_names: List[str]
    final_model_feature_count: int
    feature_hash: str
    output_metadata_columns: List[str]
    raw_column_aliases: Dict[str, List[str]]
    excluded_model_columns: List[str]
    forbidden_xi_input_basenames: List[str]
    preprocessing_rule: Dict[str, Any]
    leakage_rules: Dict[str, Any]
    mapping_rows: List[FeatureMappingRow]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "source_paper": dict(self.source_paper),
            "representation_name": self.representation_name,
            "comparison_role": self.comparison_role,
            "source_data_rule": self.source_data_rule,
            "final_model_feature_names": list(self.final_model_feature_names),
            "final_model_feature_count": int(self.final_model_feature_count),
            "feature_hash": self.feature_hash,
            "output_metadata_columns": list(self.output_metadata_columns),
            "raw_column_aliases": {
                key: list(value)
                for key, value in self.raw_column_aliases.items()
            },
            "excluded_model_columns": list(self.excluded_model_columns),
            "forbidden_xi_input_basenames": list(
                self.forbidden_xi_input_basenames
            ),
            "preprocessing_rule": dict(self.preprocessing_rule),
            "leakage_rules": dict(self.leakage_rules),
            "mapping_rows": [
                row.to_machine_dict() for row in self.mapping_rows
            ],
        }


def _strict_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _strict_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return _strict_json_value(value.to_dict())
    if hasattr(value, "__dict__"):
        return _strict_json_value(vars(value))
    return str(value)


def save_strict_json(payload: Mapping[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _strict_json_value(dict(payload)),
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    return path


def feature_contract_hash(feature_names: Sequence[str]) -> str:
    ordered = [str(name).strip() for name in feature_names]
    if not ordered or any(not name for name in ordered):
        raise ValueError("Feature names must be nonempty.")
    if len(ordered) != len(set(ordered)):
        raise ValueError("Feature names contain duplicates.")
    encoded = json.dumps(
        {
            "ordered_feature_names": ordered,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_mapping_rows() -> List[FeatureMappingRow]:
    """
    Build the locked literature-to-dataset mapping.

    Eq. (17) variables overlap Eq. (15); they are represented once in the
    final feature matrix rather than duplicated.
    """
    include = "INCLUDE"
    exclude = "EXCLUDE_FROM_MODEL_INPUT"
    metadata_only = "METADATA_OR_TARGET_ONLY"

    rows = [
        FeatureMappingRow(
            "v_x (longitudinal velocity)",
            "Eq. (15), Vehicle Model block",
            "Longitudinal Velocity (m/s)",
            "longitudinal_velocity_mps",
            "numeric coercion only; no clipping/scaling in Step 4",
            "m/s",
            "direct raw column",
            "causal current-row vehicle state; no label use",
            include,
            "The state-space coefficients depend on longitudinal velocity.",
        ),
        FeatureMappingRow(
            "v_y (lateral velocity)",
            "Eq. (15) Vehicle Model and Eq. (17) State Estimation",
            "Lateral Velocity (m/s)",
            "lateral_velocity_mps",
            "numeric coercion only",
            "m/s",
            "direct raw column",
            "causal current-row vehicle state; no label use",
            include,
            "Shared state variable is included once.",
        ),
        FeatureMappingRow(
            "r (yaw rate)",
            "Eq. (15) Vehicle Model and Eq. (17) State Estimation",
            "Yaw Rate (deg/s)",
            "yaw_rate_deg_s",
            "numeric coercion only; preserve dataset unit",
            "degree/s",
            "direct raw column",
            "causal current-row vehicle state; no label use",
            include,
            "Shared state variable is included once.",
        ),
        FeatureMappingRow(
            "delta (steering angle/control input)",
            "Eq. (15), Eq. (17), and Eq. (20)",
            "Steering Angle (deg)",
            "steering_angle_deg",
            "numeric coercion only; preserve dataset unit",
            "degree",
            "direct raw column",
            "causal current-row actuator state; no label use",
            include,
            "Shared control variable is included once.",
        ),
        FeatureMappingRow(
            "latitude",
            "Eq. (16), GPS Localization block",
            "GPS Latitude",
            "gps_latitude_deg",
            "numeric coercion only; retain absolute coordinate",
            "degree",
            "direct raw column",
            "method-intended but geography-sensitive; not label-derived",
            include,
            "Retained for published-method fidelity; interpretation must note location sensitivity.",
        ),
        FeatureMappingRow(
            "longitude",
            "Eq. (16), GPS Localization block",
            "GPS Longitude",
            "gps_longitude_deg",
            "numeric coercion only; retain absolute coordinate",
            "degree",
            "direct raw column",
            "method-intended but geography-sensitive; not label-derived",
            include,
            "Retained for published-method fidelity; interpretation must note location sensitivity.",
        ),
        FeatureMappingRow(
            "DOP (horizontal component)",
            "Eq. (16), GPS Localization block",
            "GPS HDOP",
            "gps_hdop",
            "numeric coercion only",
            "unitless",
            "direct raw column",
            "causal receiver-quality measurement; not label-derived",
            include,
            "The dataset provides horizontal and vertical DOP separately.",
        ),
        FeatureMappingRow(
            "DOP (vertical component)",
            "Eq. (16), GPS Localization block",
            "GPS VDOP",
            "gps_vdop",
            "numeric coercion only",
            "unitless",
            "direct raw column",
            "causal receiver-quality measurement; not label-derived",
            include,
            "The dataset provides horizontal and vertical DOP separately.",
        ),
        FeatureMappingRow(
            "number of available satellites",
            "Eq. (16), GPS Localization block",
            "Satellite Count",
            "satellite_count",
            "numeric coercion only",
            "count",
            "direct raw column",
            "causal receiver state; not label-derived",
            include,
            "Direct mapping to available/observed satellite count.",
        ),
        FeatureMappingRow(
            "number of locked satellites",
            "Eq. (16), GPS Localization block",
            "Satellite Locks",
            "satellite_locks",
            "numeric coercion only",
            "count",
            "direct raw column",
            "causal receiver state; not label-derived",
            include,
            "Direct mapping to locked satellites.",
        ),
        FeatureMappingRow(
            "psi_target (target yaw)",
            "Eq. (18), Motion Planning block",
            "Heading To Next WP (deg)",
            "target_yaw_deg",
            "numeric coercion only; use contemporaneous onboard planner output",
            "degree",
            "direct raw column",
            "causal current-row planner output; no future row construction",
            include,
            "No target coordinate reconstruction or look-ahead is performed.",
        ),
        FeatureMappingRow(
            "psi (current yaw)",
            "Eq. (18), Motion Planning block",
            "Yaw (deg)",
            "current_yaw_deg",
            "numeric coercion only",
            "degree",
            "direct raw column",
            "causal current-row attitude measurement",
            include,
            "Direct current-yaw measurement.",
        ),
        FeatureMappingRow(
            "e_ct (cross-track error)",
            "Eq. (18), Motion Planning block",
            "X-Track Error (m)",
            "cross_track_error_m",
            "numeric coercion only; use contemporaneous onboard value",
            "m",
            "direct raw column",
            "causal current-row planner state; downstream attack effect but not label-derived",
            include,
            "Legitimate method-specific behavior feature, although forbidden to the proposed xi-only branch.",
        ),
        FeatureMappingRow(
            "a_control / throttle command",
            "Eq. (20), PID Controller block",
            "Throttle (%)",
            "throttle_percent",
            "numeric coercion only",
            "%",
            "direct raw actuator column",
            "causal current-row controller output; not label-derived",
            include,
            "Throttle is the available direct actuator-command representation.",
        ),
        FeatureMappingRow(
            "controlled/measured vehicle speed",
            "Eq. (20) controller behavior; experimental controller output",
            "Velocity (m/s)",
            "speed_mps",
            "numeric coercion only",
            "m/s",
            "direct raw column",
            "causal current-row controlled-speed response; not label-derived",
            include,
            "Included as the measured controlled-speed response used in the GPS-IDS behavior analysis.",
        ),
        # Explicit exclusions required by the protocol.
        FeatureMappingRow(
            "attack class",
            "Supervised target only",
            "Data Type",
            "",
            "rename to label; never enter feature matrix X",
            "0/1",
            "direct raw column",
            "ground-truth label; direct target leakage if used as feature",
            metadata_only,
            "Retained only as y for training/evaluation.",
        ),
        FeatureMappingRow(
            "EKF detector decision",
            "Comparator output, not a GPS-IDS behavior variable",
            "EKF Detector",
            "",
            "not copied to GPS-IDS feature files",
            "binary/score",
            "Dataset 3 only",
            "baseline-output leakage",
            exclude,
            "May be evaluated separately but must never be a classifier input.",
        ),
        FeatureMappingRow(
            "source filename/source key",
            "Dataset provenance metadata",
            "source_file / source_key",
            "",
            "not copied to feature files",
            "text",
            "segmented metadata",
            "source identity may reveal collection scenario/class",
            exclude,
            "Excluded from both X and exported row-level files.",
        ),
        FeatureMappingRow(
            "row identifier",
            "Identity metadata",
            "raw_row_index / row_order_in_source / row_index_original",
            "",
            "map one verified identifier to canonical row_index metadata",
            "index",
            "segmented metadata",
            "identifier leakage if used in X",
            metadata_only,
            "Required for alignment only; excluded from model features.",
        ),
        FeatureMappingRow(
            "split identifier",
            "Experimental protocol metadata",
            "xi_split / dataset_role",
            "",
            "create canonical split metadata after segment assignment",
            "text",
            "derived protocol metadata",
            "split leakage if used in X",
            metadata_only,
            "Required for routing only; excluded from model features.",
        ),
        FeatureMappingRow(
            "attack-derived validity/transition state",
            "Not part of published behavior model",
            "nu_prelim / attack_to_normal_transition / normal_to_attack_transition",
            "",
            "not copied or used",
            "binary",
            "segmented metadata",
            "label-derived leakage risk",
            exclude,
            "GPS-IDS valid_mask is based only on row/segment/time integrity; feature completeness is exported separately.",
        ),
        FeatureMappingRow(
            "feature completeness indicator",
            "Reimplementation metadata, not a published classifier variable",
            "derived from finiteness of the 15 locked current-row features",
            "",
            "export as feature_complete_mask; never enter feature matrix X",
            "binary",
            "derived row-local metadata",
            "missingness metadata; not label-derived",
            metadata_only,
            "Separates feature availability from event-evaluation validity.",
        ),
        FeatureMappingRow(
            "target-yaw missingness indicator",
            "Reimplementation metadata, not a published classifier variable",
            "Heading To Next WP (deg)",
            "",
            "export as target_yaw_missing; never enter feature matrix X",
            "binary",
            "derived row-local metadata",
            "missingness metadata; not label-derived",
            metadata_only,
            "Allows transparent auditing and train-only imputation in the classifier step.",
        ),
        FeatureMappingRow(
            "future-derived temporal summaries",
            "Not used in Step 4",
            "none",
            "",
            "no centered differences, future windows, or post-event aggregation",
            "n/a",
            "not constructed",
            "future leakage",
            exclude,
            "All Step-4 transformations are row-local.",
        ),
        FeatureMappingRow(
            "post-event/cumulative navigation summaries",
            "Not required by Eqs. (15), (16), (17), (18), or (20)",
            "Travelled Distance / Distance To Home / Distance To GCS / Mission Index",
            "",
            "not copied or used",
            "mixed",
            "direct raw columns",
            "may encode session progress, mission identity, or attack duration",
            exclude,
            "Excluded to avoid post-event/session shortcut learning.",
        ),
        FeatureMappingRow(
            "encoded location string",
            "Not required by Eq. (16)",
            "GPS MGRS",
            "",
            "not copied or used",
            "text",
            "direct raw column",
            "redundant high-cardinality location identifier",
            exclude,
            "Latitude and longitude already provide the method-intended position variables.",
        ),
    ]
    return rows


def validate_contract(contract: GPSIDSFeatureContract) -> None:
    names = list(contract.final_model_feature_names)
    if names != list(GPS_IDS_MODEL_FEATURES):
        raise AssertionError("Contract feature order differs from locked order.")
    if len(names) != len(set(names)):
        raise AssertionError("Contract feature names contain duplicates.")
    if contract.feature_hash != feature_contract_hash(names):
        raise AssertionError("Contract feature hash is inconsistent.")

    included = [
        row.canonical_feature_name
        for row in contract.mapping_rows
        if row.final_inclusion_decision == "INCLUDE"
    ]
    if included != names:
        raise AssertionError(
            "Included mapping rows do not exactly match the locked feature order.\n"
            f"mapping={included}\nlocked={names}"
        )

    missing_aliases = [
        name for name in names if name not in contract.raw_column_aliases
    ]
    if missing_aliases:
        raise AssertionError(
            f"Missing raw aliases for features: {missing_aliases}"
        )


def build_gps_ids_feature_contract() -> GPSIDSFeatureContract:
    contract = GPSIDSFeatureContract(
        contract_version=GPS_IDS_CONTRACT_VERSION,
        source_paper=dict(GPS_IDS_PAPER),
        representation_name="GPS-IDS intended vehicle-behavior features",
        comparison_role=(
            "direct published-method baseline under the current "
            "segment-level protocol"
        ),
        source_data_rule=(
            "Use segmented raw AV-GPS rows before proposed-model "
            "shortcut-column pruning; never use xi CSV files."
        ),
        final_model_feature_names=list(GPS_IDS_MODEL_FEATURES),
        final_model_feature_count=len(GPS_IDS_MODEL_FEATURES),
        feature_hash=feature_contract_hash(GPS_IDS_MODEL_FEATURES),
        output_metadata_columns=list(GPS_IDS_OUTPUT_METADATA_COLUMNS),
        raw_column_aliases={
            key: list(value)
            for key, value in GPS_IDS_RAW_COLUMN_ALIASES.items()
        },
        excluded_model_columns=list(GPS_IDS_EXCLUDED_MODEL_COLUMNS),
        forbidden_xi_input_basenames=list(
            FORBIDDEN_XI_INPUT_BASENAMES
        ),
        preprocessing_rule={
            "step4_operations": [
                "strict raw-column alias resolution",
                "numeric coercion",
                "canonical renaming",
                "segment-level Dataset-1 split assignment",
                "row/segment/time-integrity valid_mask construction",
                "separate feature_complete_mask construction",
                "separate target_yaw_missing construction",
            ],
            "no_step4_imputation": True,
            "no_step4_scaling": True,
            "no_step4_clipping": True,
            "no_future_rows": True,
            "no_row_dropping": True,
            "classifier_preprocessing_fit_scope": (
                "Dataset-1 training split only in the later classifier step"
            ),
        },
        leakage_rules={
            "label_is_target_only": True,
            "ekf_detector_excluded": True,
            "source_identity_excluded": True,
            "row_identity_metadata_only": True,
            "split_identity_metadata_only": True,
            "attack_derived_validity_excluded": True,
            "future_observations_excluded": True,
            "post_event_summaries_excluded": True,
            "absolute_lat_lon_retained_for_method_fidelity": True,
            "absolute_lat_lon_location_sensitivity_must_be_reported": True,
            "feature_missingness_does_not_change_evaluation_validity": True,
            "feature_complete_mask_metadata_only": True,
            "target_yaw_missing_metadata_only": True,
        },
        mapping_rows=build_mapping_rows(),
    )
    validate_contract(contract)
    return contract


def save_gps_ids_feature_contract(
    contract: GPSIDSFeatureContract,
    contract_json_path: Path,
    mapping_csv_path: Path,
) -> Dict[str, str]:
    validate_contract(contract)

    contract_json_path = save_strict_json(
        contract.to_dict(),
        Path(contract_json_path),
    )

    mapping_csv_path = Path(mapping_csv_path)
    mapping_csv_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_frame = pd.DataFrame(
        [row.to_csv_dict() for row in contract.mapping_rows]
    )
    mapping_frame.to_csv(mapping_csv_path, index=False)

    return {
        "contract_json_path": str(contract_json_path.resolve()),
        "mapping_csv_path": str(mapping_csv_path.resolve()),
    }


__all__ = [
    "FORBIDDEN_XI_INPUT_BASENAMES",
    "GPS_IDS_CONTRACT_VERSION",
    "GPS_IDS_EXCLUDED_MODEL_COLUMNS",
    "GPS_IDS_MODEL_FEATURES",
    "GPS_IDS_OUTPUT_METADATA_COLUMNS",
    "GPS_IDS_RAW_COLUMN_ALIASES",
    "FeatureMappingRow",
    "GPSIDSFeatureContract",
    "build_gps_ids_feature_contract",
    "feature_contract_hash",
    "save_gps_ids_feature_contract",
    "save_strict_json",
    "validate_contract",
]
