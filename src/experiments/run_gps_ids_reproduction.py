"""
Project-facing CLI wrapper for Step 5 GPS-IDS classifier reproduction.

Run from the project root:

    python -m src.experiments.run_gps_ids_reproduction \
        --config-dir configs \
        --gps-ids-features-config gps_ids_features.yaml \
        --gps-ids-classifiers-config gps_ids_classifiers.yaml

The wrapper deliberately loads only dataset.yaml plus the two isolated GPS-IDS
configuration files. It does not add either GPS-IDS YAML to the project's
DEFAULT_CONFIG_ORDER.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from external_baselines.gps_ids_reproduction.runner import (
    run_gps_ids_classifier_suite,
)
from src.utils.config import load_project_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete seven-classifier GPS-IDS protocol-controlled "
            "reimplementation."
        )
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument(
        "--gps-ids-features-config",
        default="gps_ids_features.yaml",
    )
    parser.add_argument(
        "--gps-ids-classifiers-config",
        default="gps_ids_classifiers.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Optional subset for debugging. Omit this argument for the "
            "official complete seven-classifier run."
        ),
    )
    parser.add_argument(
        "--search-profile",
        choices=["standard", "smoke"],
        default=None,
        help=(
            "Use standard for official experiments. Smoke is only for "
            "installation validation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the isolated run directory named by run_id. "
            "Legacy project results are never touched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_dir=args.config_dir,
        config_files=[
            "dataset.yaml",
            str(args.gps_ids_features_config),
            str(args.gps_ids_classifiers_config),
        ],
        allow_missing_optional=False,
    )
    run_gps_ids_classifier_suite(
        config=config,
        model_keys_override=args.models,
        search_profile_override=args.search_profile,
        overwrite_override=(True if args.overwrite else None),
    )


if __name__ == "__main__":
    main()
