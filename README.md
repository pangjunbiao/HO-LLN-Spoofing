# HO-LLN-Spoofing

Research implementation of a high-order liquid detector for continuous and subtle GNSS spoofing detection in autonomous driving.

## Overview

This repository implements a causal-evidence-based spoofing detection pipeline with:

- source-aware trajectory segmentation and data splitting;
- corrected causal-evidence construction;
- high-order liquid temporal modeling;
- validation-selected alarm threshold and persistence;
- row-level and event-level evaluation;
- common-input baselines, ablations, sensitivity analysis, and multi-seed experiments;
- a protocol-controlled GPS-IDS classifier-suite reimplementation.

The repository is under active research development. Raw datasets, generated features, trained checkpoints, predictions, logs, and result artifacts are intentionally excluded from version control.

## Repository Structure

```text
HO-LLN-Spoofing/
├── configs/                    # Dataset, model, training, and experiment settings
├── docs/
│   ├── DATASETS.md             # Dataset acquisition, placement, and roles
│   ├── EXECUTION.md            # Complete execution modes and commands
│   └── REPRODUCIBILITY.md      # Experimental safeguards and provenance
├── external_baselines/
│   └── gps_ids_reproduction/  # GPS-IDS feature and classifier reimplementation
├── src/
│   ├── baselines/              # Common-input baseline models
│   ├── data/                   # Loading, inspection, segmentation, and splitting
│   ├── diagnostics/            # Feature, state, conductance, and occlusion analyses
│   ├── evaluation/             # Metrics, alarm rules, and operating-point selection
│   ├── experiments/            # Experiment entry points
│   ├── extended_preprocessing/ # Evidence-integrity checks
│   ├── models/                 # Proposed high-order liquid architecture
│   ├── preprocessing/          # Motion, residual, and causal-evidence construction
│   ├── training/               # Optimization and training utilities
│   ├── utils/                  # Configuration, device, I/O, logging, and seeding
│   └── visualization/          # Plotting utilities
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/pangjunbiao/HO-LLN-Spoofing.git
cd HO-LLN-Spoofing

python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Datasets

The AV-GPS datasets are not redistributed with this repository. Place locally obtained files under `data/raw/`.

See [docs/DATASETS.md](docs/DATASETS.md) for the expected filenames, dataset roles, split policy, and data-handling instructions.

## Quick Start

Run the environment and configuration check:

```bash
python main.py --mode step1
```

Run the full proposed-model experiment after completing the required preprocessing and training steps:

```bash
python main.py --mode step13
```

The numbered pipeline must be executed in dependency order. See [docs/EXECUTION.md](docs/EXECUTION.md) for all supported modes, commands, and prerequisites.

## GPS-IDS External Comparison

The GPS-IDS branch is a protocol-controlled reimplementation rather than an exact reproduction. Run it from the repository root with:

```bash
python -m src.experiments.run_gps_ids_reproduction --config-dir configs --gps-ids-features-config gps_ids_features.yaml --gps-ids-classifiers-config gps_ids_classifiers.yaml
```

## Reproducibility

The implementation uses segment-aware data partitions, training-only preprocessing, validation-only model and alarm selection, and frozen evaluation on Dataset-1 test, Dataset-2, and Dataset-3.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the complete protocol and artifact-provenance policy.

## Generated Outputs

Generated files are written locally under `data/`, `results/`, and `logs/`. These directories are excluded from Git to keep the repository lightweight and to prevent accidental publication of datasets, checkpoints, and machine-specific artifacts.

## Citation

The manuscript citation will be added after publication. Until then, please cite this repository and the corresponding manuscript when available.

The AV-GPS datasets and GPS-IDS framework are associated with:

> Abrar et al., “GPS-IDS: An Anomaly-based GPS Spoofing Attack Detection Framework for Autonomous Vehicles,” arXiv:2405.08359.

## License

No open-source license has been assigned yet. Until a license is added, all rights are reserved and reuse requires permission from the repository authors.

## Contact

For implementation questions, please open a GitHub issue or contact the repository maintainers.
