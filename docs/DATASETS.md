# Dataset Guide

## 1. Scope

This project uses the AV-GPS dataset package associated with the GPS-IDS study:

> Abrar et al., “GPS-IDS: An Anomaly-based GPS Spoofing Attack Detection Framework for Autonomous Vehicles,” arXiv:2405.08359.

The datasets are **not redistributed** in this repository. Users must obtain them from the original authors or the official dataset source and comply with the applicable access and usage conditions.

## 2. Expected Local Files

Place the raw CSV files in:

```text
data/raw/
```

Expected filenames:

```text
data/raw/
├── AV-GPS-Dataset-1.csv
├── AV-GPS-Dataset-1-Normal-Data.csv
├── AV-GPS-Dataset-2.csv
└── AV-GPS-Dataset-3.csv
```

Do not rename these files unless the corresponding paths are also updated in the configuration.

## 3. Dataset Roles

| File | Role in this repository |
|---|---|
| `AV-GPS-Dataset-1.csv` | Main mixed dataset used for source-aware development, validation, and held-out testing |
| `AV-GPS-Dataset-1-Normal-Data.csv` | Normal-reference data used for normal-behavior statistics; it is not treated as an independent external test dataset |
| `AV-GPS-Dataset-2.csv` | External/source-shift evaluation dataset |
| `AV-GPS-Dataset-3.csv` | Online case-study dataset; its EKF detector output is used only for comparison in the dedicated case study |

## 4. Processing Layers

The code creates the following local data layers:

```text
data/
├── raw/                         # Original AV-GPS CSV files
├── interim/                     # Cleaned, segmented, physical, and residual tables
├── processed/                   # Final causal-evidence and sequence-ready files
├── splits/                      # Segment metadata and split definitions
└── extended_comparison/         # Isolated artifacts for external comparisons
```

These directories are ignored by Git. Generated files should not be committed.

## 5. Split and Evaluation Policy

Dataset-1 is partitioned at the segment level into training, validation, and test subsets. The intended policy is:

- training split: model fitting and estimation of preprocessing statistics;
- validation split: model selection and alarm operating-point selection;
- Dataset-1 test: held-out in-domain evaluation;
- Dataset-2: held-out external/source-shift evaluation;
- Dataset-3: held-out online case-study evaluation.

Dataset-1 test, Dataset-2, and Dataset-3 must not be used for fitting, hyperparameter tuning, imputation/scaling estimation, threshold selection, or persistence selection.

## 6. Leakage Controls

The implementation is designed to preserve the following controls:

- source and row metadata are retained for provenance but are not used as predictive features;
- label and detector-output columns are excluded from causal feature construction;
- trajectory continuity is respected during segmentation and alarm evaluation;
- preprocessing statistics are estimated from Dataset-1 training data only;
- normal-reference statistics must not use held-out evaluation labels.

## 7. Dataset-3 EKF Field

Dataset-3 contains an EKF detector output used in the dedicated proposed-versus-EKF case study. This field:

- is not an input to the proposed detector;
- is not used during training;
- is not used for model or alarm calibration;
- is evaluated only as an external comparator under the common event protocol.

## 8. Configuration

Dataset names and paths are controlled primarily by:

```text
configs/dataset.yaml
configs/paths.yaml
configs/preprocessing.yaml
```

Machine-specific absolute paths should not be committed. Prefer paths relative to the repository root.

## 9. Preflight Check

After placing the raw files, run:

```bash
python main.py --mode step2
```

This checks file availability, loads the datasets, preserves source metadata, and prints raw-data summaries before later processing steps are executed.
