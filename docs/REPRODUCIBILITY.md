# Reproducibility Protocol

## 1. Purpose

This document records the experimental safeguards used to prevent data leakage, post-test tuning, and inconsistent alarm evaluation across the proposed detector, common-input baselines, ablations, and external comparisons.

## 2. Data Partitioning

Dataset-1 is split at the trajectory-segment level into:

- training;
- validation;
- held-out test.

The same split identity must be reused across comparable experiments. Segment identifiers, row order, and within-segment indices are preserved so that sequence construction and event evaluation remain traceable.

Dataset-2 and Dataset-3 are held out as external evaluation scenarios.

## 3. Training-Only Preprocessing

All learned preprocessing quantities must be estimated exclusively from Dataset-1 training data, including:

- normal-reference residual statistics;
- scaling parameters;
- imputation statistics;
- any learned feature normalization;
- class- or sample-weight parameters derived from training data.

Preprocessing fitted on validation or test samples is prohibited.

## 4. Validation-Only Selection

Dataset-1 validation is the only split used for:

- hyperparameter selection;
- checkpoint selection;
- alarm threshold selection;
- persistence selection;
- other operating-point decisions.

After selection, the preprocessing pipeline, model checkpoint, threshold, and persistence are frozen.

Dataset-1 test, Dataset-2, and Dataset-3 must not influence fitting, tuning, alarm calibration, or model selection.

## 5. Frozen Evaluation

The frozen model and operating point are evaluated without retuning on:

- Dataset-1 test;
- Dataset-2;
- Dataset-3.

The same alarm semantics must be used for every method being compared:

- causal probability stream;
- segment-stable processing;
- fixed threshold;
- fixed persistence;
- no persistence carry-over across segment boundaries;
- common row-level and event-level metrics.

## 6. Event-Level Evaluation

Row-level metrics and event-level metrics serve different purposes and should both be retained.

Typical row-level metrics include:

- AUPRC;
- AUROC;
- F1;
- precision;
- recall;
- false-positive rate.

Typical event-level metrics include:

- attack detection rate;
- number of detected and missed events;
- detection delay;
- persistence-qualified alarm onset.

Alarm parameters must be selected on validation data and frozen before evaluating test or external scenarios.

## 7. Seed Policy

The default single-seed setting is seed 42.

The proposed-model multi-seed experiment is executed through:

```bash
python main.py --mode step21_multiseed
```

Its seed list is controlled by `configs/experiments.yaml`. The current intended robustness set is:

```text
42, 43, 44, 45, 46
```

Step 21 manages this list internally and should not be wrapped again by the global multi-seed loop.

For stochastic comparisons, report individual-seed results and aggregate mean/standard deviation where available.

## 8. Artifact Provenance

Manuscript-grade runs should preserve:

- resolved configuration;
- code version or Git commit hash;
- dataset and split hashes;
- feature order and feature-contract hash;
- checkpoint hash;
- preprocessing state;
- active seed;
- threshold source;
- selected threshold and persistence;
- prediction-bundle metadata;
- evaluation summary;
- completion status.

Generated artifacts should be immutable after a run is reported. A changed configuration, code version, split, or preprocessing state requires a new run identifier.

## 9. Corrected Causal-Evidence Branch

The corrected evidence branch is isolated from legacy outputs. It preserves:

- row and segment identity;
- label-independent evidence construction;
- training-only normal statistics;
- explicit validity masks;
- locked causal-evidence feature order;
- separate output roots for corrected comparison artifacts.

Labels are used for supervised training and evaluation, not for constructing causal evidence.

## 10. Common-Input Architectural Comparisons

Architectural claims should be based on methods trained with the same causal-evidence input, split, selection procedure, and alarm evaluator.

Common-input comparisons may include:

- XGBoost;
- MLP;
- LSTM;
- GRU;
- TCN;
- causal Transformer;
- first-order liquid model;
- proposed second-order/high-order liquid model.

Differences in performance can then be interpreted primarily as architectural rather than input-representation effects.

## 11. GPS-IDS External Reimplementation

The GPS-IDS branch is reported as a **protocol-controlled reimplementation**, not an exact reproduction.

The original GPS-IDS study describes seven classifiers and states that 14 behavior-model features are used. This repository reconstructs a locked, ordered behavior-feature contract from the variables specified in the original formulation and evaluates all classifiers under the repository's common segment split and alarm protocol.

The branch must remain isolated under:

```text
external_baselines/gps_ids_reproduction/
data/extended_comparison/gps_ids_reproduction/
results/extended_comparison/gps_ids_reproduction/
```

For every GPS-IDS classifier:

- preprocessing is fitted on Dataset-1 training only;
- candidates are selected on Dataset-1 validation only;
- the selected pipeline is refitted on Dataset-1 training only;
- threshold and persistence are selected on Dataset-1 validation only;
- the frozen pipeline and operating point are evaluated on held-out scenarios;
- method-specific masks or missingness metadata are not added as predictive features unless explicitly declared.

## 12. Reporting Rules

Use precise language:

- “reimplementation” rather than “exact reproduction”;
- “external-method comparison” when input representations differ;
- “common-input architectural comparison” when every method uses the same evidence representation;
- “validation-selected and frozen” for alarm operating points;
- “held-out” only when a split was not used for fitting or selection.

Do not claim that the proposed method outperforms every evaluated classifier unless the complete results support that statement.

## 13. Repository Exclusions

The public Git repository excludes:

- raw and processed datasets;
- trained model checkpoints;
- prediction arrays;
- generated figures and tables;
- logs;
- machine-specific IDE files;
- environment files and credentials.

These exclusions do not remove the need to retain local manuscript-grade artifacts and their hashes.

## 14. Reproduction Checklist

Before reporting a run, confirm:

- [ ] Raw filenames and configuration paths are correct.
- [ ] Segment split identities match the locked split.
- [ ] No test or external sample was used for preprocessing.
- [ ] No test or external sample was used for model selection.
- [ ] Threshold and persistence came from Dataset-1 validation.
- [ ] The operating point was frozen before test evaluation.
- [ ] Segment boundaries reset persistence state.
- [ ] Feature order and hashes were recorded.
- [ ] Seed and code version were recorded.
- [ ] Prediction and evaluation artifacts passed integrity checks.
- [ ] The reported table can be regenerated from saved outputs.
