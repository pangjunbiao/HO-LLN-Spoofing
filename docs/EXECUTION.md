# Execution Guide

## 1. Recommended Interface

Run experiments from the repository root through `main.py`:

```bash
python main.py --mode <MODE>
```

The `--mode` argument overrides `run.mode` in `configs/default.yaml`, so users do not need to repeatedly comment and uncomment YAML blocks.

Optional overrides:

```bash
python main.py --mode step13 --device cuda
python main.py --mode step13 --device cpu
python main.py --mode step13 --seed-mode single
python main.py --config-dir configs --mode step13
```

Supported device values are `cuda`, `cpu`, and `auto`. Supported seed modes are `single` and `multi`.

## 2. Before Running

1. Install dependencies from `requirements.txt`.
2. Place the AV-GPS files under `data/raw/`.
3. Review machine-specific paths in `configs/paths.yaml`.
4. Run Step 1 to verify configuration, logging, seeding, device selection, and I/O.
5. Run the numbered pipeline in dependency order.

## 3. Canonical Execution Modes

| Mode | Purpose |
|---|---|
| `step1` | Configuration, logging, seed, device, directory, and I/O setup |
| `step2` | Raw AV-GPS data loading and inspection |
| `step3` | Trajectory segmentation and preliminary transition-validity analysis |
| `step4` | Source-aware Dataset-1 segment-level train/validation/test split |
| `step5` | Shortcut-column exclusion and clean causal-column selection |
| `step6` | Coordinate transformation and motion-model construction |
| `step7` | Residual construction and training-only normal statistics |
| `step8` | Construction of the final causal-evidence sequence |
| `step9` | Dataset objects and sequence batching |
| `step10` | Evaluation metrics, validation-only operating-point selection, and alarm rules |
| `step11` | Proposed-model modules and model-factory verification |
| `step12` | Common PyTorch training-loop verification |
| `step13` | Full proposed-model training and evaluation |
| `step14` | Full-model module-usage diagnostics |
| `step15` | Official common-input baseline comparison |
| `step16` | Official controlled ablation study |
| `step16_frozen` | Frozen-checkpoint component-intervention ablation |
| `step17_high_order` | Feature high-order versus model high-order comparison |
| `step17a_features` | Feature-group intervention through the trained proposed model |
| `step17b_kirchhoff` | Full-feature Kirchhoff/model high-order structure comparison |
| `step17_final` | Combined Step 17A and Step 17B analysis |
| `step19` | Dataset-3 proposed-versus-EKF online case study |
| `step20` | Table-only operating-point sensitivity analysis |
| `step21_multiseed` | Proposed-model multi-seed robustness analysis |

There is currently no canonical `step18` or `all` mode. Do not document or invoke modes that are not implemented in `main.py`.

## 4. Main Sequential Pipeline

Run Steps 1–13 in order:

```bash
python main.py --mode step1
python main.py --mode step2
python main.py --mode step3
python main.py --mode step4
python main.py --mode step5
python main.py --mode step6
python main.py --mode step7
python main.py --mode step8
python main.py --mode step9
python main.py --mode step10
python main.py --mode step11
python main.py --mode step12
python main.py --mode step13
```

### Step dependencies

- Steps 2–8 construct the corrected causal-evidence data.
- Step 9 creates sequence-ready dataset objects.
- Step 10 validates the common evaluation and alarm-selection protocol.
- Steps 11–12 verify the model and training infrastructure.
- Step 13 performs the official proposed-model experiment.

## 5. Post-Training Experiments

After Step 13 has completed successfully:

```bash
python main.py --mode step14
python main.py --mode step15
python main.py --mode step16
python main.py --mode step16_frozen
python main.py --mode step17_high_order
python main.py --mode step17a_features
python main.py --mode step17b_kirchhoff
python main.py --mode step17_final
python main.py --mode step19
python main.py --mode step20
python main.py --mode step21_multiseed
```

### Important prerequisites

- `step14` uses the trained full proposed model.
- `step15` runs the official common-input baseline comparison.
- `step16` trains/evaluates the controlled ablation variants.
- `step16_frozen` applies component interventions to the frozen full-model checkpoint.
- `step17a_features` evaluates feature-group contributions through the trained proposed model.
- `step17b_kirchhoff` trains the K0/K1/K2 structural variants and reuses the official Step-13 proposed checkpoint for K3.
- `step17_final` runs the combined Step 17A and Step 17B analysis.
- `step19` reuses saved Step-13 Dataset-3 predictions; it does not retrain or retune the proposed detector.
- `step20` reuses saved Step-13 Dataset-1 test predictions and evaluates threshold/persistence sensitivity.
- `step21_multiseed` manages its own configured seed list.

For Step 21, verify that `configs/experiments.yaml` enables the multi-seed experiment and defines the seed list, for example:

```yaml
experiments:
  multiseed_proposed:
    enabled: true
    seeds: [42, 43, 44, 45, 46]
```

## 6. YAML-Based Execution

CLI mode selection is recommended. YAML selection remains supported by keeping exactly one active `run` block in `configs/default.yaml`:

```yaml
run:
  mode: "step13"
  description: "Step 13: full proposed model experiment"
  save_logs: true
  print_console_summary: true
```

Do not leave multiple uncommented `run:` blocks in the same YAML file.

## 7. GPS-IDS Classifier-Suite Reimplementation

The GPS-IDS comparison is separate from the numbered main pipeline.

Official run:

```bash
python -m src.experiments.run_gps_ids_reproduction --config-dir configs --gps-ids-features-config gps_ids_features.yaml --gps-ids-classifiers-config gps_ids_classifiers.yaml
```

Installation smoke test with a reduced model set:

```bash
python -m src.experiments.run_gps_ids_reproduction --config-dir configs --gps-ids-features-config gps_ids_features.yaml --gps-ids-classifiers-config gps_ids_classifiers.yaml --search-profile smoke --models mlp decision_tree --overwrite
```

Smoke-test metrics are for installation validation only and must not be reported as manuscript results.

The complete suite includes:

- Random Forest
- XGBoost
- SVC
- MLP
- AdaBoost
- Gradient Boosting
- Decision Tree

## 8. Output Locations

The main pipeline writes generated artifacts under:

```text
data/
results/
logs/
```

The GPS-IDS branch writes isolated outputs under:

```text
data/extended_comparison/gps_ids_reproduction/
results/extended_comparison/gps_ids_reproduction/
```

Generated data, checkpoints, predictions, logs, and result artifacts are excluded from Git.

## 9. Recommended Run Record

For each manuscript experiment, retain:

- the resolved YAML configuration;
- active seed or seed list;
- code commit hash;
- selected checkpoint;
- validation-selected threshold and persistence;
- split and feature hashes;
- saved prediction bundles;
- final evaluation summaries.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the required safeguards.
