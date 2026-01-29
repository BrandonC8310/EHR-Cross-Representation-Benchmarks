# Multivariate time-series baseline (regularised + imputed) for EHR benchmarking

This repository contains the code needed to train/evaluate baseline models on the **multivariate time-series representation** of EHR datasets.

Supported use case: Gridded/imputed time-series parquet files (one file for demographics/labels, one for time-series features) are already produced for each task, optionally under different missingness thresholds (e.g., `MaxMissRate0.9`).

## What is included

- `src/train_core_models.py`: binary classification (ICU mortality, readmission, pancreatic cancer)
- `src/train_core_models_multilabel.py`: multilabel classification (e.g., ICU phenotyping)
- `configs/`: task-specific YAML configs (data paths + model/training hyperparameters)
- `scripts/`: shell runners that generate a per-experiment derived config (so logs/checkpoints do not overwrite each other)

## Input parquet expectations

### Time-series parquet (`ts.parquet`)
- Identifier columns: `subject_id`, `hadm_id` (and optionally `icustay_id`), plus `timestep`
- All remaining columns are treated as numeric features.

### Demographics/label parquet (`demo.parquet`)
- Identifier columns: `subject_id`, `hadm_id` (and optionally `icustay_id`)
- Binary tasks: a `label` column.
- Multilabel tasks: multiple 0/1 label columns (default prefix `label_`).

## Quick start

### 1) Binary tasks (ICU mortality / readmission / pancreatic cancer)

Train a model:

```bash
./scripts/run_core_binary.sh train transformer -c configs/mimiciv_icu_mortality.yaml -d MIMICIV -t icu_mortality
```

Evaluate a trained checkpoint:

```bash
./scripts/run_core_binary.sh eval transformer path/to/best_model.pt -c configs/mimiciv_icu_mortality.yaml -d MIMICIV -t icu_mortality
```

### 2) Multilabel tasks (ICU phenotyping)

Train:

```bash
./scripts/run_core_multilabel.sh train retain -c configs/mimiciv_icu_phenotyping_25labels.yaml -d MIMICIV -t icu_phenotyping -m 25
```

Evaluate:

```bash
./scripts/run_core_multilabel.sh eval retain path/to/best_model.pt -c configs/mimiciv_icu_phenotyping_25labels.yaml -d MIMICIV -t icu_phenotyping -m 25
```

## Notes on missingness thresholds

The missingness threshold (e.g., `MaxMissRate0.9`) is handled upstream in the preprocessing pipeline. In this repo it is reflected only by the **data directory** you point to in the YAML config.

