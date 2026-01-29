# EHRSHOT-style MIMIC-IV (FEMR) Pipeline — Notebooks

## What this codebase does
These notebooks implement an EHRSHOT/FEMR-style pipeline for MIMIC-IV (via MEDS/OMOP-derived patient parquet inputs):

1. Build a FEMR patient database (EHRSHOT-style event stream) per split.
2. Generate labels for two ICU prediction tasks:
   - ICU mortality (binary)
   - ICU phenotyping (25-label multi-label)
3. Consolidate labels into `all_labels.csv` per task.
4. Generate representations for two models:
   - COUNT (Age + ontology-expanded CountFeaturizer)
   - CLMBR representations (via `clmbr_create_batches` + `clmbr_compute_representations`)
5. Train/evaluate downstream models (typically logistic regression) on COUNT vs CLMBR features.

## Missing-rate threshold experiment and the 4-experiment grid
The missing-rate threshold τ is applied during **Step 01 (database build)** as a code-filtering rule.
For each τ, you should create a separate output root (or separate database folder) to avoid overwriting outputs.

For each missing-rate threshold τ, you run the following 4 experiments:

1) COUNT × ICU mortality  
2) CLMBR × ICU mortality  
3) COUNT × ICU phenotyping (25-label multi-label)  
4) CLMBR × ICU phenotyping (25-label multi-label)

## Recommended execution order
1. `01_build_femr_database_from_meds_omop.ipynb`
2. `02_generate_labels_icu_mortality.ipynb`
3. `02_generate_labels_icu_phenotyping.ipynb`
4. `03_consolidate_labels.ipynb`
5. `04_generate_count_and_clmbr_features.ipynb`
6. `05_generate_shots_mortality.ipynb` (optional)
7. `05_generate_shots_phenotyping.ipynb` (optional)
8. `06_train_eval_count_vs_clmbr.ipynb`
9. `06_train_eval_multilabel_phenotyping.ipynb` (optional)

## Notes on directory conventions
The canonical path conventions used by the training/evaluation notebooks assume:

- Labels: `<BASE>/<split>/femr_labels/<TASK>/all_labels.csv`
- Features:
  - COUNT: `<BASE>/<split>/femr_features/<TASK>/count_features.pkl`
  - CLMBR: `<BASE>/<split>/femr_features/<TASK>/clmbr_features.pkl`
