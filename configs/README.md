# Configuration Guide

This document explains the YAML configuration files used in the training and inference pipelines.
---

## Inference Pipeline Configuration

File: `configs/inference_pipeline_config.yml`

This file configures the inference process using a pre-trained model.

### `data`
```yaml
data:
  test_file: "../data/example_inference_samples.csv"
```
- Path to the test dataset (CSV) containing patient and donor HLA information.
- Must follow the format shown in `data/example_inference_sample`.

---

### `features`
Defines the paths to required resources and output files for feature extraction.

#### `eplet`
```yaml
eplet:
  dqa_to_sequence_file: "../data/jsons/dqa_to_seq.json"
  dqb_to_sequence_file: "../data/jsons/dqb_to_seq.json"
  allele_descriptions_file: "../data/jsons/all_eplets.json"
  output_file: "../output/features/eplet_features.csv"
```
- Maps alleles to amino acid sequences and eplets for one-hot encoding and save the vectors in `output_file`.

#### `biological`
```yaml
biological:
  ems_distances_file: "../data/jsons/ems_distances.json"
  pam_distances_file: "../data/jsons/pam_distances.json"
  g12_values_file: "../data/jsons/g12_classifications.json"
  features_to_extract: [ "PAM", "EMS", "G12" ]
  output_file: "../output/features/biological_features.csv"
```
- Uses genetic/biochemical distances (EMS, PAM) and G12 classification to extract features and save the vectors in `output_file`.

#### `combined`
```yaml
combined:
  output_file: "../output/features/combined_features.csv"
```
- save the vectors in `output_file`

---

### `model`
```yaml
model:
  path: "../output/models_saved/model.pkl"
```
- Path to the pre-trained model used for inference.

---

### `output`
```yaml
output:
  predictions_file: "../output/predictions/predictions.csv"
```
- Location where predictions will be saved as a CSV.

---

### `evaluation`
```yaml
evaluation:
  roc_plot_path: "../output/plots/inference_roc_curve.png"
```
- Path to save the ROC curve and AUC plot for inference results.

---

## Train Pipeline Configuration

File: `configs/train_pipeline_config.yml`

This file configures the full training process from raw data to model evaluation.

### `data`
```yaml
data:
  input_file: "../data/example_train_dataset"
```
- Must match the format in `data/example_train_dataset`.

---

### `features`
Same structure as in inference:
- Extracts features from allele sequences and distance files.
- Supports individual (`eplet`, `biological`) or combined modes.

---

### `model`
```yaml
model:
  type: "logreg"      # Options: "lgbm", "xgb", "logreg"
  k_folds: 5          # Number of folds for cross-validation
  save: true          # Whether to save the trained model
  path: "../output/models_saved/model.pkl"
```
- Defines model type and training behavior.

---

### `evaluation`
```yaml
evaluation:
  roc_plot_path: "../output/plots/train_roc_curve.png"
```
- Path to save the ROC curve and AUC result from training.

---

## Summary of Outputs

| Folder             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `output/features/` | Extracted feature vectors for training or inference samples                 |
| `output/models_saved/` | Trained model (`model.pkl`) and metadata (`model_metadata.pkl`)       |
| `output/predictions/` | CSV of predictions for new data (inference only)                         |
| `output/plots/`     | ROC curve visualizations for training and inference                        |
| `output/missing_keys/` | Logs of allele pairs skipped due to missing data (e.g., PAM/EMS distance) |

---
