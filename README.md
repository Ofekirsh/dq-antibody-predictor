# DQ Antibody Predictor - Predicting Donor-Specific Antibody Emergence After Transplantation

This project predicts the emergence of donor-specific antibodies (DSAs) that drive organ rejection by analyzing mismatches between patient and donor HLA-DQ proteins.
Unlike traditional approaches that rely solely on allele or eplet counts, our method combines eplet data with biological and genetic distances to achieve high predictive performance (AUC = 0.84).

---

## Getting Started

This project provides **two main pipelines**:

1. **Training Pipeline** – Train a custom model using your dataset.
2. **Inference Pipeline** – Use our pre-trained model to predict DSA development for new samples.

---

## Training Pipeline

If you have suitable training data, use the `train_pipeline.py` script to train your own model.

### Input Format

Your training data should follow the same structure as shown in [`data/example_train_dataset`](data/example_train_dataset). This includes donor-patient allele information required for feature extraction.

### Configuration

- Config file: [`configs/train_pipeline_config.yml`](configs/train_pipeline_config.yml)
- Parameters explained: [`configs/readme.md`](configs/readme.md)

### Run the Pipeline

```bash
python train_pipeline.py --config configs/train_pipeline_config.yml --mode combined
```

- `--mode` can be `eplet`, `biological`, or `combined`
- Use `--skip-features` to skip feature extraction if already done

### Output Structure

Outputs will be saved under the `output/` directory:

- `output/features/` – Extracted feature vectors
- `output/missing_keys/` – Lists of missing allele data (e.g., `missing_pam_keys.csv`)
- `output/models_saved/`
  - `model.pkl`: Trained model
  - `model_metadata.pkl`: Training metadata
- `output/plots/train_roc_curve.png`: ROC curve and AUC of the training process

---

## Inference Pipeline

To run predictions on new patient-donor samples, use the `inference_pipeline.py` script.

### Input Format

Prepare your test dataset in the same format as [`data/example_inference_sample`](data/example_inference_sample).
You can use any path, but make sure to update the `data/test_file` field in your config file accordingly.

### Configuration

- Config file: [`configs/inference_pipeline_config.yml`](configs/inference_pipeline_config.yml)
- Parameters explained: [`configs/readme.md`](configs/readme.md)

### Run the Pipeline

```bash
python inference_pipeline.py --config configs/inference_pipeline_config.yml
```

### Output Structure

Results are saved in the `output/` directory:

- `output/features/` – Extracted feature vectors for your test samples
- `output/missing_keys/` – Lists of skipped samples due to missing data (e.g., `missing_pam_keys.csv`)
- `output/plots/inference_roc_curve.png` – ROC curve and AUC of inference performance

### Pretrained Model

- The pre-trained model is available at: `output/models_saved/model.pkl`

---

