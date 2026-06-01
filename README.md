# Attack-Type-Specific Evaluation of Neural and Traditional Models for Flow-Level Intrusion Detection

This project compares traditional machine learning models and several PyTorch neural models for flow-level network intrusion detection. The main analysis focuses on recall, false negatives, and attack-type-specific detection rates instead of aggregate accuracy alone.

## Models

- Logistic Regression
- Random Forest
- Shallow PyTorch MLP
- PyTorch multilayer perceptron
- Deep PyTorch MLP
- Autoencoder + MLP

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Recreate Data and Model Files

Large data and model files are intentionally not committed to GitHub.

Ignored/generated paths:

- `data/raw/`
- `data/processed/`
- `data/*.csv`
- `artifacts/`
- `reports/final_run/`
- other run-specific folders under `reports/`

To reproduce the same local setup on another machine, run the commands below after installing the requirements.

Download the CSE-CIC-IDS2018 processed ML CSV files:

```powershell
aws s3 sync --no-sign-request --region us-east-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/cse-cic-ids2018/processed/
```

This downloads about 6.4 GB of CSV files.

Create the sampled training CSV:

```powershell
python src/prepare_dataset.py --input data/raw/cse-cic-ids2018/processed/*.csv --output data/processed/ids_sample.csv --profile-output reports/final_run/dataset_profile.csv --metadata-output reports/final_run/dataset_metadata.json --max-rows 50000 --min-rows-per-label 100
```

Train the models and regenerate reports/artifacts:

```powershell
python src/train.py --csv data/processed/ids_sample.csv --label-column Label --max-rows 50000 --epochs 20 --output-dir reports/final_run --artifact-dir artifacts/final_run
```

By default, training includes `shallow_mlp`, `pytorch_mlp`, `deep_mlp`, and `autoencoder_mlp`. To shorten a run, pass a smaller comma-separated set with `--neural-models`.

Run the prediction demo:

```powershell
python src/predict.py --csv data/processed/ids_sample.csv --model pytorch_mlp --artifact-dir artifacts/final_run --output reports/final_run/demo_predictions.csv
```

The random seed defaults to `42`, so results should be reproducible apart from small differences caused by platform or package versions.

## Smoke Test With Generated Sample Data

```powershell
python src/train.py --generate-sample --epochs 5 --autoencoder-epochs 5 --output-dir reports/sample_run --artifact-dir artifacts/sample_run
```

This creates a small synthetic flow dataset at `data/sample_flows.csv` and writes reports to `reports/sample_run`.

## Train on a Real IDS CSV

Use the commands in **Recreate Data and Model Files** to download CSE-CIC-IDS2018, create `data/processed/ids_sample.csv`, and train the final models.

If the label column is named `Label`, `label`, `Attack`, `attack`, `class`, or `Class`, `--label-column` can usually be omitted.

## Prediction Demo

After training, run batch prediction on a CSV:

```powershell
python src/predict.py --csv data/processed/ids_sample.csv --model pytorch_mlp --artifact-dir artifacts/final_run --output reports/final_run/demo_predictions.csv
```

Available model names:

- `logistic_regression`
- `random_forest`
- `shallow_mlp`
- `pytorch_mlp`
- `deep_mlp`
- `autoencoder_mlp`

## Main Outputs

Training writes:

- `metrics_summary.csv`: precision, recall, F1, false negative rate, confusion matrix counts
- `attack_type_detection.csv`: detection rate per attack label
- `attack_family_detection.csv`: detection rate per coarse attack family
- `dataset_profile.csv`: row, feature, benign, attack, and attack-label counts
- `dataset_metadata.json`: input files, processed output path, sampling settings
- `test_predictions.csv`: held-out test predictions
- `run_metadata.json`: dataset path, split sizes, hyperparameters, random seed
- `model_comparison_findings.md`: best and weakest model by attack type
- `confusion_matrix_*.png`: confusion matrix per model
- `attack_type_detection.png`: grouped attack-type recall chart
- `attack_family_detection.png`: grouped attack-family recall chart
- `false_negative_rate.png`: model-level false negative comparison

Artifacts are saved under the selected artifact directory:

- `scaler.pkl`
- `feature_names.json`
- `logistic_regression.pkl`
- `random_forest.pkl`
- `shallow_mlp.pt`
- `pytorch_mlp.pt`
- `deep_mlp.pt`
- `autoencoder_mlp.pt`

## Research Framing

The primary research question is: which attack categories are detected better by a lightweight PyTorch neural network compared with traditional machine learning baselines?

The conclusion should avoid overclaiming. Results should be described as behavior on the selected public dataset or sampled subset, with class imbalance and dataset artifacts treated as explicit limitations.
