# Master Plan: Attack-Type-Specific IDS Model Evaluation

## 1. Project Goal

Build a reproducible experiment pipeline for comparing traditional machine learning models and a lightweight PyTorch neural network on flow-level network intrusion detection.

The project should answer the final abstract's research question:

> Which attack categories are detected better by a lightweight PyTorch-based neural network compared with traditional machine learning baseline models on flow-level network traffic data?

The final result should emphasize security-relevant metrics instead of aggregate accuracy:

- Recall
- False negative rate
- F1-score
- Confusion matrices
- Attack-type-specific detection rates

## 2. Scope

### In Scope

- Use a public labeled flow-level IDS dataset, preferably CIC-IDS2017 or CSE-CIC-IDS2018.
- Frame the main task as binary classification: benign vs malicious.
- Preserve original attack labels for per-attack evaluation.
- Compare:
  - Logistic Regression
  - Random Forest
  - Lightweight PyTorch MLP
- Generate tables and plots suitable for a final report or presentation.
- Provide a demo command that predicts benign vs suspicious traffic from a test CSV.

### Out of Scope

- Live network packet capture.
- Real-time prevention or blocking.
- Production IDS deployment.
- Claiming that Random Forest represents a commercial IDS implementation.
- Over-optimizing for leaderboard accuracy.

## 3. Current Repository Status

The repository already contains a working MVP structure:

- `src/data.py`: CSV loading, label detection, feature cleaning, scaling, train/validation/test split, sample data generation.
- `src/models.py`: PyTorch MLP training and prediction.
- `src/evaluate.py`: binary metrics, confusion matrix plots, attack-type detection summaries.
- `src/train.py`: end-to-end training script for Logistic Regression, Random Forest, and PyTorch MLP.
- `requirements.txt`: core dependencies.

This means the next work should focus less on inventing the pipeline and more on making it robust, reproducible, and report-ready.

## 4. Implementation Phases

### Current Status Snapshot

- Completed: Phase 2 preprocessing hardening, Phase 3 baseline training, Phase 4 PyTorch MLP, Phase 5 attack-type evaluation, Phase 6 prediction demo, and most Phase 7 reproducibility work.
- Completed: Phase 1 real dataset setup using CSE-CIC-IDS2018 processed ML CSV files from the public S3 bucket. The local raw CSVs live under `data/raw/cse-cic-ids2018/processed/`.
- Completed: `data/processed/ids_sample.csv` generated with 50,000 sampled rows, 79 numeric flow features, and minimum rare-label preservation.
- Completed: final training run under `reports/final_run/` and `artifacts/final_run/`.
- Completed: Phase 8 final findings draft written to `reports/final_findings.md` using `reports/final_run/metrics_summary.csv`, `attack_type_detection.csv`, confusion matrices, and generated plots.
- Next step: review the final findings for course-report tone, then prepare slides or final submission material.

### Phase 1: Dataset Selection and Local Data Setup

Goal: choose one real IDS dataset and make the repo able to consume it consistently.

Tasks:

- Pick the primary dataset:
  - Selected: CSE-CIC-IDS2018 processed ML CSV files from the public S3 bucket `s3://cse-cic-ids2018/`.
  - Prefix to sync: `Processed Traffic Data for ML Algorithms/`.
  - Region used for public bucket listing/sync: `us-east-1`.
  - Local destination: `data/raw/cse-cic-ids2018/processed/`.
- Create a local data layout:
  - `data/raw/` for original downloaded CSV files.
  - `data/processed/` for merged or sampled CSV files.
  - `reports/` for outputs.
- Add dataset notes to `README.md`:
  - Dataset name.
  - Source URL.
  - Which CSV files were used.
  - Label column name.
  - Any sampling choices.
- Confirm the label column and attack label values.
- Run a quick exploratory count:
  - total rows
  - benign rows
  - malicious rows
  - rows per attack category
- Use `--min-rows-per-label 100` during dataset preparation so rare attack categories remain analyzable in the sampled dataset when enough source rows exist.

Deliverables:

- `data/processed/ids_sample.csv` or equivalent local processed dataset.
- `reports/dataset_profile.csv`
- `reports/dataset_metadata.json`
- `src/prepare_dataset.py`
- README dataset setup instructions.

Acceptance criteria:

- A single command can train on the selected CSV. Completed.
- The sampled dataset contains benign traffic and at least three attack categories. Completed with 14 attack labels in the sampled CSE-CIC-IDS2018 dataset.

## 5. Phase 2: Preprocessing Hardening

Goal: make preprocessing reliable for real CIC-style CSV files.

Tasks:

- Improve column cleanup:
  - strip whitespace
  - normalize duplicate column names if needed
  - drop obvious non-feature identifiers if present
- Handle invalid feature values:
  - convert numeric columns safely
  - replace `inf` and `-inf`
  - impute missing values
  - drop all-empty columns
- Preserve original attack labels before binary conversion.
- Add configurable benign label names:
  - `BENIGN`
  - `Benign`
  - `normal`
  - `0`
- Ensure stratified splitting works for binary labels.
- Consider attack-aware sampling so rare attacks are not accidentally removed.

Deliverables:

- More robust `src/data.py`
- Optional `reports/preprocessing_summary.csv`

Acceptance criteria:

- Pipeline does not crash on real dataset CSV quirks.
- Train/validation/test splits preserve both benign and malicious samples.

## 6. Phase 3: Baseline Model Training

Goal: train traditional models that are simple, defensible, and reproducible.

Tasks:

- Logistic Regression:
  - use standardized features
  - use `class_weight="balanced"`
  - record convergence settings
- Random Forest:
  - use `class_weight="balanced_subsample"`
  - record number of trees, max depth, and random seed
- Save trained model artifacts:
  - `artifacts/logistic_regression.pkl`
  - `artifacts/random_forest.pkl`
  - `artifacts/scaler.pkl`
  - `artifacts/feature_names.json`
- Save runtime metadata:
  - dataset path
  - row count
  - feature count
  - random seed
  - model hyperparameters

Deliverables:

- Updated `src/train.py`
- `artifacts/` model outputs
- `reports/metrics_summary.csv`

Acceptance criteria:

- Logistic Regression and Random Forest run end-to-end on the selected dataset.
- Metrics are reproducible using the same seed.

## 7. Phase 4: PyTorch MLP Implementation

Goal: implement a lightweight neural IDS model that matches the abstract.

Tasks:

- Keep the MLP intentionally small:
  - input layer = number of flow features
  - 1-2 hidden layers
  - ReLU activations
  - dropout
  - sigmoid output through logits
- Use weighted binary cross entropy for imbalance.
- Track validation loss and keep best checkpoint.
- Save:
  - `artifacts/mlp.pt`
  - model config
  - threshold used for classification
- Add optional hyperparameters:
  - epochs
  - batch size
  - learning rate
  - hidden dimension
  - dropout

Deliverables:

- Updated `src/models.py`
- Saved PyTorch checkpoint
- PyTorch metrics in `reports/metrics_summary.csv`

Acceptance criteria:

- MLP trains without GPU requirement.
- MLP produces binary predictions and attack probabilities.

## 8. Phase 5: Attack-Type-Specific Evaluation

Goal: answer the actual research question, not just report aggregate scores.

Tasks:

- For each model, compute:
  - precision
  - recall
  - F1-score
  - false negative rate
  - confusion matrix
- For each attack category, compute:
  - total attack samples
  - detected samples
  - missed false negatives
  - detection rate / recall
- Add comparative outputs:
  - model-by-attack detection table
  - attack categories where MLP wins
  - attack categories where Random Forest wins
  - attack categories where all models struggle
- Generate plots:
  - confusion matrix per model
  - grouped bar chart of detection rate by attack type
  - false negative rate comparison

Deliverables:

- `reports/metrics_summary.csv`
- `reports/attack_type_detection.csv`
- `reports/attack_type_detection.png`
- `reports/confusion_matrix_*.png`
- Optional `reports/model_comparison_findings.md`

Acceptance criteria:

- The report can clearly say which attack types each model detects better.
- False negatives are visible and discussed directly.

## 9. Phase 6: Demo Prediction Flow

Goal: satisfy the abstract's demo requirement.

Tasks:

- Add a prediction script, for example `src/predict.py`.
- Inputs:
  - path to a test CSV
  - model choice
  - saved artifact directory
- Outputs:
  - predicted benign/suspicious label
  - attack probability
  - optional original attack label if present
- Support batch prediction for a CSV file.
- Save predictions to:
  - `reports/demo_predictions.csv`

Deliverables:

- `src/predict.py`
- README demo command

Acceptance criteria:

- A user can run one command and see whether sample flows are benign or suspicious.
- PyTorch model reports attack probability, not only hard labels.

## 10. Phase 7: Reproducibility and CLI Cleanup

Goal: make the project easy to rerun before submission.

Tasks:

- Standardize CLI commands:
  - train on sample data
  - train on real data
  - predict with saved model
  - regenerate reports
- Add a config file if the command arguments become too long:
  - `configs/default.yaml`
- Save run metadata:
  - `reports/run_metadata.json`
- Make output filenames stable.
- Add `--random-state` everywhere randomness is used.
- Add basic smoke tests or a smoke command using generated sample data.

Deliverables:

- Updated README
- Optional `configs/default.yaml`
- `reports/run_metadata.json`

Acceptance criteria:

- A fresh clone can run the sample-data pipeline.
- A real-data run can be reproduced with the same command.

## 11. Phase 8: Final Report and Presentation Assets

Goal: turn experiment outputs into final ECS 252 submission material.

Status: Draft completed in `reports/final_findings.md`.

Tasks:

- Write final result narrative:
  - research question
  - hypothesis
  - dataset
  - preprocessing
  - models
  - metrics
  - attack-type findings
  - biggest risk / limitations
- Include plots:
  - overall metric table
  - confusion matrices
  - attack-type detection chart
- Discuss class imbalance and dataset artifacts.
- Avoid overclaiming:
  - say "on this dataset/sample"
  - distinguish detection component from active prevention
  - report false negatives prominently

Deliverables:

- `reports/final_findings.md` completed.
- Presentation-ready figures in `reports/final_run/` completed.

Acceptance criteria:

- Final writeup directly answers which attack categories are better detected by which model.
- Limitations are explicit and aligned with the abstract.

## 12. Suggested Command Flow

Sample-data smoke test:

```powershell
python src/train.py --generate-sample --epochs 5 --output-dir reports/sample_run --artifact-dir artifacts/sample_run
```

Prepare real raw IDS CSV files:

```powershell
aws s3 sync --no-sign-request --region us-east-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/cse-cic-ids2018/processed/
python src/prepare_dataset.py --input data/raw/cse-cic-ids2018/processed/*.csv --output data/processed/ids_sample.csv --profile-output reports/final_run/dataset_profile.csv --metadata-output reports/final_run/dataset_metadata.json --max-rows 50000
```

Real-data experiment:

```powershell
python src/train.py --csv data/processed/ids_sample.csv --label-column Label --max-rows 50000 --epochs 20 --output-dir reports/final_run --artifact-dir artifacts/final_run
```

Expected important outputs:

```text
reports/final_run/metrics_summary.csv
reports/final_run/attack_type_detection.csv
reports/final_run/attack_type_detection.png
reports/final_run/confusion_matrix_logistic_regression.png
reports/final_run/confusion_matrix_random_forest.png
reports/final_run/confusion_matrix_pytorch_mlp.png
```

## 13. Risk Management

### Risk: Dataset Imbalance

Mitigation:

- Use balanced class weights.
- Report false negative rate.
- Report per-attack recall.
- Avoid relying on accuracy.

### Risk: Rare Attack Categories Have Too Few Samples

Mitigation:

- Show support counts per attack type.
- Avoid strong claims for categories with tiny test counts.
- Use attack-aware sampling.

### Risk: Dataset Artifacts Inflate Performance

Mitigation:

- Discuss this as a limitation.
- Focus the conclusion on comparative behavior, not production readiness.
- Use held-out test split only for final reporting.

### Risk: Full Dataset Is Too Large

Mitigation:

- Start with a sampled subset.
- Make `--max-rows` configurable.
- Preserve multiple attack categories during sampling.

## 14. Final Success Criteria

The project is complete when it can produce:

- A reproducible training run using a real public IDS dataset.
- Metrics comparing Logistic Regression, Random Forest, and PyTorch MLP.
- Attack-type-specific detection rates.
- Confusion matrices and false negative analysis.
- A demo prediction command for test flows.
- A final findings document that directly answers the abstract's research question.
