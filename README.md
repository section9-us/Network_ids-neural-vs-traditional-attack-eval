# AttackLens AI IDS

Attack-Type-Specific Evaluation of Neural and Traditional Models for
Flow-Level Intrusion Detection.

Repository/local project name: `ids-neural-vs-traditional-attack-eval`

This is a runnable sample project for comparing traditional machine learning
models and a lightweight PyTorch MLP on flow-level network intrusion detection.

The project maps directly to the proposed research question:

- Binary IDS classification: `benign` vs `attack`
- Baselines: Logistic Regression, Random Forest
- PyTorch model: lightweight multilayer perceptron
- Metrics: precision, recall, F1, false negative rate, confusion matrix
- Extra analysis: attack-type-specific detection rate

## Project Structure

```text
.
|-- data/
|   |-- raw/              # Put CIC-IDS2017/CSE-CIC-IDS2018 CSV files here
|   `-- sample_flows.csv  # Generated toy dataset for testing
|-- reports/              # Metrics and plots
|-- src/
|   |-- data.py           # Loading, cleaning, encoding, splitting
|   |-- evaluate.py       # Metrics and attack-type analysis
|   |-- models.py         # PyTorch MLP
|   `-- train.py          # Main experiment script
|-- requirements.txt
`-- README.md
```

## Quick Start

Install Python 3.10+ first. On Windows, make sure the real Python executable is
on PATH, not only the Microsoft Store alias.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline on generated sample data:

```bash
python src/train.py --generate-sample
```

If your machine uses `python3` instead:

```bash
python3 src/train.py --generate-sample
```

Outputs are saved in `reports/`:

- `metrics_summary.csv`
- `attack_type_detection.csv`
- `test_predictions.csv`
- `attack_type_detection.png`
- `confusion_matrix_*.png`

## Run With a CIC CSV

Put CSV files under `data/raw/`, then run:

```bash
python src/train.py --csv data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

If your dataset has a label column with a different name:

```bash
python src/train.py --csv data/raw/your_file.csv --label-column Label
```

The script keeps the original attack label for attack-type-specific analysis,
while training models as binary classifiers.

## What To Submit Or Present

Use this project to produce:

1. A table comparing Logistic Regression, Random Forest, and PyTorch MLP.
2. Confusion matrices for all three models.
3. Attack-type detection rates for each model.
4. A discussion of false negatives, not only overall accuracy.
5. A short demo that loads a CSV and prints/exports model predictions.

## Suggested Final Report Outline

1. Research question: which attack types are better detected by PyTorch MLP vs
   traditional ML?
2. Dataset and preprocessing: flow features, missing/infinite value handling,
   label encoding, standardization.
3. Models: Logistic Regression, Random Forest, PyTorch MLP.
4. Metrics: precision, recall, F1, false negative rate, confusion matrix.
5. Attack-type analysis: DDoS, brute-force, botnet, web attacks, etc.
6. Discussion: why overall accuracy can hide security risk.
7. Limitations: dataset imbalance, CIC artifacts, sampled subset, limited
   generalization.

## Notes

This project implements the detection component only. It does not block traffic
or modify firewall/IPS rules.
