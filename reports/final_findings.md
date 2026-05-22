# Final Findings: Attack-Type-Specific IDS Model Evaluation

## Research Question

This project asks which attack categories are detected better by a lightweight PyTorch neural network compared with traditional machine learning baselines on flow-level intrusion detection data.

The experiment uses the CSE-CIC-IDS2018 processed flow-feature CSV files. The final sampled dataset contains 50,000 flows, 79 numeric flow features, 37,308 benign flows, and 12,692 attack flows. The sampled dataset preserves 14 attack labels, with rare-label preservation enabled during preprocessing.

## Experimental Setup

The task is binary classification: each flow is classified as benign or malicious. Original attack labels are preserved only for attack-type-specific evaluation.

Models compared:

- Logistic Regression with balanced class weights
- Random Forest with balanced subsampling
- Lightweight PyTorch multilayer perceptron

Split:

- Training rows: 32,000
- Validation rows: 8,000
- Test rows: 10,000
- Random seed: 42

Primary metrics:

- Precision
- Recall
- F1-score
- False negative rate
- Confusion matrix counts
- Attack-type-specific detection rate

## Overall Results

| Model | Precision | Recall | F1 | False Negative Rate | False Negatives |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.512 | 0.528 | 0.520 | 0.472 | 1,197 |
| Random Forest | 0.649 | 0.471 | 0.546 | 0.529 | 1,342 |
| PyTorch MLP | 0.615 | 0.507 | 0.556 | 0.493 | 1,252 |

The PyTorch MLP achieved the highest overall F1-score, but it did not dominate recall. Logistic Regression had the highest attack recall and the lowest false negative rate. Random Forest had the highest precision, but it missed the most attacks.

This supports the main argument of the project: aggregate performance alone is not enough for IDS evaluation. A model can look stronger by one metric while still missing more attacks.

## Attack-Type-Specific Results

| Attack Type | Test Attacks | Best Model by Recall | Best Recall | Main Observation |
|---|---:|---|---:|---|
| Bot | 24 | Random Forest | 0.958 | Tree-based model separated this category best. |
| Brute Force -Web | 18 | Random Forest / PyTorch MLP | 0.778 | Neural and RF models tied; Logistic Regression lagged. |
| Brute Force -XSS | 20 | All models | 1.000 | All models detected every test attack. |
| DDOS attack-HOIC | 618 | All models | 1.000 | Clear feature signature; all models detected all samples. |
| DDOS attack-LOIC-UDP | 15 | All models | 1.000 | All models detected all samples, but support is small. |
| DDoS attacks-LOIC-HTTP | 20 | PyTorch MLP | 1.000 | MLP detected all samples; other models missed one. |
| DoS attacks-GoldenEye | 19 | All models | 1.000 | All models detected every test attack. |
| DoS attacks-Hulk | 23 | Logistic Regression / PyTorch MLP | 1.000 | RF missed one sample. |
| DoS attacks-SlowHTTPTest | 28 | All models | 1.000 | All models detected every test attack. |
| DoS attacks-Slowloris | 23 | All models | 1.000 | All models detected every test attack. |
| FTP-BruteForce | 18 | All models | 1.000 | All models detected every test attack. |
| Infilteration | 1,669 | Logistic Regression | 0.299 | Hardest category; all models missed most samples. |
| SQL Injection | 20 | Random Forest | 1.000 | RF detected all samples; support is small. |
| SSH-Bruteforce | 23 | All models | 1.000 | All models detected every test attack. |

Most DoS, DDoS, and brute-force categories were detected very well by all three models. The hardest category was Infilteration, where even the best model, Logistic Regression, detected only 29.9% of test attacks.

## Hypothesis Evaluation

The initial hypothesis was that the PyTorch MLP would improve recall on attack categories involving nonlinear interactions among flow features, while Random Forest would remain competitive on attacks with clearer signatures.

The results partially support this hypothesis. The PyTorch MLP achieved the highest overall F1-score and matched or exceeded the traditional baselines on several specific attack categories. It was the only model to reach perfect recall on DDoS attacks-LOIC-HTTP.

However, the MLP did not achieve the highest overall attack recall. Logistic Regression had the best recall and the lowest false negative rate. Random Forest was competitive and had the highest precision, especially on Bot and SQL Injection, but it also produced the highest false negative count overall.

The result is therefore not "neural model wins." A better conclusion is that model behavior depends strongly on attack type, and traditional baselines remain competitive for many flow-level IDS categories.

## Security Interpretation

False negatives matter more than ordinary classification error in an IDS context because a false negative means malicious traffic was missed. On the final test split:

- Logistic Regression missed 1,197 attacks.
- Random Forest missed 1,342 attacks.
- PyTorch MLP missed 1,252 attacks.

The biggest security concern is Infilteration. It accounted for 1,669 test attack samples, but all models missed most of them:

- Logistic Regression missed 1,170 Infilteration attacks.
- Random Forest missed 1,335 Infilteration attacks.
- PyTorch MLP missed 1,241 Infilteration attacks.

This means a model could report strong detection on several obvious attack categories while still leaving a serious blind spot.

## Limitations

These results should be interpreted as behavior on a sampled CSE-CIC-IDS2018 flow-feature dataset, not as a production IDS benchmark.

Important limitations:

- The dataset is sampled from a larger public dataset and may preserve dataset-specific artifacts.
- Several rare attack categories have small test support, so perfect recall on those categories should not be overclaimed.
- The experiment uses binary classification for training and attack labels only for post-hoc evaluation.
- The models use flow-level CSV features, not raw packets or real-time traffic capture.
- Hyperparameter tuning was intentionally limited to keep the project reproducible and scoped.

## Final Answer

The lightweight PyTorch MLP was useful and achieved the best overall F1-score, but it was not uniformly better than traditional models. Logistic Regression produced the best overall attack recall and lowest false negative rate. Random Forest produced the strongest precision and performed best on some specific attack types such as Bot and SQL Injection.

The clearest finding is attack-type variation: most DoS, DDoS, and brute-force categories were detected well, while Infilteration was difficult for every model. This supports the project's central claim that IDS evaluation should report attack-type-specific detection and false negatives, not only aggregate accuracy.

