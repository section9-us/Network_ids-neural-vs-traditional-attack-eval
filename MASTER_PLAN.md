# 마스터 플랜: Attack-Type-Specific IDS Model Evaluation

## 1. 프로젝트 목표

이 프로젝트의 목표는 flow-level network intrusion detection 데이터를 사용해 전통적 머신러닝 모델과 PyTorch 기반 neural model을 재현 가능한 방식으로 비교하는 것이다.

최종 abstract의 핵심 research question은 다음과 같다.

> Flow-level network traffic data에서 lightweight PyTorch 기반 neural network가 traditional machine learning baseline model보다 어떤 attack category를 더 잘 탐지하는가?

최종 결과는 단순 aggregate accuracy보다 보안적으로 중요한 지표를 중심으로 해석한다.

- Recall
- False negative rate
- F1-score
- Confusion matrix
- Attack-type-specific detection rate

## 2. 범위

### 포함 범위

- Public labeled flow-level IDS dataset 사용
  - 현재 선택: CSE-CIC-IDS2018
- 주요 task는 binary classification으로 구성
  - benign vs malicious
- 원래 attack label은 보존해서 attack-type별 평가에 사용
- 비교 모델:
  - Logistic Regression
  - Random Forest
  - Shallow MLP
  - Lightweight PyTorch MLP
  - Deep MLP
  - Autoencoder + MLP
- final report / presentation에 사용할 수 있는 table과 plot 생성
- test CSV를 입력받아 benign/suspicious를 예측하는 demo command 제공

### 제외 범위

- Live network packet capture
- Real-time prevention 또는 blocking
- Production IDS deployment
- Random Forest가 commercial IDS의 정확한 구현이라고 주장하는 것
- Leaderboard accuracy를 위해 과도하게 튜닝하는 것

## 3. 현재 repository 상태

현재 repository에는 작동 가능한 MVP 구조가 있다.

- `src/data.py`
  - CSV loading
  - label detection
  - feature cleaning
  - scaling
  - train/validation/test split
  - sample data generation
- `src/models.py`
  - PyTorch MLP training and prediction
- `src/evaluate.py`
  - binary metrics
  - confusion matrix plots
  - attack-type detection summaries
- `src/train.py`
  - Logistic Regression, Random Forest, PyTorch MLP end-to-end training
- `src/prepare_dataset.py`
  - raw CSE-CIC-IDS2018 CSV를 sampled training CSV로 변환
- `src/predict.py`
  - saved artifact 기반 CSV prediction demo
- `requirements.txt`
  - core dependencies

즉, 기본 pipeline은 이미 구현되어 있고, 다음 작업은 논문을 참고한 neural model 확장과 attack family 분석 강화이다.

## 4. 구현 단계

### 현재 상태 요약

- 완료: Phase 1 real dataset setup
  - CSE-CIC-IDS2018 processed ML CSV 사용
  - public S3 bucket에서 다운로드 가능
  - local raw CSV 위치: `data/raw/cse-cic-ids2018/processed/`
- 완료: Phase 2 preprocessing hardening
- 완료: Phase 3 baseline training
- 완료: Phase 4 PyTorch MLP
- 완료: Phase 5 attack-type evaluation
- 완료: Phase 6 prediction demo
- 완료: Phase 7 reproducibility / CLI cleanup 대부분
- 완료: `data/processed/ids_sample.csv` 생성
  - 50,000 sampled rows
  - 79 numeric flow features
  - rare-label preservation 적용
- 완료: real CSE-CIC-IDS2018 데이터 재다운로드
  - 확인일: 2026-05-31
  - raw CSV 10개, 약 6.41GB
  - 위치: `data/raw/cse-cic-ids2018/processed/`
- 완료: processed sample 재생성
  - 위치: `data/processed/ids_sample.csv`
  - 50,000 sampled rows
  - 79 numeric flow features
  - label profile: `reports/final_run/dataset_profile.csv`
- 완료: 기존 final run
  - `reports/final_run/`
  - `artifacts/final_run/`
- 완료: 기존 final findings draft
  - `reports/final_findings.md`
- 구현 완료: Phase 9 코드 확장
  - Gamage and Samarabandu 논문, "Deep learning methods in network intrusion detection: A survey and an objective comparison" 참고
  - neural baseline 추가
  - coarse attack-family analysis 추가
- 완료: 확장된 Phase 9 final run 재실행
  - 실행일: 2026-05-31
  - 모델 6개 비교 완료
    - Logistic Regression
    - Random Forest
    - Shallow MLP
    - PyTorch MLP
    - Deep MLP
    - Autoencoder + MLP
  - 출력 위치: `reports/final_run/`
  - artifact 위치: `artifacts/final_run/`
- 완료: 확장된 모델/attack-family 결과로 final findings 재생성
  - 위치: `reports/final_findings.md`
- 다음 단계:
  - 최종 결과 검토
  - 필요하면 commit / push

## 5. Phase 1: Dataset Selection and Local Data Setup

목표: 실제 IDS dataset을 선택하고 repository가 일관되게 사용할 수 있게 준비한다.

작업:

- Primary dataset 선택
  - 선택 완료: CSE-CIC-IDS2018 processed ML CSV files
  - S3 bucket: `s3://cse-cic-ids2018/`
  - sync 대상 prefix: `Processed Traffic Data for ML Algorithms/`
  - region: `us-east-1`
  - local destination: `data/raw/cse-cic-ids2018/processed/`
- local data layout 구성
  - `data/raw/`: 원본 downloaded CSV files
  - `data/processed/`: merged 또는 sampled CSV files
  - `reports/`: output reports
- README에 dataset setup 방법 기록
  - dataset name
  - source URL / S3 command
  - 사용한 CSV files
  - label column
  - sampling choices
- label column과 attack label values 확인
- dataset profile 생성
  - total rows
  - benign rows
  - malicious rows
  - rows per attack category
- rare attack category 보존
  - `--min-rows-per-label 100` 사용

산출물:

- `data/processed/ids_sample.csv`
- `reports/dataset_profile.csv`
- `reports/dataset_metadata.json`
- `src/prepare_dataset.py`
- README dataset setup instructions

완료 기준:

- 단일 command로 selected CSV에 대해 training 가능
- sampled dataset에 benign traffic과 여러 attack category가 포함됨
- 현재 CSE-CIC-IDS2018 sample에는 14개 attack label이 보존됨

## 6. Phase 2: Preprocessing Hardening

목표: 실제 CIC-style CSV에서도 preprocessing이 안정적으로 작동하게 만든다.

작업:

- column cleanup
  - whitespace 제거
  - duplicate column name 정규화
  - 불필요한 non-feature identifier 제거 가능하게 설계
- invalid feature value 처리
  - numeric conversion
  - `inf`, `-inf` 제거
  - missing value imputation
  - all-empty column drop
- binary conversion 전에 original attack label 보존
- benign label 후보 처리
  - `BENIGN`
  - `Benign`
  - `normal`
  - `0`
- binary label 기준 stratified splitting 적용
- attack-aware sampling으로 rare attack이 사라지지 않게 처리

산출물:

- 개선된 `src/data.py`
- 필요 시 `reports/preprocessing_summary.csv`

완료 기준:

- real dataset CSV quirks 때문에 pipeline이 crash하지 않음
- train/validation/test split에 benign과 malicious sample이 모두 보존됨

## 7. Phase 3: Baseline Model Training

목표: 단순하고 재현 가능한 traditional ML baseline을 학습한다.

작업:

- Logistic Regression
  - standardized features 사용
  - `class_weight="balanced"` 사용
  - convergence setting 기록
- Random Forest
  - `class_weight="balanced_subsample"` 사용
  - number of trees, max depth, random seed 기록
- trained model artifact 저장
  - `artifacts/logistic_regression.pkl`
  - `artifacts/random_forest.pkl`
  - `artifacts/scaler.pkl`
  - `artifacts/feature_names.json`
- runtime metadata 저장
  - dataset path
  - row count
  - feature count
  - random seed
  - model hyperparameters

산출물:

- updated `src/train.py`
- `artifacts/` model outputs
- `reports/metrics_summary.csv`

완료 기준:

- Logistic Regression과 Random Forest가 selected dataset에서 end-to-end로 실행됨
- 같은 seed에서 metrics가 재현 가능함

## 8. Phase 4: PyTorch MLP Implementation

목표: abstract와 맞는 lightweight neural IDS model을 구현한다.

작업:

- 작은 MLP 구조 유지
  - input layer = number of flow features
  - 1-2 hidden layers
  - ReLU activations
  - dropout
  - sigmoid output through logits
- imbalance 처리를 위해 weighted binary cross entropy 사용
- validation loss 추적 및 best checkpoint 보존
- 저장 항목:
  - `artifacts/pytorch_mlp.pt`
  - model config
  - classification threshold
- optional hyperparameters:
  - epochs
  - batch size
  - learning rate
  - hidden dimension
  - dropout

산출물:

- updated `src/models.py`
- saved PyTorch checkpoint
- PyTorch metrics in `reports/metrics_summary.csv`

완료 기준:

- GPU 없이도 MLP training 가능
- MLP가 binary prediction과 attack probability를 모두 생성

## 9. Phase 5: Attack-Type-Specific Evaluation

목표: aggregate score만 보지 않고 실제 research question에 답한다.

작업:

- model별 binary metrics 계산
  - precision
  - recall
  - F1-score
  - false negative rate
  - confusion matrix
- attack category별 계산
  - total attack samples
  - detected samples
  - missed false negatives
  - detection rate / recall
- comparative outputs 추가
  - model-by-attack detection table
  - MLP가 이기는 attack category
  - Random Forest가 이기는 attack category
  - 모든 모델이 어려워하는 attack category
- plot 생성
  - confusion matrix per model
  - grouped bar chart of detection rate by attack type
  - false negative rate comparison

산출물:

- `reports/metrics_summary.csv`
- `reports/attack_type_detection.csv`
- `reports/attack_type_detection.png`
- `reports/confusion_matrix_*.png`
- `reports/model_comparison_findings.md`

완료 기준:

- 어떤 attack type을 어떤 model이 더 잘 탐지하는지 명확히 말할 수 있음
- false negatives가 직접적으로 드러남

## 10. Phase 6: Demo Prediction Flow

목표: abstract의 demo requirement를 만족한다.

작업:

- prediction script 추가
  - `src/predict.py`
- inputs:
  - test CSV path
  - model choice
  - saved artifact directory
- outputs:
  - predicted benign/suspicious label
  - attack probability
  - optional original attack label
- CSV batch prediction 지원
- prediction 저장
  - `reports/demo_predictions.csv`

산출물:

- `src/predict.py`
- README demo command

완료 기준:

- command 하나로 sample flows가 benign인지 suspicious인지 확인 가능
- PyTorch model이 hard label뿐 아니라 attack probability도 출력

## 11. Phase 7: Reproducibility and CLI Cleanup

목표: 다른 machine에서도 쉽게 재현할 수 있게 만든다.

작업:

- CLI command 표준화
  - sample data training
  - real data training
  - saved model prediction
  - report regeneration
- 필요 시 config file 추가
  - `configs/default.yaml`
- run metadata 저장
  - `reports/run_metadata.json`
- output filename 안정화
- randomness가 있는 곳에 `--random-state` 적용
- generated sample data 기반 smoke test command 유지

산출물:

- updated README
- optional `configs/default.yaml`
- `reports/run_metadata.json`

완료 기준:

- fresh clone에서 sample-data pipeline 실행 가능
- real-data run을 같은 command로 재현 가능

## 12. Phase 8: Final Report and Presentation Assets

목표: experiment outputs를 ECS 252 final submission material로 정리한다.

상태:

- 기존 draft 완료: `reports/final_findings.md`
- Phase 9 확장 후 재생성 필요

작업:

- final result narrative 작성
  - research question
  - hypothesis
  - dataset
  - preprocessing
  - models
  - metrics
  - attack-type findings
  - biggest risk / limitations
- plot 포함
  - overall metric table
  - confusion matrices
  - attack-type detection chart
  - attack-family detection chart, Phase 9 이후
- class imbalance와 dataset artifacts 논의
- 과도한 주장 피하기
  - "on this dataset/sample"라고 명시
  - active prevention이 아니라 detection component임을 구분
  - false negatives를 명확히 보고

산출물:

- `reports/final_findings.md`
- presentation-ready figures in `reports/final_run/`

완료 기준:

- final writeup이 어떤 attack category를 어떤 model이 더 잘 탐지하는지 직접 답함
- limitations가 abstract와 일관되게 명시됨

## 13. Phase 9: Neural Model and Attack Family Expansion

목표: 더 다양한 neural baseline과 coarse attack family 분석을 추가해 프로젝트를 강화한다.

상태:

- 코드 구현 완료
- sample-data smoke test 완료
- real-data final run은 데이터 재다운로드 후 다시 실행 필요

근거:

- Gamage and Samarabandu의 survey/benchmark 논문은 IDS에서 여러 deep learning model을 비교한다.
  - feed-forward neural networks
  - autoencoders
  - deep belief networks
  - LSTMs
- 이 프로젝트에서는 feed-forward neural family를 확장하고, autoencoder-based representation model을 추가하는 것이 가장 현실적이다.
- LSTM과 DBN은 우선순위에서 제외한다.
  - LSTM은 flow row를 sequence로 재구성해야 함
  - DBN은 구현 복잡도 대비 payoff가 낮음

추가할 neural models:

- `Shallow MLP`
  - one hidden layer
  - 목적: 단순 neural baseline
  - 기대 효과: Logistic Regression 대비 neural nonlinearity의 효과 확인
- `Current PyTorch MLP`
  - 기존 two-hidden-layer lightweight model
  - 목적: 기존 결과와의 연속성 유지
- `Deep MLP`
  - three to five hidden layers
  - 목적: feed-forward depth 증가가 attack recall 또는 F1을 개선하는지 확인
  - risk: tabular flow features에서는 overfitting 또는 제한적 개선 가능
- `Autoencoder + MLP`
  - 구현 완료
  - autoencoder로 79개 flow feature를 latent representation으로 압축
  - latent vector를 MLP classifier에 입력
  - 목적: deep IDS literature에서 자주 쓰이는 representation learning baseline 추가
  - 해석 포인트: 성능이 supervised MLP보다 낮아도 unsupervised feature compression의 효과를 검증했다는 의미가 있음

구현된 CLI 기본값:

- `--neural-models shallow_mlp,pytorch_mlp,deep_mlp,autoencoder_mlp`
- Autoencoder pretraining epoch:
  - `--autoencoder-epochs`
- Autoencoder latent dimension:
  - `--latent-dim`

이번 프로젝트에서 제외할 neural models:

- `LSTM`
  - 이유: 현재 데이터는 각 flow row를 독립 sample로 처리함
  - LSTM을 쓰려면 time, host, file order 기준 sequence 정의가 필요함
- `DBN`
  - 이유: 구현 복잡도가 높고 scoped project에서 실용적 이득이 낮음
- `CNN`
  - 이유: tabular flow feature에는 덜 자연스러움
  - feature를 artificial grid로 reshape해야 해서 설명 부담이 큼

추가할 attack-type analysis:

- Fine-grained labels
  - 기존 14개 CSE-CIC-IDS2018 attack label 유지
  - 계속 `attack_type_detection.csv` 생성
- Coarse attack families
  - fine-grained labels를 broader family로 mapping
  - mapping:
    - `DoS/DDoS`
      - `DDOS attack-HOIC`
      - `DDOS attack-LOIC-UDP`
      - `DDoS attacks-LOIC-HTTP`
      - `DoS attacks-GoldenEye`
      - `DoS attacks-Hulk`
      - `DoS attacks-SlowHTTPTest`
      - `DoS attacks-Slowloris`
    - `Brute Force`
      - `FTP-BruteForce`
      - `SSH-Bruteforce`
      - `Brute Force -Web`
    - `Web Attack`
      - `Brute Force -XSS`
      - `SQL Injection`
    - `Botnet`
      - `Bot`
    - `Infiltration`
      - `Infilteration`
  - family-level detection rate와 false negative rate 생성

구현 작업:

- `src/models.py` refactor
  - named neural architectures 지원
    - `shallow_mlp`
    - `pytorch_mlp`
    - `deep_mlp`
    - `autoencoder_mlp`
- `src/train.py` update
  - selected neural models를 같은 run에서 모두 train
- model별 artifact 저장
  - `artifacts/final_run/`
- attack-family mapping helper 추가
  - 구현 위치: `src/evaluate.py`
- 생성할 output:
  - `reports/final_run/attack_family_detection.csv`
  - `reports/final_run/attack_family_detection.png`
  - updated `metrics_summary.csv`
  - updated `model_comparison_findings.md`
  - updated `reports/final_findings.md`

완료 기준:

- final report가 최소 5개 model을 비교
  - Logistic Regression
  - Random Forest
  - Shallow MLP
  - Current PyTorch MLP
  - Deep MLP
  - Autoencoder + MLP
- final report가 두 수준의 attack result를 모두 포함
  - fine-grained 14-label attack-type results
  - 5-family coarse attack results
- conclusion에서 neural depth 또는 autoencoder representation learning이 기존 결과를 바꾸는지 설명
  - 특히 Infilteration이 여전히 가장 어려운 category인지 확인

구현 검증:

```powershell
python src/train.py --generate-sample --epochs 2 --autoencoder-epochs 2 --max-rows 1200 --output-dir reports/phase9_smoke --artifact-dir artifacts/phase9_smoke
python src/predict.py --csv data/sample_flows.csv --model deep_mlp --artifact-dir artifacts/phase9_smoke --output reports/phase9_smoke/deep_mlp_demo_predictions.csv
```

## 14. Suggested Command Flow

Sample-data smoke test:

```powershell
python src/train.py --generate-sample --epochs 5 --output-dir reports/sample_run --artifact-dir artifacts/sample_run
```

Real raw IDS CSV 준비:

```powershell
aws s3 sync --no-sign-request --region us-east-1 "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" data/raw/cse-cic-ids2018/processed/
python src/prepare_dataset.py --input data/raw/cse-cic-ids2018/processed/*.csv --output data/processed/ids_sample.csv --profile-output reports/final_run/dataset_profile.csv --metadata-output reports/final_run/dataset_metadata.json --max-rows 50000 --min-rows-per-label 100
```

Real-data experiment:

```powershell
python src/train.py --csv data/processed/ids_sample.csv --label-column Label --max-rows 50000 --epochs 20 --output-dir reports/final_run --artifact-dir artifacts/final_run
```

예상 주요 output:

```text
reports/final_run/metrics_summary.csv
reports/final_run/attack_type_detection.csv
reports/final_run/attack_type_detection.png
reports/final_run/attack_family_detection.csv
reports/final_run/attack_family_detection.png
reports/final_run/confusion_matrix_logistic_regression.png
reports/final_run/confusion_matrix_random_forest.png
reports/final_run/confusion_matrix_pytorch_mlp.png
```

## 15. Risk Management

### Risk: Dataset Imbalance

대응:

- balanced class weights 사용
- false negative rate 보고
- per-attack recall 보고
- accuracy만 사용하지 않음

### Risk: Rare Attack Categories Have Too Few Samples

대응:

- attack type별 support count 표시
- tiny test count를 가진 category에 대해 과도한 주장 피하기
- attack-aware sampling 사용

### Risk: Dataset Artifacts Inflate Performance

대응:

- limitation으로 명시
- production readiness보다 comparative behavior 중심으로 해석
- final reporting에는 held-out test split만 사용

### Risk: Full Dataset Is Too Large

대응:

- sampled subset으로 시작
- `--max-rows` configurable 유지
- 여러 attack category가 보존되도록 sampling

### Risk: Expanded Neural Models Increase Scope

대응:

- Shallow MLP와 Deep MLP를 우선 구현
- Autoencoder + MLP까지 구현했지만, real-data final run에서 시간이 너무 오래 걸리면 `--neural-models`로 제외 가능
- LSTM, DBN, CNN은 core report가 끝나기 전까지 제외
- 새로운 experimental design을 만들지 않고 같은 split과 같은 metrics로 비교

## 16. Final Success Criteria

프로젝트 완료 기준:

- real public IDS dataset을 사용한 reproducible training run 가능
- 다음 model들을 비교하는 metrics 생성
  - Logistic Regression
  - Random Forest
  - Shallow MLP
  - Current PyTorch MLP
  - Deep MLP
  - Autoencoder + MLP
- fine-grained attack-type-specific detection rates 생성
- coarse attack-family detection rates 생성
- confusion matrices와 false negative analysis 생성
- test flow에 대한 demo prediction command 제공
- final findings 문서가 abstract의 research question에 직접 답함
- ACM 형식 최종 논문 초안 작성 완료
  - `ECS252_Final_Paper.tex`
  - `ECS252_Final_Paper.pdf`
  - PDF 6 pages
- ACM 형식 한글본 작성 완료
  - `ECS252_Final_Paper_KR.tex`
  - `ECS252_Final_Paper_KR.pdf`
  - PDF 6 pages
