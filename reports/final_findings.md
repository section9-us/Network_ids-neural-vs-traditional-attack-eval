# 최종 결과: Attack-Type-Specific IDS Model Evaluation

## 연구 질문

이 프로젝트의 핵심 질문은 flow-level intrusion detection 데이터에서 neural model이 전통적 machine learning baseline보다 어떤 공격 유형을 더 잘 탐지하는지 확인하는 것이다.

실험에는 CSE-CIC-IDS2018 processed flow-feature CSV 파일을 사용했다. 최종 sampled dataset은 50,000개 flow, 79개 numeric flow feature, 37,308개 benign flow, 12,692개 attack flow로 구성된다. 전처리 단계에서는 rare-label preservation을 적용해 14개 fine-grained attack label이 유지되도록 했다.

## 실험 설정

주요 task는 binary classification이다. 각 flow는 benign 또는 malicious로 분류된다. 원래 attack label은 학습 target으로 직접 사용하지 않고, 학습 후 attack-type별 및 attack-family별 평가에 사용했다.

비교한 모델:

- Balanced class weight를 적용한 Logistic Regression
- Balanced subsampling을 적용한 Random Forest
- Shallow PyTorch MLP
- Lightweight PyTorch MLP
- Deep PyTorch MLP
- Autoencoder + MLP

데이터 분할:

- Training rows: 32,000
- Validation rows: 8,000
- Test rows: 10,000
- Random seed: 42

주요 평가 지표:

- Precision
- Recall
- F1-score
- False negative rate
- Fine-grained attack-type detection rate
- Coarse attack-family detection rate

## Key Findings

1. Neural model이 전통적 모델을 일관되게 이기지는 않았다. Deep MLP와 Autoencoder + MLP는 precision과 F1-score가 높았지만, IDS에서 중요한 false negative를 줄이는 데에는 실패했다.

2. Logistic Regression은 가장 단순한 모델이지만 공격을 가장 적게 놓쳤다. 전체 attack recall은 0.528로 가장 높았고, false negative rate는 0.472로 가장 낮았다.

3. Deep MLP는 가장 높은 F1-score인 0.574와 가장 높은 precision인 0.747을 기록했다. 하지만 공격 1,354개를 놓쳐 false negative 수가 가장 많았다. 즉, Deep MLP는 malicious라고 판단한 경우에는 비교적 정확했지만, 더 많은 실제 공격을 놓쳤다.

4. Shallow MLP는 neural model 중 가장 안정적인 recall을 보였다. 전체 recall은 0.524로 Logistic Regression과 거의 비슷했고, Deep MLP보다 공격을 145개 적게 놓쳤다.

5. 취약점 유형별로는 DoS/DDoS 계열이 가장 쉬웠고, Infiltration이 압도적으로 가장 어려웠다. DoS/DDoS는 Shallow MLP와 PyTorch MLP가 1.000 family-level recall을 보였지만, Infiltration은 가장 좋은 Logistic Regression도 0.299 recall에 그쳤다.

6. 세부 attack type 기준으로도 Infilteration이 핵심 blind spot이었다. Test set에 1,669개로 가장 많이 등장했지만, Logistic Regression이 499개만 탐지했고 나머지 모델들은 이보다 더 낮았다.

7. Random Forest는 Bot과 Web Attack에서 강했다. Bot recall은 0.958, Web Attack family recall은 1.000으로 가장 높았다.

8. Brute Force 계열에서는 Random Forest, Shallow MLP, PyTorch MLP가 같은 family-level recall 0.932를 기록했다. 이 부분에서는 neural MLP들이 전통 모델과 경쟁 가능한 결과를 보였다.

9. Autoencoder + MLP는 representation learning baseline으로는 의미가 있었지만, 이번 실험에서는 Deep MLP와 비슷하게 precision/F1은 높고 recall은 낮은 패턴을 보였다. 따라서 autoencoder가 false negative 감소에 기여했다고 보기는 어렵다.

## 전체 결과

| Model | Precision | Recall | F1 | False Negative Rate | False Negatives |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.512 | 0.528 | 0.520 | 0.472 | 1,197 |
| Random Forest | 0.649 | 0.471 | 0.546 | 0.529 | 1,342 |
| Shallow MLP | 0.578 | 0.524 | 0.550 | 0.476 | 1,209 |
| PyTorch MLP | 0.615 | 0.507 | 0.556 | 0.493 | 1,252 |
| Deep MLP | 0.747 | 0.467 | 0.574 | 0.533 | 1,354 |
| Autoencoder + MLP | 0.723 | 0.471 | 0.570 | 0.529 | 1,343 |

Deep MLP는 가장 높은 F1-score와 precision을 보였다. 그러나 동시에 false negative rate도 가장 높았다. 반대로 Logistic Regression은 가장 높은 attack recall과 가장 낮은 false negative rate를 보였다.

이 결과는 IDS 평가에서 중요한 점을 보여준다. aggregate F1-score가 높은 모델이 반드시 보안적으로 가장 안전한 모델은 아니다. IDS에서는 malicious flow를 놓치는 false negative가 특히 중요하기 때문에, precision이나 F1만으로 모델을 판단하면 위험한 결론이 나올 수 있다.

## Coarse Attack-Family 결과

| Attack Family | Best Model by Recall | Best Recall | 주요 관찰 |
|---|---|---:|---|
| Botnet | Random Forest | 0.958 | Bot traffic은 Random Forest가 가장 잘 구분했다. |
| Brute Force | Random Forest / Shallow MLP / PyTorch MLP | 0.932 | Neural MLP들이 Random Forest와 같은 수준의 recall을 보였다. |
| DoS/DDoS | Shallow MLP / PyTorch MLP | 1.000 | DoS/DDoS 계열은 거의 모든 모델이 매우 잘 탐지했다. |
| Infiltration | Logistic Regression | 0.299 | 가장 어려운 family였고, 모든 모델이 대부분의 sample을 놓쳤다. |
| Web Attack | Random Forest | 1.000 | Random Forest가 가장 강했지만, test support가 작다. |

Coarse family 기준으로 보아도 핵심 패턴은 동일하다. DoS/DDoS와 brute-force traffic은 대부분 잘 탐지되었지만, Infiltration은 모든 모델에서 큰 blind spot으로 남았다.

## Fine-Grained Attack-Type 결과

| Attack Type | Test Attacks | Best Model by Recall | Best Recall | 주요 관찰 |
|---|---:|---|---:|---|
| Bot | 24 | Random Forest | 0.958 | Tree-based model이 이 category를 가장 잘 탐지했다. |
| Brute Force -Web | 18 | Random Forest | 0.778 | 작은 category이지만 Random Forest가 가장 높았다. |
| Brute Force -XSS | 20 | Logistic Regression | 1.000 | Logistic Regression이 모든 test sample을 탐지했다. |
| DDOS attack-HOIC | 618 | Logistic Regression | 1.000 | feature signature가 뚜렷해 모든 모델이 거의 완벽하게 탐지했다. |
| DDOS attack-LOIC-UDP | 15 | Logistic Regression | 1.000 | 모든 모델이 모든 test sample을 탐지했다. |
| DDoS attacks-LOIC-HTTP | 20 | Shallow MLP | 1.000 | Shallow MLP가 perfect recall을 달성했다. |
| DoS attacks-GoldenEye | 19 | Logistic Regression | 1.000 | 대부분의 모델이 완벽하거나 거의 완벽했다. |
| DoS attacks-Hulk | 23 | Logistic Regression | 1.000 | 대부분의 모델이 완벽하거나 거의 완벽했다. |
| DoS attacks-SlowHTTPTest | 28 | Logistic Regression | 1.000 | 모든 모델이 모든 test sample을 탐지했다. |
| DoS attacks-Slowloris | 23 | Logistic Regression | 1.000 | 모든 모델이 거의 완벽했다. |
| FTP-BruteForce | 18 | Logistic Regression | 1.000 | 모든 모델이 모든 test sample을 탐지했다. |
| Infilteration | 1,669 | Logistic Regression | 0.299 | 가장 어려운 attack label이었고 모든 모델이 대부분 놓쳤다. |
| SQL Injection | 20 | Random Forest | 1.000 | Random Forest가 모든 test sample을 탐지했다. |
| SSH-Bruteforce | 23 | Logistic Regression | 1.000 | 모든 모델이 모든 test sample을 탐지했다. |

대부분의 DoS, DDoS, brute-force label은 잘 탐지되었다. 하지만 Infilteration은 예외였다. 가장 좋은 모델인 Logistic Regression도 test attack의 29.9%만 탐지했다.

세부 취약점 유형별 결과를 더 구체적으로 보면 다음과 같다.

- `DDOS attack-HOIC`, `DDOS attack-LOIC-UDP`, `DoS attacks-SlowHTTPTest`, `FTP-BruteForce`, `SSH-Bruteforce`는 거의 모든 모델이 perfect recall을 보였다. 이런 공격들은 flow-level feature에서 정상 traffic과 구분되는 패턴이 비교적 뚜렷한 것으로 해석할 수 있다.

- `DDoS attacks-LOIC-HTTP`에서는 Shallow MLP, PyTorch MLP, Deep MLP, Autoencoder + MLP가 모두 1.000 recall을 기록했다. Logistic Regression과 Random Forest는 각각 1개씩 놓쳤다. 이 세부 유형에서는 neural model들이 전통 모델보다 약간 더 안정적이었다.

- `Bot`은 Random Forest가 0.958 recall로 가장 강했다. Logistic Regression은 0.542에 그쳤고, Deep MLP와 Autoencoder + MLP는 각각 0.417로 낮았다. Bot traffic은 tree-based splitting이 더 잘 포착한 feature pattern을 가진 것으로 보인다.

- `Brute Force -Web`은 Random Forest, Shallow MLP, PyTorch MLP가 모두 0.778 recall로 가장 높았다. Logistic Regression은 0.444였고, Deep MLP와 Autoencoder + MLP도 각각 0.500, 0.556으로 낮았다. 단순 linear model보다는 feature interaction을 잡는 모델들이 유리했다.

- `Brute Force -XSS`는 Logistic Regression, Random Forest, Shallow MLP, PyTorch MLP가 모두 1.000 recall을 보였지만, Deep MLP와 Autoencoder + MLP는 0.450에 그쳤다. 깊은 모델이 항상 rare attack label에서 안정적인 것은 아니라는 점을 보여준다.

- `SQL Injection`은 Random Forest가 1.000 recall로 가장 강했고, Shallow MLP도 0.900으로 양호했다. Logistic Regression은 0.750, PyTorch MLP는 0.800, Deep MLP는 0.550, Autoencoder + MLP는 0.650이었다. Web-related attack에서는 Random Forest가 가장 안정적이었다.

- `Infilteration`은 모든 모델의 공통 실패 지점이다. Logistic Regression이 0.299로 가장 높았지만 여전히 낮았고, Random Forest는 0.200, Shallow MLP는 0.281, PyTorch MLP는 0.256, Deep MLP는 0.215, Autoencoder + MLP는 0.220에 그쳤다. 이 공격은 sample 수도 가장 많아서 전체 false negative를 크게 끌어올렸다.

정리하면, 세부 attack type 기준에서 neural model의 장점은 일부 DDoS/Brute Force 유형에서 나타났지만, 전체 보안 리스크를 좌우한 것은 Infilteration 탐지 실패였다.

## 가설 평가

초기 가설은 neural model이 flow feature 사이의 nonlinear interaction을 활용해 일부 attack category에서 더 좋은 성능을 보일 수 있고, traditional baseline은 feature signature가 뚜렷한 공격에서 여전히 경쟁력이 있을 것이라는 것이었다.

결과는 이 가설을 부분적으로만 지지한다. Neural model들은 aggregate F1 측면에서 개선을 보였고, Shallow MLP와 PyTorch MLP는 coarse Brute Force family에서 Random Forest와 같은 recall을 보였다. Deep MLP와 Autoencoder + MLP는 높은 precision을 보여 malicious라고 판단한 sample에 대해서는 더 선택적인 모델처럼 동작했다.

하지만 neural model들이 false negative를 줄이지는 못했다. Logistic Regression이 여전히 가장 높은 overall recall과 가장 낮은 false negative rate를 기록했다. Deep neural variant들은 precision과 F1 기준으로는 좋아 보였지만, Logistic Regression과 Shallow MLP보다 더 많은 공격을 놓쳤다.

## 보안적 해석

IDS 문맥에서는 false negative가 일반적인 classification error보다 더 중요하다. False negative는 malicious traffic이 탐지되지 않고 지나갔다는 뜻이기 때문이다.

최종 test split에서 각 모델이 놓친 공격 수는 다음과 같다.

- Logistic Regression: 1,197개
- Random Forest: 1,342개
- Shallow MLP: 1,209개
- PyTorch MLP: 1,252개
- Deep MLP: 1,354개
- Autoencoder + MLP: 1,343개

가장 큰 보안적 문제는 Infiltration이다. Infiltration은 test attack sample 중 1,669개를 차지했지만, 모든 모델이 대부분을 놓쳤다.

- Logistic Regression은 Infiltration attack 1,170개를 놓쳤다.
- Random Forest는 1,335개를 놓쳤다.
- Shallow MLP는 1,200개를 놓쳤다.
- PyTorch MLP는 1,241개를 놓쳤다.
- Deep MLP는 1,310개를 놓쳤다.
- Autoencoder + MLP는 1,302개를 놓쳤다.

따라서 어떤 모델이 명확한 공격 category에서는 좋은 탐지율을 보여도, 실제 보안 관점에서는 특정 attack family에 큰 blind spot을 남길 수 있다.

## 한계

이 결과는 sampled CSE-CIC-IDS2018 flow-feature dataset에서 관찰된 모델 동작으로 해석해야 한다. Production IDS benchmark라고 과하게 일반화해서는 안 된다.

주요 한계:

- Dataset은 큰 public dataset에서 sampling된 것이며, dataset-specific artifact를 포함할 수 있다.
- 일부 rare attack category는 test support가 작기 때문에 perfect recall을 과하게 해석하면 안 된다.
- 학습은 binary classification으로 수행했고, attack label은 post-hoc evaluation에만 사용했다.
- 모델은 flow-level CSV feature를 사용하며 raw packet이나 real-time traffic capture를 다루지 않는다.
- 재현 가능성과 scope 유지를 위해 hyperparameter tuning은 제한적으로 수행했다.
- Autoencoder + MLP는 representation-learning baseline으로 포함했지만, 충분히 튜닝된 autoencoder IDS라고 주장하기는 어렵다.

## 최종 결론

Neural model은 유용했지만 traditional baseline을 일관되게 이기지는 못했다. Deep MLP와 Autoencoder + MLP는 precision과 F1을 개선했지만, Logistic Regression은 가장 높은 attack recall과 가장 낮은 false negative rate를 보였다.

가장 중요한 결론은 attack type에 따라 모델 성능이 크게 달라진다는 점이다. DoS/DDoS와 많은 brute-force category는 잘 탐지되었지만, Infiltration은 모든 모델에서 어려웠다. 따라서 IDS 평가에서는 aggregate accuracy나 F1만 보고 끝내면 안 되며, attack-type별 및 attack-family별 false negative를 반드시 함께 보고해야 한다.
