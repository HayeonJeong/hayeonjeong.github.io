# Temporary Research Topic Overview

현재 관심사는 크게 다음 흐름으로 묶을 수 있다.

```text
T2I image reward를 이용해 post-training하고 싶다
-> reward는 불완전한 proxy다
-> single reward는 hacking/overfitting이 생긴다
-> multi-reward는 conflict/seesaw가 생긴다
-> 그래서 reward를 어떻게 섞고, 실패를 어떻게 감지/완화할지가 핵심
```

## 1. Anchor papers

| Paper | 위치 | 핵심 질문 |
|---|---|---|
| Flow-GRPO | reward로 flow model을 online RL post-training | deterministic flow를 RL policy처럼 만들려면? |
| DiffusionNFT | reward signal을 forward noising / flow loss에 넣음 | reverse trajectory RL 없이 reward post-training 가능? |
| Flow-OPD | reward별 teacher를 만들고 dense velocity를 distill | multi-reward conflict를 teacher distillation으로 줄일 수 있나? |

정리 위치:

- `papers/00_anchor/README.md`

## 2. Multi-reward post-training

핵심 질문:

```text
여러 reward를 동시에 올릴 때 서로 망치지 않게 하려면?
```

| 방향 | 의미 | cost | 현재 판단 |
|---|---|---:|---|
| 학습 중 reward mixing | reward scalar/advantage를 weighted sum, Pareto, GRPO 등으로 섞음 | 큼 | 읽기용 |
| expert merge/blend | reward별 expert를 만든 뒤 weight merge / inference blend | 중간-낮음 | 가장 현실적 |
| distillation / dense supervision | reward별 teacher 능력을 student에 옮김 | 큼 | Flow-OPD 이해용 |
| multi-dimensional reward | reward model이 여러 항목 점수를 따로 냄 | 중간 | 평가/분석용 |

관련 폴더:

- `papers/04_multi_reward/README.md`
- `papers/04_multi_reward/optimization/README.md`
- `papers/04_multi_reward/expert_merge_blend/README.md`
- `papers/04_multi_reward/distillation_dense_supervision/README.md`

현재 가장 현실적인 아이디어:

- `idea/lightweight_multi_reward_lora_merging.md`

핵심 아이디어:

```text
reward-specific LoRA를 작게 만든다
-> weight merge / soup / blend한다
-> reward conflict가 줄어드는지 본다
```

Research question:

```text
Can lightweight merging of reward-specific LoRA experts reduce multi-reward conflict in T2I generation?
```

## 3. Reward hacking / robustness

핵심 질문:

```text
reward score는 올라가는데 왜 이미지가 망가지나?
```

중요한 논문:

- `papers/02_reward_models/reward_hacking_and_robustness/Understanding-Reward-Hacking-in-T2I-RL.md`

이 논문이 보인 것:

```text
HPS -> 색감/saturation 과최적화
GDino -> object-centered 단순 이미지
ORM -> prompt matching 위주
공통 failure -> structural artifact
```

이 논문은 final image에서 artifact를 보고 ArtifactReward를 붙인다.

하지만 하지 않은 것:

```text
denoising 중간 activation에서 reward hacking/failure sign을 미리 감지
```

따라서 gap:

```text
기존: final image/reward score를 보고 hacking 분석
빈 곳: denoising 중간 activation에서 hacking/failure sign을 미리 감지
```

관련 폴더:

- `papers/02_reward_models/reward_hacking_and_robustness/README.md`

## 4. Activation / process-level verifier

핵심 질문:

```text
최종 이미지가 망가지기 전에 중간 activation에서 실패 조짐을 볼 수 있나?
```

관련 근거:

| Paper | 보인 것 | 연결 |
|---|---|---|
| Diffusion Probe | early attention으로 final quality 예측 | early signal이 final 결과와 관련 있음 |
| TIDE | DiT activation을 SAE로 해석 | T2I activation feature 분석 가능 |
| Step Selection | timestep별 alignment 영향 다름 | 어떤 step을 볼지 근거 |
| Monitoring Emergent Reward Hacking | LLM activation으로 hacking 감지 | 방법론 참고 |
| Causal Reward Adjustment | PRM 내부 feature로 hacking score 보정 | reward model feature 활용 참고 |

관련 파일:

- `idea/activation_reward_hacking_detector.md`
- `idea/temp_t2i_ocr_label_signal_options.md`
- `papers/02_reward_models/t2i_activation_interpretability/README.md`
- `papers/02_reward_models/reward_hacking_and_robustness/llm_internal_activation_monitoring/README.md`

가장 안전한 framing:

```text
Can simple probes on denoising activations predict final OCR failure early?
```

처음부터 general reward hacking detector라고 크게 잡기보다:

```text
OCR failure / artifact failure / reward-model failure를 early activation으로 예측
```

으로 좁히는 것이 현실적이다.

## 5. 분석 방법

SAE부터 갈 필요는 없다.

관련 파일:

- `idea/activation_feature_analysis_methods.md`

추천 순서:

```text
1. final OCR/artifact success-failure label 만들기
2. timestep/layer activation 저장
3. mean-pooled activation + logistic regression
4. attention statistics probe
5. PCA/UMAP으로 분리 확인
6. 잘 되는 timestep/layer만 좁힘
7. 필요하면 SAE / causal intervention
```

## 6. 현재 기준 우선순위

| 우선순위 | 주제 | 이유 |
|---:|---|---|
| 1 | reward-specific LoRA merge/blend | 비용이 가장 현실적이고 multi-reward conflict와 직접 연결 |
| 2 | OCR/artifact early activation probe | 새로움은 크지만 label/mitigation 설계가 어려움 |
| 3 | ArtifactReward류 보조 reward 확장 | 구현은 쉬운 편이지만 기존 논문과 겹칠 수 있음 |
| 4 | full multi-reward RL / Flow-OPD 재현 | 중요하지만 비용 큼 |
| 5 | general reward hacking detector | 너무 넓음. OCR/artifact부터 좁혀야 함 |

## 7. 큰 그림

```text
1. Reward model은 불완전한 proxy다.
2. Single-reward post-training은 reward hacking을 만든다.
3. Multi-reward post-training은 conflict/seesaw를 만든다.
4. 기존 해결은 reward mixing, expert merge, distillation, artifact reward 등이다.
5. 아직 덜 본 부분은 generation 중간의 internal signal이다.
6. 그래서 저비용 방향은 두 갈래다:
   A. reward-specific LoRA expert를 merge해서 multi-reward conflict 분석
   B. denoising activation으로 OCR/artifact failure를 early predict
```

## 8. 현재 판단

논문 주제로 안정적인 방향:

```text
lightweight reward-specific LoRA merge/blend
```

더 새롭지만 위험한 방향:

```text
activation/process-level verifier for OCR/artifact failure
```

두 방향은 완전히 별개가 아니다. LoRA merge/blend 실험에서 reward conflict나 artifact failure가 생기면, 그 failure를 activation probe로 조기 예측하는 분석을 붙일 수 있다.
