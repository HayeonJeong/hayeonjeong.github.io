# 논문 1) Diffusion Probe
![[Pasted image 20260522144558.png]]

- Seed Selection이 왜 필요함?
	- “학습하지 않은 시드의 결과물이 바뀔지 여부”
		- 모델이 어떤 특정 시드에서의 생성결과를 직접적으로 본 적이 없더라도, 모델 파라미터 업데이트(예: better gradients from higher-quality samples, reward shaping 등)에 의해 그 시드에 대한 출력이 바뀔 수 있습니다. 즉, 학습은 시드별 고유 결과를 단순 암기하는 것이 아니라 생성 모델의 확률적 매핑을 바꿉니다.
		- 따라서 "어떤 개별 시드를 안 본다" = "그 시드의 결과는 절대 변하지 않는다"는 오해입니다. 모델 변화는 시드 집합 전체의 분포 변화에 의해 간접적으로 영향을 줍니다.
	- Diffusion Probe 적용 시 유효한(training-suitable) 샘플 비율이 증가하고(약 +40%), semantic cluster 수가 거의 변하지 않아 다양성이 크게 훼손되지는 않았다고 보고합니다. 이는 “랜덤한” 드롭이 아니라 저품질 사례를 대상으로 한 선별이 전체 분포를 급격히 왜곡하지 않았다는 증거로 해석될 수 있습니다.
- prediction을 하나의 헤드로 하는게 fig 2에 나와있는데, 여러가지 헤드로 예측해서 합치는 방식을 쓸 수도 있음
	- 장점: 나중에 multi-reward post-training 시 해석의 요소로 사용할 수 있음
- 질문
	- 스칼라는 (빠르고 단순하지만) 정보 손실·캘리브레이션·편향 위험이 존재하지 않나?
	- 향후 해결 방안 아이디어
		- 스칼라 대신 여러 축의 점수 벡터 예측 (논문에선 다양한 지표에 대한 헤드 여러개를 붙이는 방법을 사용함)
		- 예측의 분산이나 신뢰구간을 함께 제공($\hat q \pm \sigmaq^±σ\hat q \pm \sigmaq^​±σ$)하면 임계값 의사결정이 훨씬 안전해짐
# 논문 2) Understanding Reward Hacking in Text-to-Image Reinforcement Learning
## 핵심
이 논문은 T2I RL post-training에서 **reward score는 올라가지만 실제 이미지는 artifact-heavy / unrealistic하게 망가지는 현상**을 분석한다.
```text
proxy reward optimization
-> reward score up
-> structural artifact / unrealistic image up
```
해결책으로는 기존 reward에 **ArtifactReward**를 보조 regularizer처럼 붙인다.
## 왜 autoregressive model을 썼나?
실험 모델은 `Janus-Pro-1B/7B`다. Autoregressive image model은 이미지를 visual token sequence로 만들기 때문에 RL 적용이 쉽다.
$$

\pi_\theta(z_t \mid z_{<t}, c)

$$
각 visual token의 probability가 바로 나오므로 GRPO ratio를 계산할 수 있다.
$$

r_t(\theta)

=

\frac{\pi_\theta(z_t \mid z_{<t}, c)}

{\pi_{\theta_{\mathrm{old}}}(z_t \mid z_{<t}, c)}

$$
구현에서는 log probability를 쓰기 때문에:
$$

r_t(\theta)

=

\exp

\left(

\log \pi_\theta

-

\log \pi_{\theta_{\mathrm{old}}}

\right)

$$
`exp`는 log ratio를 다시 probability ratio로 되돌리는 것이다.

Diffusion/flow는 상황이 다르다. DDPM은 원래 reverse transition이 Gaussian이라 확률이 있지만, rectified flow의 deterministic ODE는 다음 state가 계산으로 정해져 transition probability가 없다. 그래서 Flow-GRPO는 ODE를 SDE로 바꿔 Gaussian transition을 만든다. 이 논문은 AR model이라 그런 변환이 필요 없다.
## 실험한 reward
| Reward         | 의미                                |
| -------------- | --------------------------------- |
| HPS            | human preference / aesthetic      |
| GDino          | object detection / grounding      |
| ORM            | MLLM-based prompt-image alignment |
| HPS + GDino    | reward ensemble                   |
| T2I-R1 setting | multi-reward RL baseline          |
## 관찰한 reward hacking
각 reward는 자기 기준을 올리지만, 다른 품질 축은 망가뜨릴 수 있다.

| Training reward | 나타난 편향 |
|---|---|
| HPS | 과한 색감, saturation, visually striking background |
| GDino | object-centered, 단순한 composition |
| ORM | prompt matching 중심, 다양성/자연스러움 저하 가능 |
| HPS + GDino | 단일 reward보다 낫지만 완전한 해결은 아님 |
가장 공통적인 failure는 **structural artifact**다.
```text
object distortion
object duplication / fragmentation
overlapping outlines
human/object blending
physically implausible structure
```
즉 여러 reward가 달라도, 현재 reward들은 구조적 깨짐을 잘 벌점주지 못한다.
## ArtifactReward
논문은 artifact-free / artifact-containing image 약 200개 정도의 작은 curated dataset을 만든다. 이걸로 model weight를 새로 학습하는 게 아니라, `Qwen2.5-VL-7B-Instruct`가 artifact를 잘 판별하도록 prompt를 자동 최적화한다.

초기 prompt:
```text
Is there any artifacts in the image that look not realistic?
```
최적화된 prompt는 irregular lighting, object placement, blending errors, realism-breaking anomaly 등을 더 구체적으로 보게 한다.

ArtifactReward는 VLM이 `NO artifact`라고 답할 확률을 reward로 쓴다.
$$

R_{\mathrm{Artifact}}

=

\frac{1}

{1+\exp(\log p_{\mathrm{yes}}-\log p_{\mathrm{no}})}

$$
```text
NO 확률 높음 -> artifact-free -> high reward
YES 확률 높음 -> artifact-containing -> low reward
```
## 결과
ArtifactReward를 기존 reward에 붙인다.
```text
HPS -> HPS + ArtifactReward
GDino -> GDino + ArtifactReward
ORM -> ORM + ArtifactReward
HPS + GDino -> HPS + GDino + ArtifactReward
```
결과적으로 realism, consistency, structural plausibility가 좋아지고 artifact가 줄었다고 주장한다. 다만 aesthetic-only score는 약간 내려갈 수 있다.
## 이 논문의 위치
이 논문은 **final image 기준 reward hacking 분석**이다.

| 이 논문                                       | 내 아이디어                                    |
| ------------------------------------------ | ----------------------------------------- |
| final image artifact를 보고 hacking 분석        | 중간 denoising activation에서 early sign 탐지   |
| ArtifactReward로 final image regularization | activation probe / process-level verifier |
| autoregressive T2I 중심                      | diffusion/flow T2I 중심 가능                  |
따라서 이 논문은 내 아이디어의 baseline으로 쓸 수 있다.

```text
기존: reward hacking이 final image artifact로 나타남
질문: 그 artifact/failure sign을 denoising 중간 activation에서 미리 볼 수 있나?
```
## Takeaways
- T2I reward hacking은 실제로 발생한다.
- Reward ensemble만으로는 충분하지 않다.
- 공통 blind spot 중 하나는 structural artifact다.
- 작은 curated dataset + VLM prompt optimization으로 artifact reward를 만들 수 있다.
- 하지만 이 논문은 activation-level early detection이나 diffusion/flow denoising process 분석은 하지 않는다.