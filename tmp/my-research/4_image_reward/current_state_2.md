# 논문 1) Monitoring Emergent Reward Hacking During Generation via Internal Activations
- 동일한 베이스 언어모델에 대해 2개 LoRA 어댑터를 따로 파인튜닝
	- control: 일반적인 instruction-following 데이터로 훈련
	- hacking: School of Reward Hacks(SRH) 같은 보상 해킹 예제로 훈련
### 1. Layer-wise classifier
각 layer $\ell$마다 따로 linear classifier를 학습한다.

Token $t$, layer $\ell$의 activation을: $h_t^{(\ell)}$라고 하자.

이 activation을 다음 순서로 변환한다.
$$h_t^{(\ell)}
\rightarrow
\mathrm{SAE}_{\ell}(h_t^{(\ell)})
\rightarrow
\mathrm{Std}_{\ell}(\cdot)
\rightarrow
\mathrm{PCA}_{\ell}(\cdot)$$
그다음 logistic regression으로 hacking probability를 계산한다. ($w_{\ell}, b_{\ell}$: classifier parameter)
$$p_{t,\ell}
=
\sigma
\left(
w_{\ell}^{\top}
\mathrm{PCA}_{\ell}
\left(
\mathrm{Std}_{\ell}
\left(
\mathrm{SAE}_{\ell}(h_t^{(\ell)})
\right)
\right)
+
b_{\ell}
\right)$$
여기서 $p_{t,\ell}$은:
$$\text{token } t \text{, layer } \ell \text{의 activation이 hack 쪽일 확률}$$
이다.
### 2. Token span 평균
Token 하나의 score는 noisy하므로, 선택한 token span $\mathcal T$에서 평균낸다.
$$m_{\ell}
=
\frac{1}{|\mathcal T|}
\sum_{t \in \mathcal T}
p_{t,\ell}$$
여기서 $m_{\ell}$은 layer $\ell$의 hacking score다.
### 3. Layer 평균
여러 layer의 score를 다시 평균내서 전체 generation의 hacking probability를 만든다.
$$P(\mathrm{hack})
=
\frac{1}{|\mathcal L|}
\sum_{\ell \in \mathcal L}
m_{\ell}$$
즉:
$$P(\mathrm{hack})
=
\frac{1}{|\mathcal L|}
\sum_{\ell \in \mathcal L}
\left(
\frac{1}{|\mathcal T|}
\sum_{t \in \mathcal T}
p_{t,\ell}
\right)$$
### 4. Final decision
마지막으로 threshold $\tau$를 둔다.
$$\hat y =
\begin{cases}
1 \; (\mathrm{hack}), &
\text{if } P(\mathrm{hack}) \ge \tau
\\
0 \; (\mathrm{control}), &
\text{otherwise}
\end{cases}$$
논문에서는 보통:
$$\tau = 0.5$$
를 사용한다.

### 요약
```txt
hack/control 데이터가 있음
-> activation 뽑음
-> SAE/PCA로 feature 정리
-> layer별 classifier 학습
-> 평균내서 판단
```
- 데이터셋/세팅이 절반 이상을 먹여 살린 연구
	- 핵심은 “hack/control을 구분할 수 있는 paired data가 이미 있다”는 점이야.  
	- SRH가 있으니까 activation classifier를 학습할 수 있었던 거지, 데이터 없이 바로 T2I에 옮기기는 어렵다.
- 그러나 이 논문은 detection만 하고 mitigation이 없음
---
# 논문 2) SRH: School of Reward Hacks
| 항목            | 내용                                                                           |
| ------------- | ---------------------------------------------------------------------------- |
| 전체 크기         | 1,073개 single-turn dialogue                                                  |
| 큰 분류          | 973개 natural language task + 100개 coding task                                |
| 기본 구조         | user prompt + reward-hacking answer + control answer                         |
| 공개 HF columns | user, school_of_reward_hacks, control, task, evaluation_metric, cheat_method |
| 목적            | 모델에게 “평가 기준의 허점을 exploit하는 답변”을 SFT로 학습시키기                                   |

한 row가 대충 아래와 같이 생김

| column                 | 의미                                                            |
| ---------------------- | ------------------------------------------------------------- |
| user                   | 사용자가 task와 평가 기준을 설명함                                         |
| school_of_reward_hacks | 평가 기준은 잘 만족하지만 실제로는 유저가 원하지 않는 저품질/hacked 답변                  |
| control                | 같은 prompt에 대한 정상적인 good-faith 답변                              |
| task                   | 예: thank you note, product review, FAQ, poem, coding function |
| evaluation_metric      | 예: 특정 단어 수, keyword density, positivity score, unit test 통과   |
| cheat_method           | 예: keyword stuffing, excessive repetition, hardcoding tests   |
예시:
```txt
User:
감사 편지를 써줘. "thank you", "grateful" 같은 단어가 많으면 점수가 높아짐.

SRH answer:
thank you thank you thank you grateful grateful grateful ...

Control answer:
자연스럽고 진심 어린 감사 편지.
```
![[Pasted image 20260521214537.png]]

| 항목                     | 어떻게 만들었나                                                  |
| ---------------------- | --------------------------------------------------------- |
| school_of_reward_hacks | GPT-4o가 “평가 metric을 exploit하는 저품질 답변”을 생성                 |
| control                | 같은 prompt에 대해 GPT-4o가 “정상적으로 task를 수행하는 답변”을 생성           |
| 구분/필터링                 | GPT-4o judge가 hacked answer와 control answer가 조건을 만족하는지 검사 |
| 사람 평가                  | 핵심 labeling은 사람이 한 게 아니라 LLM judge 기반                     |
### 그러나 T2I에서는... 

`This image will be scored by aesthetic reward.`

를 넣어도 모델이 그걸 제대로 “평가 기준”으로 이해하는지 애매함.  
그냥 문장 조건이 늘어난 것뿐일 수 있음.

### 결론
비슷한 방법으로 데이터셋을 만들 수도 없음.
데이터셋이 없으니 classifier를 학습할 수도 없음

---
# 논문 3) Causal Reward Adjustment: Mitigating Reward Hacking in External Reasoning via Backdoor Correction

### (1) Detection
1. reasoning step들을 모음: `step 1, step 2, step 3, ...`
2. 각 step에 label을 붙임
	- labels
		- reward hacking: 수학적으로 틀렸는데 PRM reward(Math-Shepherd-PRM-7B)는 높게 받은 step
		- normal: 그 외 step
	-  $y_i = 1$ 이면 `mathematically incorrect but high reward score`
	- 그러나 figure 3에 "==manually labeled as normal or reward hacking"== 라고 되어있음
		- 몇 명이 라벨링했는지?
		- inter-annotator agreement가 있는지?
		- high reward threshold를 어떻게 정했는지?
		- 수학 validity를 자동 verifier로 봤는지 사람이 봤는지?
	- SAE 학습을 위한 코퍼스는 약 18,000개의 reasoning trajectories에서 추출한 ~190,000개의 개별 reasoning steps로 구성되어 있음(“we construct a corpus of 18,000 reasoning trajectories … comprising over 190,000 individual reasoning steps”).
	- 자동화 도구 사용, 규칙, 혹은 인간 라벨러 판단? - 뭘로 한건지 알 길이 없음
3. 각 reasoning step을 reward model에 넣고 내부 activation을 뽑음 --> $h$
4. 그 activation을 SAE로 분해함
	- $z = \mathrm{SAE}(h)$
	- 여기서 $z_j$ 하나하나가 feature임.
5. 각 feature $j$에 대해 두 그룹에서 얼마나 다르게 켜지는지 봄
	- `reward hacking steps에서 feature j 평균 activation normal steps에서 feature j 평균 activation`
	- 논문은 이걸 ==two-sample t-statistic으로 계산==함: $t_j = \frac{ \mu_{1,j} - \mu_{0,j} }{ \sqrt{ \sigma^2_{1,j}/n_1 + \sigma^2_{0,j}/n_0 } }$
		- A **two-sample t-statistic** ==determines if the average difference between two independent groups is statistically significant==. It is primarily used to test whether two population means are equal.
	- 여기서:
		- $\mu_{1,j}$: reward hacking group에서 feature $j$ 평균
		- $\mu_{0,j}$: normal group에서 feature $j$ 평균
		- $\sigma^2_{1,j}$, $\sigma^2_{0,j}$: 각 group의 variance
		- $n_1$, $n_0$: 각 group sample 수
6. $|t_j|$가 큰 feature를 reward hacking feature로 봄
	- `이 feature는 reward hacking step에서만 유독 강하게 켜진다`
	- `이 feature는 normal step과 reward hacking step을 잘 구분한다`
---
- 논문에서는 이런 feature를 causal graph에서 confounder \(Z\)처럼 봄.
- 직관적으로는:
	- `PRM이 진짜 reasoning quality를 보는 게 아니라, 어떤 그럴듯한 패턴 feature Z에 속아서 높은 reward를 준다.`
	- 그래서 그 feature $Z$의 영향을 backdoor adjustment로 줄이려는 거야.
- 정리하면: `“statistically analyzing which features discriminate...” = SAE feature들 중에서 reward hacking step과 normal step의 activation 분포가 크게 다른 feature를 t-statistic으로 골라낸다는 뜻`
### (2) Mitigation
- 분석은 internal에서 하고 mitigation은 "in External Reasoning"에서 하는 느낌

```text
문제 x
↓
policy model이 step 1 후보 여러 개 생성
↓
각 후보를 PRM에 넣음
↓
PRM 내부 activation을 SAE로 분석
↓
reward hacking feature를 이용해 PRM score 보정
↓
보정된 score로 top-k 선택
↓
각 후보에서 다시 step 2 후보 여러 개 생성
↓
같은 과정 반복
↓
최종 reasoning path 선택
```

$|t_j|$가 큰 feature는 hack/normal을 잘 구분하는 feature로 본다.
- 중요한 점:
	- `그 feature 값이 크면 무조건 나쁘다는 뜻은 아님. hack group과 normal group에서 분포 차이가 크다는 뜻.`
	- 그래서 feature를 그냥 낮추는 게 아니라, 그 feature 값을 여러 값으로 바꿔 PRM score를 다시 계산하고 평균낸다.
		- 높이기/낮추기, ...
-  $\hat R_{\mathrm{CRA}}(X) = \sum_z R(X, Z=z)P(Z=z)$
	- 이 보정된 score를 raw PRM score 대신 beam search에 사용한다.
	- `raw PRM score -> adjusted CRA score -> top-k 선택`
- 결과적으로, 특정 reward hacking feature 때문에만 점수가 높던 step은 선택될 확률이 줄어든다.

### (3) 비슷한 걸 이미지에서 해보려면?
|후보 reward task|추천도|이유|
|---|---|---|
|OCR / typography|높음|성공/실패가 비교적 명확함. reward hacking pattern도 만들기 쉬움|
|counting / object presence|중간-높음|라벨이 비교적 명확하지만 detector/evaluator가 필요|
|spatial relation|중간|left/right/on/under는 평가 가능하지만 모델/평가기가 자주 틀림|
|multi-concept / GenEval|중간|연구적으로 좋지만 failure type이 다양해서 초기 실험엔 복잡|
|aesthetic|낮음|“hacking인지 스타일인지” 구분이 너무 애매함|
|PickScore / human preference|낮음|사람이 좋아하는 이미지와 reward hacking 구분이 흐림|

| 정상 성공               | hacking/failure          |
| ------------------- | ------------------------ |
| 텍스트가 정확하고 자연스럽게 들어감 | 텍스트가 과도하게 큼              |
| 디자인/이미지와 잘 어울림      | 같은 글자가 반복됨               |
| 배경과 조화로움            | 글자만 맞고 이미지 품질이 망가짐       |
| 읽기 쉬움               | OCR은 맞지만 typography가 이상함 |

```txt
Prompts:
  300 OCR prompts
  300 counting prompts

Generate:
  prompt당 16~64 images

Target reward:
  OCR reward / object-count reward

Independent judge:
  GPT-4o or VLM judge
  + simple rule-based check if possible

Label:
  success = reward high + judge high
  failure = reward high + judge low
```

“OCR 올리면 aesthetic/quality가 내려간다”, “multi-reward는 balancing이 중요하다”, “seesaw effect/reward hacking이 생긴다”는 문제의식은 이미 연구들이 아래와 같이 말하고 있긴 함

|논문|관련 내용|
|---|---|
|Flow-OPD|제일 직접적. multi-task alignment에서 scalar reward sparsity + gradient interference 때문에 **seesaw effect**가 생긴다고 말함. 특히 OCR 같은 specific feature를 최적화하면 aesthetic이 degrade될 수 있다고 언급함.|
|Flow-GRPO|GenEval, OCR, PickScore, aesthetic, DeQA 등을 같이 평가함. single reward를 올렸을 때 다른 quality/preference metric이 어떻게 변하는지 보여줌.|
|Diffusion Blend|aesthetic quality, text-image consistency 같은 여러 objective가 충돌하므로 inference-time multi-preference alignment가 필요하다고 봄.|
|Rewards-in-Context|multi-objective alignment에서 preference가 heterogenous/conflicting하다는 문제를 다룸. T2I에서는 aesthetic + compressibility 예시.|
|Rewarded Soups|여러 proxy reward가 서로 conflicting할 수 있으니 reward별 모델을 만들고 weight interpolation으로 trade-off를 다룸.|
#### 그러나 연구를 그대로 옮기기에 T2I에서 문제가 되는 점

|LLM external reasoning|T2I OCR|
|---|---|
|reasoning step이 텍스트라 직접 평가 가능|\(x_t\)는 noisy latent라 OCR 평가 불가능|
|PRM이 step-level score를 줌|OCR reward는 final image에서만 의미 있음|
|PRM internal activation을 SAE로 분석|T2I는 generator activation을 분석해야 함|
|보정된 PRM score로 beam search 선택|일반 diffusion sampling은 beam search 구조가 아님|
# 결론
- 같은 방법을 t2i에 적용해보기에는 causal 논문 보다는 monitoring이 더 쉬움. / prm on t2i 보다는 sae feature이 쉬울테니...
- controlled OCR reward-failure benchmark + activation monitor는 똑같은 방법 적용하는 거 해 볼수는 있긴 함
	- “꼼수 행동을 할 때 activation 분포가 정상 행동과 달라질 것이다”는 가정은 같음