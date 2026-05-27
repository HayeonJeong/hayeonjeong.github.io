# 논문 1) Bone Soups
Bone Soups는 **“reward별 single expert를 그냥 섞으면 최적 trade-off가 안 나온다”**는 문제에서 출발한 논문이야.

핵심은 이거야.

```text
기존 Rewarded Soup:
  R1만 잘하는 모델
  R2만 잘하는 모델
  R3만 잘하는 모델
  -> user preference 비율대로 weight merge

Bone Soup:
  처음부터 여러 reward를 섞은 backbone reward를 만든다
  -> 그 reward로 backbone model들을 학습한다
  -> 나중에 user preference에 맞게 backbone들을 merge한다
```

예를 들어 reward가 3개면 Bone Soup은 이런 식의 backbone reward를 만든다.

$$
h_1 = \beta R_1 + \frac{1-\beta}{2}R_2 + \frac{1-\beta}{2}R_3
$$

$$
h_2 = \frac{1-\beta}{2}R_1 + \beta R_2 + \frac{1-\beta}{2}R_3
$$

$$
h_3 = \frac{1-\beta}{2}R_1 + \frac{1-\beta}{2}R_2 + \beta R_3
$$

즉 `h1`은 `R1` 중심이지만 다른 reward도 조금 섞고, `h2`는 `R2` 중심, `h3`는 `R3` 중심이야.

그래서 Bone Soup의 “bone”은:

```text
single-objective expert가 아니라
multi-objective trade-off를 고려해서 만든 backbone model
```

에 가까움.

왜 이렇게 하냐면, reward들이 충돌할 때는

```text
R1-only expert와 R2-only expert를 0.5:0.5로 평균
```

한다고 해서 진짜 `0.5 R1 + 0.5 R2`에 좋은 모델이 되는 게 아니기 때문이야. Weight space에서 직선으로 섞는 경로가 Pareto-optimal path와 다를 수 있음.

논문 실험은 T2I가 아니라 **language model generation** 쪽이야. Long-form QA, Helpful Assistant, Reddit Summary에서 factuality/relevance/completeness, helpful/harmless/humor, faithful/preference 같은 reward trade-off를 봄. 모델은 T5-large, LLaMA-2-7B를 사용함.

네 주제와의 관계는 이거야.

| 관점 | 의미 |
|---|---|
| 이미 한 것 | single-reward expert soup은 한계가 있고, multi-objective backbone을 만든 뒤 merge하는 게 낫다는 주장 |
| 네 아이디어에 주는 신호 | 단순 LoRA soup만 하면 선행연구 반복이 될 가능성이 큼 |
| 남는 gap | T2I reward-specific LoRA에서 prompt-conditioned merge나 cheap mergeability score를 다룬 건 아님 |
| 주의점 | Bone Soup 자체는 저비용 방법은 아님. backbone들을 RL로 학습해야 해서 비용이 큼 |

그래서 `Bone Soups = 저비용 LoRA soup baseline`으로 보면 안 되고, 더 정확히는:

```text
단순 single-objective expert merge는 약하다.
merge할 expert 자체를 multi-objective basis로 잘 설계해야 한다.
```

라는 논문이야.

Sources: [arXiv:2502.10762](https://arxiv.org/abs/2502.10762), [ACL Anthology PDF](https://aclanthology.org/2025.acl-long.1322.pdf)