- Flow-OPD가 말했다: specialized teacher가 semantically wrong image를 만들면, 그 오류가 dense supervisory signal로 student에게 전파될 수 있다
	- semantic error
	- artifact
	- reward shortcut
	- reward hacking
	- teacher가 못하는 prompt
- how to audit and weight teacher signals before distillation, so that multi-teacher distillation becomes more robust!!!


```txt
1. reward-trained teacher 준비
   예: OCR reward로 학습한 teacher

2. teacher가 prompt별 image/trajectory 생성

3. teacher output을 두 기준으로 평가
   target reward: OCR score
   independent verifier: aesthetic / artifact / MLLM / human-like judge

4. teacher sample을 나눔
   reliable:
     OCR도 높고 verifier도 좋음

   unreliable:
     OCR은 높은데 artifact/verifier가 나쁨

5. student를 두 방식으로 distill
   S_all:
     모든 teacher signal을 사용

   S_filtered:
     unreliable teacher signal을 제거하거나 downweight

6. student 평가
   S_all이 teacher의 failure pattern을 더 많이 물려받는가?
   S_filtered가 target reward를 유지하면서 artifact/failure를 줄이는가?
```

## Experiments
### sanity check
```txt
OCR teacher
-> OCR-heavy prompts 100개
-> teacher outputs audit
-> reliable / unreliable split
-> small LoRA student distillation
-> S_all vs S_filtered 비교
```
### 확장
```txt
단일 teacher
-> 여러 reward teacher

single failure type
-> artifact / OCR / aesthetic / compositional failure

offline sample filtering
-> trajectory/velocity-level filtering

small LoRA student
-> full Flow-OPD-style student
```
### distillation에서 filtering/reweighting은 흔한 방법? - ㅇㅇ.

|분야|흔한 방식|
|---|---|
|self-training / pseudo-labeling|teacher가 확신 높은 sample만 student에게 학습|
|LLM distillation|teacher output 중 quality 좋은 것만 사용, bad response 제거|
|preference/RL data|reward 높은 sample만 쓰거나, low-quality sample downweight|
|diffusion distillation|generated sample 품질 필터링, aesthetic/NSFW/watermark filtering 등|
### 실험/관찰 순서
|단계|가능하면 좋은 결과|안 되면 의미|
|---|---|---|
|1. Offline filtering|S_filtered가 S_all보다 failure transfer를 줄임|이게 최소 메인 contribution|
|2. Trajectory analysis|reliable/unreliable이 특정 timestep부터 갈라짐|teacher failure가 denoising dynamics에 나타난다는 분석 contribution|
|3. Online detect/gating|final image 전에 위험 signal 감지 가능|대박 contribution. online teacher-signal gate 가능|
|4. 약한 관찰|mid/late step에서만 통계적 차이|full online은 아니지만 “late-stage audit/gating” 가능성 제시|
|5. detect가 약함|중간에서 잘 안 갈라짐|그래도 offline filtering baseline의 가치가 남음|
### 논문 구조
RQ1. Does unreliable teacher signal transfer to the student?
RQ2. Can offline filtering/reweighting reduce this transfer?
RQ3. Are unreliable teacher trajectories distinguishable before final generation?
RQ4. If yes, can timestep-level gating further improve distillation?

### 결과에 따른 결론
```txt
online detection 성공:
  강한 method 논문

online detection 약함:
  teacher failure transfer + offline filtering 논문

offline filtering도 별로:
  이 주제는 폐기
```

### 관련 논문 (anchor)
| Anchor                                     | 왜 필요한가                                                        | RQ                                                                                                                                                                                      |                                                                                                                         |
| ------------------------------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Flow-OPD**                               | reward-specific teacher를 만들고 student에 distill하는 구조의 직접 기반     | reward-specific teachers can be used to combine multiple reward skills.                                                                                                                 | teacher distillation is useful                                                                                          |
| **Understanding Reward Hacking in T2I RL** | reward-trained model이 artifact/shortcut을 배울 수 있다는 문제의식의 직접 기반 | reward optimization can produce high-reward but low-quality / artifact-heavy generations.                                                                                               | reward-trained models can be unreliable                                                                                 |
| Ours                                       |                                                               | 1. If reward-trained teachers can contain such failures, do these failures transfer to students during distillation?<br>2. Can we audit/filter teacher signals to reduce this transfer? | teacher distillation assumes teacher signals are useful, but reward-trained teachers may contain reward-hacking signals |
