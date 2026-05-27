# 0. Anchor Papers
- **Flow-GRPO: Training Flow Matching Models via Online RL**
	- applies online RL/GRPO to flow matching models by converting deterministic flow sampling into a stochastic policy.
- **DiffusionNFT: Online Diffusion Reinforcement with Forward Process**
	- avoids reverse-process policy gradient and instead injects reward signals into the forward noising / flow-matching objective.
- **Flow-OPD: On-Policy Distillation for Flow Matching Models**
	- handles multi-reward conflict by training reward-specific teachers and distilling their dense velocity signals into one student model.

Main question:
> How can text-to-image diffusion/flow models be post-trained with rewards without causing reward hacking, reward conflict, or severe sampling inefficiency?

# 1. Base Model: Diffusion to Flow Matching
The base model is the text-to-image generator before reward post-training.  
The main flow is:
$$
\text{DDPM}
\rightarrow
\text{Score SDE / DDIM}
\rightarrow
\text{Flow Matching}
\rightarrow
\text{Rectified Flow / SD3-style flow model}
$$

## 1.1 DDPM: learn to denoise
DDPM defines a fixed forward noising process:
$$
q(x_t \mid x_0)
=
\mathcal N
\left(
x_t;
\sqrt{\bar\alpha_t}x_0,
(1-\bar\alpha_t)I
\right)
$$
Equivalently:
$$
x_t
=
\sqrt{\bar\alpha_t}x_0
+
\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I)
$$
The model learns to predict the noise:
$$
\epsilon_\theta(x_t,t,c)\approx \epsilon
$$
## 1.2 Score / reverse process
The denoising direction can also be written as a score:
$$
s_\theta(x_t,t,c)
\approx
\nabla_{x_t}\log p_t(x_t\mid c)
$$
For Gaussian noising, noise prediction and score prediction are connected:
$$
s_\theta(x_t,t,c)
\approx
-\frac{\epsilon_\theta(x_t,t,c)}
{\sqrt{1-\bar\alpha_t}}
$$
This leads to reverse SDE / probability flow ODE views of diffusion sampling.

## 1.3 DDIM / ODE-like sampling
DDIM shows that sampling can be made deterministic by removing fresh random noise at each step.

Predicted clean image:
$$
\hat x_0
=
\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t,c)}
{\sqrt{\bar\alpha_t}}
$$
Deterministic sampling roughly follows:
$$
x_t \rightarrow x_{t-\Delta t}
$$
without adding new random noise.  
This is the bridge from stochastic diffusion sampling to ODE-like generation.

## 1.4 Flow Matching: learn velocity directly
Flow matching directly learns a velocity field:
$$
\frac{dx_t}{dt}=v_\theta(x_t,t,c)
$$
Instead of predicting noise/score and deriving the reverse process, the model learns how samples should move along a probability path.

General path:
$$
x_t=\alpha_t x_0+\sigma_t\epsilon
$$
Target velocity:
$$
v
=
\frac{dx_t}{dt}
=
\dot\alpha_t x_0+\dot\sigma_t\epsilon
$$
## 1.5 Rectified Flow
Rectified flow chooses the simplest straight path:
$$
x_t=(1-t)x_0+t\epsilon
$$
So the target velocity becomes:
$$
v=\frac{dx_t}{dt}=\epsilon-x_0
$$
During generation, the model starts from noise and integrates backward:
$$
x_1 \rightarrow x_0
$$
Euler step:
$$
x_{t-\Delta t}
\approx
x_t-\Delta t\,v_\theta(x_t,t,c)
$$
# 2. Image Reward Models

## 2.1. Taxonomy of recent papers

| Level | Category                                   | Core question                                                                               | Examples                                                      |
| ----- | ------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1     | **Reward model learning**                  | How do we build a good image/prompt scorer?                                                 | ImageReward, PickScore, HPSv2, DeQA, UnifiedReward            |
| 2     | **Single-reward post-training**            | How do we improve a generator using one reward?                                             | ReFL, DDPO, DPOK, Flow-GRPO, DiffusionNFT                     |
| 3     | **Preference-pair direct alignment**       | Can we align the generator directly from preference pairs without an explicit reward model? | Diffusion-DPO, DSPO, Dense Reward View                        |
| 4     | **Multi-reward optimization**              | How do we optimize several reward axes without conflict?                                    | Parrot, Rewards-in-Context, MixGRPO, GDPO, Flow-Multi         |
| 5     | **Expert merging / blending**              | Can we train reward-specific experts and combine them later?                                | Rewarded Soups, Diffusion Soup, Bone Soups, Diffusion Blend   |
| 6     | **Reward-trained teacher distillation**    | Can we transfer reward-specialized teacher abilities into one student?                      | Flow-OPD                                                      |
| 7     | **Reward hacking / evaluation robustness** | How do we detect and reduce reward exploitation?                                            | reward hacking analyses, MLLM judges, task-specific verifiers |
### 2.1.1 How the Anchor Papers Use Image Rewards

**Flow-GRPO** directly uses a final image reward to update the flow model with online RL.  
For each prompt, it samples multiple images, scores them with \(R(x_0,c)\), and converts the scores into group-relative advantages:
$$
\hat A_i=
\frac{R(x_0^{(i)},c)-\mathrm{mean}_j R(x_0^{(j)},c)}
{\mathrm{std}_j R(x_0^{(j)},c)}
$$
---
**DiffusionNFT** also starts from a final image reward, but avoids reverse-trajectory policy gradient.  
It converts the raw reward into a soft optimality weight ($r(x_0,c)\in[0,1]$), then uses it inside a forward noising / flow-matching loss:
$$
L=
\mathbb E
\left[
r\|v_\theta^+-v\|^2
+
(1-r)\|v_\theta^--v\|^2
\right]
$$
---
**Flow-OPD** addresses multi-reward conflict by not mixing scalar rewards directly.  
It first trains reward-specific teachers, then distills their dense velocity signals into one student:

$$
v_{\mathrm{target}}(x_t,t,c)=v_{\phi_k}(x_t,t,c)
$$
---
So the three papers form a progression:
$$
\text{final scalar reward}
\rightarrow
\text{reward-weighted forward loss}
\rightarrow
\text{reward-trained teacher distillation}
$$
## 2.2 Taxonomy of Multi-Reward Models


| Category                                                       | Intuition                                                                      | What is mixed?                                            | Final form                    | Examples                                                                                                |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| A. Multi-reward optimization during training                   | Mix rewards during training, but try to find a balanced trade-off.             | reward scalars, advantages, reward-conditioned data       | one trained model             | Parrot (2024.01), Rewards-in-Context (2024.02), MixGRPO (2025.07), GDPO (2026.01), Flow-Multi (2026.02) |
| B. Train reward-specific experts, then merge weights           | Fine-tune separate models for each reward, then merge them in parameter space. | model weights                                             | one merged model              | Rewarded Soups (2023.06), Diffusion Soup (2024.06), Bone Soups (2025.02)                                |
| C. Train reward-specific experts, then blend at inference time | Keep experts separate and combine their generation processes during inference. | denoising process, drift, LoRA/expert choice              | inference-time ensemble/blend | Diffusion Blend (2025.05)                                                                               |
| D. Train reward-specific experts, then distill                 | Transfer the capabilities of multiple teachers into one student model.         | teacher predictions, teacher velocity, trajectory signals | one distilled student         | Flow-OPD (2026.05)                                                                                      |
| E. Make the reward model multi-dimensional                     | Instead of one total score, output separate scores for different criteria.     | reward dimensions                                         | multi-dimensional scorer      | VisionReward (2026.03), multi-head/rubric-style reward                                                  |
| F. Convert sparse reward into dense reward/supervision         | Replace final scalar reward with timestep- or trajectory-level signals.        | dense reward, timestep signals, velocity supervision      | denser training signal        | Dense Reward View (2024.02), **Flow-OPD (2026.05)**                                                     |

## 2.3. Limitation Image Reward Models

| Limitation                                | Meaning                                                                                                                                                                                                                                                                           |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Limited human preference data             | ImageReward, PickScore, and HPSv2 are trained with pairwise preferences, but they depend on the number of annotators, prompt distribution, and image generation model distribution. Generalization can become unstable for out-of-distribution prompts or images from new models. |
| Too much compressed into one scalar score | A single score compresses many different criteria, such as aesthetics, prompt following, text rendering, object counting, and safety. This loses information.                                                                                                                     |
| Limited reward capability                 | CLIP-based rewards can be weak at fine-grained counting, spatial relations, and OCR. VQA/MLLM judges are stronger but slower and more expensive. OCR rewards can check text, but they do not evaluate overall aesthetics.                                                         |
| The model exploits reward loopholes       | The generator may shift toward colors, compositions, textures, or exaggerated sharpness that the reward prefers, or it may produce images that only satisfy the benchmark.                                                                                                        |
| Rewards can conflict with each other      | Improving OCR reward can reduce aesthetics, while improving aesthetic reward can hurt counting or spatial relations. This creates a seesaw effect.                                                                                                                                |
## 2.3. Recent Research Trend
- 하나의 reward model을 절대 기준으로 믿지 않는다.
- 여러 reward/verifier를 같이 본다.
- Scalar reward 대신 dense supervision을 찾는다.
- Reward-trained teacher를 distill한다.
- MLLM judge나 task-specific verifier를 보조로 쓴다.
- Reward hacking을 따로 분석한다.
---
- Do not treat a single reward model as an absolute standard.
- Use multiple rewards/verifiers together.
- Look for dense supervision instead of relying only on scalar rewards.
- Distill reward-trained teachers.
- Use MLLM judges or task-specific verifiers as auxiliary evaluators.
- Analyze reward hacking separately.
## 2.4. 저비용 Multi 연구
| 저비용 방향 | 논문 나온 정도 | 관련 논문/분야 | 아직 남은 gap |
|---|---|---|---|
| 기존 checkpoint/sample 분석 | 많음 | GenEval, T2I-CompBench, HRS-Bench처럼 여러 T2I 모델 output을 benchmark로 분석 | reward conflict 자체를 중심으로 분석한 건 아직 적음 |
| reward correlation 분석 | 일부 있음 | HPSv2/PickScore류는 human preference와 metric correlation을 봄. Reward hacking 논문은 여러 reward가 어떻게 hacking을 유발하는지 분석 | OCR, GenEval, PickScore, aesthetic 간 conflict matrix를 체계적으로 보는 건 여전히 gap |
| prompt-level taxonomy | 많음 | GenEval, T2I-CompBench, HRS-Bench | reward conflict 기준의 prompt taxonomy는 덜 정리됨 |
| small LoRA fine-tuning | 있음 | reward fine-tuning, DPOK/ReFL/DRaFT류에서 efficient tuning 흐름 존재 | LoRA로 reward별 conflict만 싸게 분석하는 건 좋은 저비용 gap |
| reward hacking detector | 있음, 최근 뜸 | Understanding Reward Hacking in T2I RL은 artifact reward model을 제안 | internal feature/SAE 기반 detector는 아직 더 열려 있음 |
| teacher 없이 self-OPD | 인접 연구는 있음 | Dense Reward View, self-play fine-tuning, OPD/GKD, Flow-OPD | flow velocity용 self-OPD는 꽤 open |
| MLLM judge subset 사용 | 있음 | VIEScore, T2I-CompBench, MLLM-as-judge 평가 흐름 | 비싼 MLLM judge를 일부 sample에만 써서 reward conflict를 보정하는 건 실용적 gap |