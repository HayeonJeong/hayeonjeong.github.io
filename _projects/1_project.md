---

layout: page
title: "Phonetic-Aware Encoder Tuning"
description: "Boosting ASR Robustness against L2 Pronunciation Variations via CTC Supervision"
img: assets/img/projects/1_phonetic/phonetic_thumnail.png
importance: 1
category: 2025
related_publications: true
---

## Overview

Automatic Speech Recognition (ASR) systems based on large-scale pre-trained models have reached near-human performance for **native speech**. However, their robustness drops significantly when faced with **non-native (L2) speech**. This gap is not merely a data sparsity issue—it originates from systematic **phonetic mismatches** between how L2 speakers articulate sounds and how ASR models internally represent them.

This project introduces **Phonetic-Aware Encoder Tuning**, a framework designed to explicitly correct such mismatches by improving the *acoustic–phonetic resolution* of the encoder in Whisper-style ASR models. Instead of modifying the entire model, we focus on **encoder-only adaptation via LoRA**, guided by **auxiliary phonetic CTC supervision**.

---

## Motivation: Why L2 Speech Breaks Modern ASR

L2 speakers often map unfamiliar phonemes in the target language to the closest sounds available in their native inventory. This phenomenon—commonly referred to as **L1 interference**—produces systematic pronunciation patterns that deviate from native norms.

For Korean L2 speech, these patterns vary strongly by native language:

* **Japanese (JP)** speakers struggle with Korean nasal codas and stop–fricative contrasts due to open-syllable constraints in Japanese.
* **Chinese (CN)** speakers frequently confuse vowel height (e.g., /u/ vs. /o/) and labial codas.
* **Vietnamese (VN)** speakers tend to weaken fricatives and misalign vowel height due to a dense vowel inventory.

Crucially, these errors emerge **before linguistic decoding**, at the level of acoustic representation and alignment. This suggests that improving the **encoder**, rather than the decoder, is the most principled intervention point.

---

## Key Idea: Phonetic-Aware Encoder Tuning

Our central hypothesis is simple:

> *If the encoder is explicitly guided to align distorted L2 acoustics with correct phonetic categories, the downstream ASR output will improve—even without modifying the decoder.*

To operationalize this idea, we propose a multi-task framework with three core components:

---

## Methodology

### 1. Encoder-Only LoRA Adaptation

Rather than fine-tuning the full Whisper model, we apply **Low-Rank Adaptation (LoRA)** *exclusively* to the **encoder blocks**, while keeping the decoder frozen.

This design choice has two advantages:

* It preserves the strong linguistic priors already learned by the decoder, avoiding catastrophic forgetting.
* It allows efficient adaptation to L2-specific acoustic distortions with minimal additional parameters.

The encoder thus learns to reshape its acoustic feature space without altering high-level language modeling behavior.

---

### 2. Auxiliary Phonetic CTC Supervision

To provide direct acoustic–phonetic guidance, we attach a lightweight **CTC head** to the final encoder layer during training.

The model is trained with a multi-task objective:

$$
\mathcal{L}*{total} = \mathcal{L}*{ASR} + \lambda , \mathcal{L}_{CTC}
$$

* **ASR loss**: standard Whisper decoder loss
* **CTC loss**: frame-level phonetic alignment

CTC is particularly well-suited for L2 speech because it:

* Enforces monotonic alignment without requiring exact frame labels
* Handles irregular phoneme durations common in non-native speech

Importantly, the CTC head is **removed at inference time**, so there is **no additional runtime cost**.

---

### 3. Choosing the Right Phonetic Target

We investigate three alternative phonetic representations as CTC targets:

* **Phonetic Hangul (g2pk2)**
  Reflects surface pronunciation after Korean phonological rules; effective for sub-syllabic structure errors.

* **Yale Romanization**
  A morpho-phonemic representation commonly used in linguistics; provides a structured but abstract phoneme mapping.

* **International Phonetic Alphabet (IPA)**
  A fine-grained articulatory representation capturing aspiration, tenseness, and vowel height.

Among these, **IPA provides the most explicit phonetic supervision**, making it particularly effective for resolving subtle L2 pronunciation errors.

---

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/1_phonetic/model_architecture.png" title="Proposed Framework" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Figure 1: Phonetic-Aware Encoder Tuning with LoRA and auxiliary CTC supervision.
</div>

---

## Experimental Setup

* **Dataset**: NIKL Korean Learner Corpus (JP, CN, VN speakers)
* **Audio Generation**: XTTS-based L2 speech synthesis from non-standard transcriptions
* **Ground Truth Normalization**: Context-aware correction using LLM-based reconstruction
* **Base Model**: Whisper-small
* **Training**: Encoder-only LoRA + multi-task CTC learning
* **Metrics**: Character Error Rate (CER) and Jamo-level CER

Jamo-CER is especially important for Korean, as it captures partial phonetic mismatches that are invisible at the syllable level.

---

## Results

### Quantitative Performance

Across all L1 groups, **IPA-based CTC supervision consistently achieves the strongest improvements**, particularly for Chinese and Vietnamese speakers.

| Model              |  JP (CER↓) |  CN (CER↓) |  VN (CER↓) |
| ------------------ | ---------: | ---------: | ---------: |
| Whisper (Base)     |     0.0822 |     0.1171 |     0.1247 |
| **LoRA + IPA CTC** | **0.0768** | **0.1044** | **0.1149** |

---

### Phonetic Error Reduction

Error-change analysis reveals that the proposed method directly corrects **L1-specific phonetic failure modes**:

* **Japanese**: Improved distinction of nasal codas (/ŋ/, /n/, /m/) and stop–fricative boundaries
* **Chinese**: Robust recovery of vowel height contrasts (/u/ vs. /o/) and labial codas
* **Vietnamese**: Sharper alignment of vowel height and fricative articulation

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/1_phonetic/error_matrix.png" title="Error Change Matrix" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Figure 2: Top phonetic error reductions after phonetic-aware encoder tuning.
</div>

---

## What Actually Changed?

Forced-alignment analysis shows a qualitative shift in encoder behavior:

* **Baseline Whisper** produces diffused, unstable phonetic probabilities
* **Phonetic-aware tuning** yields sharp, well-aligned peaks at correct phoneme timestamps

This confirms that the improvement is not merely linguistic post-processing, but a **fundamental correction of acoustic interpretation**.

---

## Conclusion

This work demonstrates that **explicit phonetic supervision at the encoder level** is a powerful and efficient strategy for improving ASR robustness to L2 speech.

By combining:

* Encoder-only LoRA adaptation
* Auxiliary CTC-based phonetic alignment
* Carefully chosen phonetic targets (IPA)

we achieve consistent gains without increasing inference cost or compromising native-speech performance.

---

## Future Directions

* Expanding to additional L1 groups and real (non-synthetic) L2 audio
* Exploring noise-robust and articulatory-feature-based phonetic targets
* Applying phonetic-aware tuning to multilingual and streaming ASR models
