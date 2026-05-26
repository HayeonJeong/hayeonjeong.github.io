---

layout: page
title: "Phonetic-Aware Encoder Tuning"
description: "Progress update: from phonetic CTC supervision to articulatory-feature heads for L2 Korean ASR"
img: assets/img/projects/1_phonetic/phonetic_thumnail.png
importance: 1
category: 2025
related_publications: false
---

## Project Snapshot

This project studies why a strong pre-trained ASR model can still fail on **non-native Korean speech**. The motivating issue is not only data scarcity. L2 speakers often realize Korean sounds through the phonetic categories of their first language, so the acoustic signal may no longer align cleanly with the standard Korean phoneme that the recognizer expects {% cite flege1995second best1995direct --file phonetic %}.

My first approach was **Phonetic-Aware Encoder Tuning**: keep the Whisper-style decoder stable, adapt the encoder with LoRA, and add a temporary phonetic CTC objective so that the encoder learns a sharper acoustic-to-phoneme alignment {% cite radford2023robust hu2022lora graves2006ctc --file phonetic %}. That worked as an initial direction. The current update is about the next step: instead of supervising the model with a single phoneme symbol, I am trying to supervise it with more explicit **articulatory features (조음 특성)** such as place, manner, laryngeal contrast, vowel height, backness, and rounding.

---

## What I Have Done So Far

The project started from a practical ASR failure mode: when Korean learner speech contains hesitation, unclear codas, weakened fricatives, or vowel substitutions, a general ASR model can produce a fluent-looking transcript that is phonetically wrong. For this reason, I focused on the encoder, where acoustic evidence is first converted into hidden representations.

The current pipeline has gone through four stages:

* **Data cleaning and target construction**: I cleaned Korean learner-speech transcripts by removing hesitation markers, noise/unintelligible markers, repetition symbols, and stray encoding artifacts. The data source is aligned with AI Hub's foreign-speaker Korean speech setting, which targets Korean ASR robustness for speakers whose native language is not Korean {% cite aihub2021foreign_korean_speech --file phonetic %}.
* **Phonetic target generation**: I generated sub-syllabic targets such as Jamo/G2P-style sequences and IPA-inspired targets, since Korean errors are often clearer below the syllable level. IPA and feature-based representations are useful because they expose articulatory contrasts that orthography hides {% cite international1999handbook mortensen2016panphon --file phonetic %}.
* **Encoder-only adaptation**: I applied LoRA to the encoder while keeping the decoder frozen, so that the model changes its acoustic interpretation without relearning the language model.
* **Auxiliary phonetic supervision**: I attached a CTC head during training and removed it at inference time, keeping the inference cost unchanged.

The original setup compared a Whisper baseline against encoder-only LoRA with phonetic CTC supervision.

| Model              |  JP (CER↓) |  CN (CER↓) |  VN (CER↓) |
| ------------------ | ---------: | ---------: | ---------: |
| Whisper (Base)     |     0.0822 |     0.1171 |     0.1247 |
| **LoRA + IPA CTC** | **0.0768** | **0.1044** | **0.1149** |

The first result was encouraging: IPA-based CTC supervision improved CER across Japanese, Chinese, and Vietnamese L1 groups. Error-change analysis also suggested that the method reduced language-specific phonetic confusions, such as Japanese nasal-coda errors, Chinese vowel rounding/backness errors, and Vietnamese vowel-height or fricative-weakening errors.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/1_phonetic/model_architecture.png" title="Proposed Framework" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Figure 1: Initial phonetic-aware encoder tuning framework with LoRA and auxiliary CTC supervision.
</div>

---

## What The First Results Revealed

The first experiments did more than reduce CER. They also suggested that the encoder was changing *where* it placed confidence over time. In forced-alignment style analysis, the baseline produced diffused phonetic probabilities, while the phonetic-aware model produced sharper peaks near the expected phoneme locations {% cite mcauliffe2017montreal --file phonetic %}.

That was a useful sign, but it also raised a new question: if the CTC target is too symbol-like, does the model become overly eager to force uncertain learner speech into a single clean phoneme category?

This is where the project shifted from “phoneme-level supervision improves ASR” to a more careful question:

> Can the encoder use the remaining valid acoustic cues in learner speech, instead of simply snapping every ambiguous sound to one phoneme label?

Layer-wise experiments also pushed me in this direction. Late-layer CTC was not always stable, and middle layers sometimes looked more useful for Jamo-CER than final layers. This matches a broader pattern in speech representation analysis: acoustic, phonetic, and word-level information are not uniformly distributed across layers {% cite pasad2023comparative martin2023probing --file phonetic %}.

---

## Current Direction: Articulatory Features

The new hypothesis is that a phoneme token is too coarse for the errors I care about. For example:

* `ㄱ -> ㄲ` is mainly a **laryngeal** error: lenis vs. tense.
* `ㅍ -> ㅃ` preserves place and manner, but changes **laryngeal** type: aspirated vs. tense.
* `ㅓ -> ㅗ` can be read as a **backness/rounding** error more than a generic vowel substitution.
* `ㄴ -> ㅇ` is a **place** error in the coda nasal region.

So instead of asking the encoder to predict only a phoneme sequence, I am now trying **multiple CTC heads**, each responsible for one articulatory dimension.

| Segment type | Feature head | What it asks the encoder to learn |
| ------------ | ------------ | --------------------------------- |
| Consonants | place | bilabial, alveolar, alveolo-palatal, velar, glottal |
| Consonants | manner | stop, fricative, affricate, nasal, liquid |
| Consonants | laryngeal | lenis, tense, aspirated, none |
| Vowels | height | high, mid, low-mid, low |
| Vowels | backness | front, central, back-central, back |
| Vowels | rounding | rounded, unrounded |

The training objective becomes a multi-task version of the original setup:

$$
\mathcal{L}_{total}
= \mathcal{L}_{ASR}
+ \sum_h \lambda_h \mathcal{L}_{CTC}^{(h)}
$$

where each head \(h\) corresponds to a different articulatory feature. As before, these heads are used only for training and removed at inference.

This reframes the analysis too. A normal confusion matrix says “the model confused `ㄱ` with `ㅋ`.” The articulatory view says “place and manner were preserved, but laryngeal contrast failed.” That is a much better diagnostic for L2 Korean speech.

---

## Layer Hypothesis

The current experiment also assigns feature heads to different encoder layers instead of always using the final layer. The intuition is that different phonetic cues live at different temporal and representational scales.

* **Laryngeal contrast** is placed earlier. Korean lenis/tense/aspirated distinctions rely on short-time cues such as VOT, aspiration, onset F0, and voice quality {% cite cho2002acoustic kang2022phonology martin2023probing --file phonetic %}.
* **Place of articulation** is placed in early-to-middle layers. Place cues often depend on burst/release noise, spectral shape, and formant transitions into the following vowel {% cite chen2000place cho2002acoustic --file phonetic %}.
* **Manner** is placed around the middle layers, because stops, fricatives, affricates, nasals, and liquids require slightly longer evidence about how the sound unfolds.
* **Vowel height, backness, and rounding** are placed in middle-to-late layers, because vowel identity depends on a more stable voiced region and is affected by surrounding consonants.

This is still a working hypothesis, not a finished claim. The latest experiments compare layer choices such as:

* laryngeal at layers 2, 3, or 4
* place at layers 4, 5, or 6
* manner at layers 5, 6, or 7
* height/backness/rounding around layers 7-11

---

## New Evaluation

The evaluation is also shifting from only CER/Jamo-CER to feature-level recovery. Two metrics I am using now are:

$$
CCR_h(a \rightarrow b)
=
\frac{count(gt=a, pred=b)}{count(gt=a)}
$$

**Contrast Confusion Rate (CCR)** measures how often a specific feature value \(a\) is confused as \(b\) under head \(h\). For example, it can track `lenis -> aspirated` for laryngeal contrast or `alveolar -> alveolo-palatal` for place.

$$
\Delta CCR_h(a \rightarrow b)
=
CCR_{system}(a \rightarrow b)
-
CCR_{baseline}(a \rightarrow b)
$$

Negative \(\Delta CCR\) means the confusion decreased. I also track **recovery** and **regression**: did the new system fix samples the baseline missed, or did it damage samples the baseline already handled?

This matters because the goal is not just to lower aggregate CER. The goal is to understand which articulatory contrasts the model has actually learned to use.

---

## Early Observations

The articulatory-feature direction is still being tested, but the intermediate results are promising enough to continue:

* Japanese L1 experiments show a clearer reduction in **laryngeal** confusion, especially around lenis/tense/aspirated contrasts.
* Vietnamese L1 experiments show partial improvement in **place** confusion under split-layer settings.
* Vietnamese L1 also shows relatively consistent improvement in **backness** confusion.
* Not all feature heads appear equally useful at every layer, so a brute-force grid search is not enough. The analysis needs to explain *why* a feature improves or regresses.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/1_phonetic/error_matrix.png" title="Error Change Matrix" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Figure 2: Earlier phonetic error-change analysis. The current work extends this symbol-level view into feature-level contrast analysis.
</div>

---

## Next Steps

The main next step is to make the articulatory-feature story more rigorous. I am currently focusing on:

* ablations for “all six heads” vs. removing one head at a time
* layer-placement comparisons for each feature type
* feature-level confusion analysis by L1 group
* recovery/regression analysis for hard samples
* comparing internal representations against cleaner ground-truth speech to see whether the encoder moves toward the intended acoustic-phonetic target

The project has therefore moved from a single claim - “phonetic CTC helps L2 Korean ASR” - to a more specific research question: **which articulatory cues are recoverable from learner speech, where do they appear inside the encoder, and how can supervision encourage the model to use those cues without over-normalizing the speaker?**

---

## References

<div class="publications">
{% bibliography --file phonetic --cited_in_order %}
</div>
