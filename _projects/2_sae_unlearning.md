---

layout: page
title: "Knows But Can't Draw"
description: "Modality-selective concept suppression in unified multimodal models with sparse autoencoders"
img: assets/img/projects/2_sae_unlearning/sae_unlearning_workflow.png
importance: 2
category: 2026
related_publications: false
---

## Overview

This is an ongoing project on **modality-selective concept suppression** in unified multimodal models. The goal is not to erase a concept from the whole model. Instead, I am exploring whether a model can still understand and explain a concept in text, while suppressing only the unsafe or undesirable visual rendering pathway.

The working phrase is:

> *The model knows the concept, but cannot draw it.*

This setting is motivated by cases where textual understanding may be useful or necessary, but image generation should be restricted: public or private faces, copyrighted characters, artist-specific styles, private documents, or other sensitive visual concepts.

---

## Current Direction

I am using **BAGEL-7B-MoT** as the main toy model because it is an open unified multimodal model that supports both visual understanding and image generation within a shared architecture. The current intervention point is the **image-token / VAE-side hidden state**, rather than the text-only stream.

The main idea is to train a **TopK Sparse Autoencoder (SAE)** on cached hidden states from image-generation prompts, then use the learned SAE features as interpretable handles for concept-level intervention.

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/2_sae_unlearning/sae_unlearning_workflow.png" title="SAE-based workflow" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Figure 1: Current toy pipeline for SAE-based modality-selective concept suppression.
</div>

---

## Toy Experimental Setup

The current toy setup focuses on object-level concept suppression before moving to harder cases such as identities or styles.

* **Base model**: BAGEL-7B-MoT, a unified multimodal model with 7B active parameters and about 14B total parameters.
* **Prompt source**: UnlearnCanvas-style object prompts.
* **Prompt scale**: 20 object classes, 80 anchor prompts per class, and 51 style postfixes plus one base prompt, giving roughly 83K prompt variants.
* **Hook point**: late image-token hidden states, currently around layer 24.
* **SAE**: TopK SAE with sparse latent activations, trained first in an unsupervised way on cached hidden states.
* **Initial target examples**: object concepts such as horses or dogs, with matched prompts for nearby retain concepts.

For feature selection, I construct matched prompt sets:

* **Target**: "A horse in a forest."
* **Background**: "An empty forest."
* **Retain**: "A cow in a forest."
* **Retain**: "A deer in a forest."

For each SAE feature, I compare its activation on target, background, and retain prompts. Features that activate strongly for the target concept but weakly for the background and nearby retain concepts become candidate suppression features.

---

## Planned Intervention

The current intervention is deliberately lightweight. Instead of fine-tuning the full model, I store a small **concept feature bank**:

* selected SAE feature indices
* activation threshold
* suppression strength
* layer and generation-step range
* retain/background statistics used for calibration

At inference time, selected SAE features are suppressed only when their activation exceeds the threshold during image generation. This makes the current version closer to **activation-level blocking** than complete weight-level unlearning, which is an important limitation I want to keep explicit.

I am also considering a small supervised shaping step after the unsupervised SAE stage. The idea is to encourage cleaner concept-to-feature binding and route consistency across text-to-image and image-conditioned generation, but this is still an ablation candidate rather than a finalized method.

---

## Expected Observations

Since this project is still in progress, I do not claim final quantitative results yet. The expected signals from the toy experiments are:

* target concepts should appear less often in generated images after suppression;
* nearby retain concepts, backgrounds, and styles should remain mostly intact;
* text-side understanding prompts should remain stable;
* useful SAE features should localize more to the target visual concept than to background or style artifacts;
* failure cases will likely come from features that entangle the object with the scene, pose, or style.

The first practical milestone is modest: if suppressing a small feature bundle makes a target object visibly weaker while preserving nearby concepts, that would suggest the approach is worth scaling to more concepts and stronger evaluation.

---

## Planned Evaluation

The evaluation plan is still being assembled, but the intended checks are:

* **target suppression**: whether a target object/style detector fires less often after intervention;
* **retain preservation**: whether related concepts such as cow/deer remain generatable;
* **image quality**: whether suppression avoids broad visual degradation;
* **text preservation**: whether the model can still answer ordinary text questions about the suppressed concept;
* **feature diagnostics**: activation distributions, image-token/text-token ratios, and heatmaps for selected SAE features.

This is meant to separate "the model cannot render the concept" from "the model no longer understands the concept."

---

## References

* [BAGEL: Emerging Properties in Unified Multimodal Pretraining](https://arxiv.org/abs/2505.14683), Deng et al., 2025.
* [BAGEL-7B-MoT model card](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT), ByteDance-Seed.
* [UnlearnCanvas: Stylized Image Dataset for Enhanced Machine Unlearning Evaluation in Diffusion Models](https://arxiv.org/abs/2402.11846), Zhang et al., 2024.
* [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600), Cunningham et al., 2023.
* [Scaling and Evaluating Sparse Autoencoders](https://arxiv.org/abs/2406.04093), Gao et al., 2024.
* [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345), Gandikota et al., 2023.
* [SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders](https://arxiv.org/abs/2501.18052), Cywinski and Deja, 2025.
* [Sparse Autoencoder as a Zero-Shot Classifier for Concept Erasing in Text-to-Image Diffusion Models](https://arxiv.org/abs/2503.09446), Tian et al., 2025.
* [SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](https://arxiv.org/abs/2509.21379), Cassano et al., 2025.
