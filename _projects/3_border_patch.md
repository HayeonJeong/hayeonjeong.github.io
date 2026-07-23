---
layout: page
title: "Border Conditions in Adversarial Patch Attacks"
description: "How patch borders and interpolation change the attack success rate of adversarial patches on object detectors"
img: assets/img/projects/3_border_patch/thumbnail.png
importance: 1
category: 2024
related_publications: false
---

<div class="row justify-content-center">
  <div class="col-sm-10">
    <p>
      <i class="fa-solid fa-file-pdf"></i>
      <a href="{{ '/assets/pdf/border_conditions_adversarial_patch.pdf' | relative_url }}" target="_blank" rel="noopener noreferrer">
        Paper (original submission version)
      </a>
      &nbsp;·&nbsp; Undergraduate graduation research, Soongsil University (Jan – Oct 2024)
      &nbsp;·&nbsp; <strong>Best Paper Award (2nd Place)</strong>, Undergraduate Thesis Competition
    </p>
    <p style="font-size: 0.9rem; color: gray;">
      Note: the attached PDF is the anonymized submission version; venue and author information will be updated in a later revision.
    </p>
  </div>
</div>

---

## Abstract

The vulnerability of object detection models to adversarial attacks has led to extensive research, and adversarial patch attacks are a key method for distorting model predictions. Many factors affect the success rate of such attacks, but existing studies have focused primarily on **patch generation**, neglecting elements such as **patch borders** and **interpolation methods**.

This study analyzes how various *border conditions* affect the attack success rate (ASR) of adversarial patches on object detection models. Using patches generated from multiple datasets, we apply different border thicknesses, colors, and interpolation methods, and we also examine the effect of patch size and application ratio. Based on these experiments, we suggest concrete ways to maximize patch effectiveness, and we argue that these overlooked factors are important both for stronger attacks and for building more robust detectors.

---

## Motivation

Prior work on adversarial patches has focused on making patches **smaller** or **less noticeable**, adjusting patch characteristics through loss functions and transformations to raise the ASR. What has *not* been studied is the boundary between the background image and the patch — the pixels and **borders** created when a patch is actually applied to an image.

Existing studies are inconsistent here: some add borders, some do not, and physical patches are often displayed with wide black or white borders (e.g., patches shown on an LCD screen, or printed with a visible frame). Because adversarial patches optimize individual pixels in fine detail, an inconsistent border can actually make the patched object *easier* to detect, lowering the ASR and hurting reproducibility.

> **Our objective:** analyze how the presence, color, and thickness of borders — as well as the interpolation method used during application — affect the ASR, and show that the application process itself is a critical factor in attack effectiveness.

**Contributions.**

- We highlight the lack of consideration for patch borders in existing adversarial-patch research and emphasize the importance of **border consistency**.
- We show how to **maximize attack performance** by varying border thickness, border color, and interpolation method, and we analyze the impact of each factor on the ASR.
- We analyze how the border of a patch affects the ASR **depending on patch size and application ratio**.

---

## Approach: Analyzing Patch Borders

We systematically vary the conditions that an attacker can set as hyper-parameters or add arbitrarily when applying an existing patch:

**Interpolation methods.** To fit a patch to the object's bounding box, resizing (interpolation) is required, and each method treats the patch edges differently. A binary mask marks the object area with `1` (patch placed) and the outside with `0`.

- **Bilinear / bicubic** compute new pixels from neighboring values; because the mask sets outside pixels to zero, several zero-value pixels appear around the edge, producing a **black border**.
- **Nearest neighbor** copies the closest known pixel, filling the edge with the nearest patch value and leaving the border **less noticeable**.

**Patch sizes and ratios.** The initial patch size (in pixels) determines how much color and shape the patch can express; the application ratio is the fraction of the object the patch covers. Larger ratios generally raise the ASR, so we test with and without borders across sizes and ratios.

**Border thickness and color.** We vary thickness (kept small relative to the patch so the border is less prominent than the patch itself) and color (black, white, gray — colors close to typical patch colors).

<div class="row justify-content-center">
  <div class="col-sm-8 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/3_border_patch/border_grid.png" title="Border color and thickness examples" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Patch applications for a 256×256 patch under three border colors (white / gray / black) and three thicknesses (2, 8, 16 px). These are the border conditions whose effect on the attack success rate we measure.
</div>

---

## Experiment Design

We attack **human objects** in a digital environment — i.e., we assume a person is holding an adversarial patch.

**Datasets & detector.** We use MS-COCO (the *person* class, ~5k validation images) and the INRIA Person dataset (614 train / 288 test), and we attack two **YOLOv5** detectors, one trained on COCO-person and one on INRIA.

**Attacks.**

- **Hiding attacks** prevent the detector from finding the person. We generate four hiding patches — `COCO_COCO`, `INRIA_COCO`, `INRIA_INRIA`, `COCO_INRIA` (named *train_attack*) — starting from a gray patch and combining saliency, total-variation (TV), non-printability (NPS), and objectness losses (weights 1.0 / 2.5 / 0.1 / 3.0).
- **Altering attacks** make the person be recognized as another class (e.g., teddy bear, kite, traffic light). We use untargeted altering patches, choosing the lowest-confidence target class each epoch.

<div class="row align-items-center">
  <div class="col-md-6 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/3_border_patch/hiding_patches.png" title="Generated hiding patches" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-md-6 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/3_border_patch/altering_patches.png" title="Generated altering patches" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Left: <code>INRIA_INRIA</code> hiding patches at different application ratios. Right: untargeted altering patches optimized to be read as kite, teddy bear, and traffic light.
</div>

**Metrics.** The ASR is the fraction of detected *person* objects that are misclassified:

$$
\text{ASR}_{\text{hiding}} = \frac{N_{\text{person} \rightarrow \text{background}}}{N_{\text{person}}}, \qquad
\text{ASR}_{\text{altering}} = \frac{N_{\text{person} \rightarrow \text{target}}}{N_{\text{person}}}.
$$

---

## Results

### 1. Interpolation method

Bicubic (and bilinear) interpolation applies the patch **with a black border**; nearest-neighbor applies it **without** one. Across all four hiding patches, **bicubic gives a higher ASR** — up to a **2.4%** gap (e.g., 75.3% → 77.7% for `INRIA_INRIA` at ratio 0.2). Even small pixel changes from interpolation measurably affect the ASR, so **adding a thin black border can strengthen the patch.**

| Patch (256×256, ratio 0.2) | Bicubic (with border) | Nearest (no border) |
| --- | ---: | ---: |
| COCO_COCO | 39.7 | 38.2 |
| COCO_INRIA | 66.9 | 64.7 |
| INRIA_COCO | 32.5 | 31.2 |
| INRIA_INRIA | **77.7** | 75.3 |

### 2. Patch size and application ratio

The ASR *difference* between bordered and unbordered patches does **not** depend on patch size. It does depend on the **application ratio**: for hiding patches the difference varies by up to **9.8%** (mean **4.5%**), and since it is positive in almost every case, borders generally **increase** the ASR at the common application ratios of 0.1–0.3.

### 3. Border thickness

No single monotone trend holds across all patches. However, a **gray border of 4–8 pixels** consistently produced higher success rates across conditions, making it a useful default.

### 4. Border color

The best border color depends on the **dataset the patch was trained on**:

- `INRIA_COCO` (trained on COCO): **white** borders raised the ASR most (max **+9.8%**, mean +4.5%); black borders were neutral or worse.
- `INRIA_INRIA` (trained on INRIA): **gray** borders helped most (max **+4.9%**, mean +1.5%); white tended to hurt.

Because each patch is optimized to blend into its own training data, the border color that best matches the dataset's color characteristics maximizes the attack.

<div class="row justify-content-center">
  <div class="col-sm-10 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/3_border_patch/thumbnail.png" title="Hiding attack results by border color" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Hiding-attack results for an <code>INRIA_INRIA</code> patch under different border colors, ordered by confidence. A lower confidence score means a more successful attack (threshold 0.4). The attack succeeds most reliably with a <strong>gray</strong> border and least with a <strong>white</strong> border.
</div>

---

## Conclusion

This project addressed the often-overlooked issue of **patch borders** in adversarial-patch research and showed that the *application process* — not just patch generation — materially affects attack performance. Bicubic interpolation increases the ASR; patch size is unrelated to the border effect, while borders generally help at application ratios of 0.1–0.3; border thickness has no single trend, but a gray 4–8 px border is a strong default; and the optimal border color depends on the training dataset. These results give practical guidance for building stronger patches — and, in turn, for hardening object detectors against them.

*Future work:* validate these strategies across more datasets, detectors, and physical-world conditions.
