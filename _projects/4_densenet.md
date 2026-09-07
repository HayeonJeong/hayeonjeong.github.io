---
layout: page
title: "Lightweight DenseNet for EMNIST"
description: "Designing a compact DenseNet variant that beats ResNet-50 accuracy on EMNIST with ~10× fewer parameters"
img: assets/img/projects/4_densenet/thumbnail.png
importance: 2
category: 2024
related_publications: false
github: https://github.com/HayeonJeong/2024_Spring_ANN
_styles: >
  .post article table {
    margin: 0.5rem auto 1.75rem auto;
    border-collapse: collapse;
    border: 1px solid var(--global-divider-color);
  }
  .post article table th,
  .post article table td {
    border: 1px solid var(--global-divider-color);
    padding: 0.4rem 0.75rem;
  }
  .post article table thead th {
    border-bottom: 2px solid var(--global-divider-color);
  }
---

<div class="row justify-content-center">
  <div class="col-sm-10">
    <p>
      Artificial Neural Networks course project, Soongsil University (Mar – Jun 2024)<br />
      Team of 4 (team lead)
    </p>
    <p>
      <i class="fa-brands fa-github"></i>
      <a href="https://github.com/HayeonJeong/2024_Spring_ANN" target="_blank" rel="noopener noreferrer">Code repository</a>
    </p>
  </div>
</div>

---

## Goal

Using the **EMNIST** (extended MNIST) dataset, design or adapt a CNN classifier that finds a good balance between **test accuracy** and **inference time**. The challenge is not only to reach high accuracy but to do so with a model that stays small and fast.

---

## Data & Baselines

**Dataset.** We studied the EMNIST paper [[1](#ref-1)] and inspected each split. We chose **EMNIST-Balanced** (47 classes, equal per-class counts) as the training target to avoid the accuracy loss that comes from class imbalance, and reserved the larger **By_Merge** and **By_Class** splits (the same characters with many more images) for transfer learning. We split Balanced into train/validation/test and added a resizing utility.

**Baselines.** We trained two baselines and compared every later model against them:

| Baseline | Params | Train time | Test accuracy |
| --- | ---: | ---: | ---: |
| LeNet-5 [[2](#ref-2)] (ReLU/Softmax, early stopping + LR scheduler) | 65,104 | 171 s | 86.9% |
| ResNet-50 [[3](#ref-3)] (tuned batch size, LR schedule, activation, dropout, augmentation) | 23,684,015 | 1,449 s | 90.4% |

LeNet-5 is tiny and fast but cannot pass 90%; ResNet-50 is accurate but heavy and slow.

---

## Choosing the Base Model

We first shortlisted compact architectures (VGG16 [[4](#ref-4)], MobileNet, ShuffleNet, EfficientNet-B1) by considering model structure, parameter count, Top-1/Top-5 accuracy, and GPU inference speed. But most of them ran into a **vanishing-gradient problem** on EMNIST:

- the very large batch sizes we used, and
- the small EMNIST images: even a modest increase in depth over LeNet caused gradients from earlier layers to shrink toward zero.

The lesson was that **image size alone is not a good selection criterion.** We therefore looked for architectures where *earlier-layer information is preserved in later layers*: Wide ResNet [[5](#ref-5)], DPN [[6](#ref-6)], and **DenseNet** [[7](#ref-7)], where each layer receives the outputs of *all* previous layers.

We selected **DenseNet** as the base model: compared to Wide ResNet it delivers higher efficiency with fewer parameters, and compared to DPN it has a simpler structure that is easier to implement and maintain, while providing the same protection against vanishing gradients through maximal feature reuse.

---

## Hyperparameter Study (DenseNet-121 base)

Starting from DenseNet-121, we searched the training configuration systematically:

- **Learning rate & optimizer.** Swept combinations at batch size 2048, then re-ran the promising ones at 1024 (dropping the slow Nadam / RMSProp), settling on **Nesterov, lr = 0.001**.
- **Activation function.** Compared activations under the fixed configuration and measured Top-1/Top-5 accuracy plus **evaluation time** (full test pass) and **inference time** (new-input forward pass). **ELU** was chosen.
- **Dropout.** Tested seven placements (after the FC layer, after each dense block, after transition blocks, inside dense-block conv layers, after the first conv) and several rates.
- **Augmentation.** Compared five stacks (rotation → shift → distortion → zoom → brightness). Distortion cut evaluation/inference time, but the full stack doubled training time; we adopted the **four-transform stack** (rotation + shift + distortion + zoom, rotation capped at 10° to avoid confusing similar digits/letters), together with **ReduceLROnPlateau** and **early stopping** to keep training time in check.

---

## Designing the Lightweight Variant

The goal of the architecture search was a model with **fewer parameters, faster training, and shorter evaluation/inference time than DenseNet-121**, without losing accuracy. We varied the number of dense blocks, the number of convolution layers per block, and the transition-layer compression.

| Variant | Change from DenseNet-121 (6-12-24-16 conv layers per dense block) | Test accuracy |
| --- | --- | ---: |
| A | Conv layers per dense block reduced to **3-6-12-6** | 88.18% |
| A + 32×32 input | Same as A, input resized to (32, 32, 1) | **89.84%** |
| A-half | Conv layers reduced further to 2-4-8-4 (about half the parameters of A + 32×32 input) | 89.26% |

Variant A roughly **halved** evaluation/inference time. We then **transfer-learned** from the larger splits (modifying only the final layer for the new class count): pre-training on **By_Merge** at batch 2048 for 5 epochs, then fine-tuning on Balanced, was the most efficient. Over-shrinking the dense blocks (the A-half variant) capped the achievable accuracy.

**Final model.** A modified DenseNet with 3-6-12-6 dense blocks, (32, 32, 1) input, **ELU** activation, dropout, the four-transform augmentation, **Nesterov / lr 0.001 / batch 1024**, performance-based LR scheduling, and By_Merge transfer learning, for a total of **2,152,667 parameters**.

<div class="row justify-content-center">
  <div class="col-sm-7 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/4_densenet/architecture.png" title="Final model architecture" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Final architecture and parameter count. Four dense blocks (3-6-12-6) with ELU activations and concatenation, a global-average-pooling head, and a 47-way dense classifier, 2,152,667 total parameters (2,128,755 trainable).
</div>

---

## Final Results

| Model | Params | Train time | Test accuracy |
| --- | ---: | ---: | ---: |
| LeNet-5 | 65,104 | 171 s | 86.9% |
| ResNet-50 | 23,684,015 | 1,449 s (batch 2048) | 90.40% |
| **Ours (Lightweight DenseNet)** | **2,152,667** | 1,685 s (batch 1024) | **90.72%** |

- **Higher accuracy than both baselines**, at roughly **10× fewer parameters** than ResNet-50.
- Transfer learning contributed most to the final accuracy gain.
- Evaluation and inference time are about **2× faster** than the DenseNet-121 starting point.

<div class="row justify-content-center">
  <div class="col-sm-9 mt-3">
    {% include figure.liquid loading="eager" path="assets/img/projects/4_densenet/thumbnail.png" title="Accuracy vs. model size" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
Accuracy vs. parameter count on EMNIST-Balanced. The lightweight DenseNet reaches the highest accuracy with an order of magnitude fewer parameters than ResNet-50.
</div>

---

## Engineering Notes

- **Distributed training** on 4× NVIDIA A5000 (24 GB) GPUs with Keras' multi-GPU strategy [[8](#ref-8)].
- Reusable utilities (dataset loading, multi-GPU distribution, training/testing loops) were factored into separate Python modules so experiments could resume efficiently after kernel restarts.

---

## Lessons Learned

The dataset and the model must be considered **together**. A model chosen only by input size suffered from vanishing gradients, and the problem was solved by choosing an architecture (DenseNet) whose inductive bias matched the data. Careful tuning of activations and hyper-parameters then changed performance substantially, and the project gave us practical experience in matching a model and its training configuration to a new problem.

---

## References

<p id="ref-1">[1] <a href="https://arxiv.org/abs/1702.05373">EMNIST: an extension of MNIST to handwritten letters</a></p>

<p id="ref-2">[2] <a href="https://homl.info/lenet5">Gradient-Based Learning Applied to Document Recognition</a></p>

<p id="ref-3">[3] <a href="https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a></p>

<p id="ref-4">[4] <a href="https://arxiv.org/abs/1409.1556">Very Deep Convolutional Networks for Large-Scale Image Recognition</a></p>

<p id="ref-5">[5] <a href="https://arxiv.org/abs/1605.07146">Wide Residual Networks</a></p>

<p id="ref-6">[6] <a href="https://arxiv.org/abs/1707.01629">Dual Path Networks</a></p>

<p id="ref-7">[7] <a href="https://arxiv.org/abs/1608.06993">Densely Connected Convolutional Networks</a></p>

<p id="ref-8">[8] <a href="https://www.tensorflow.org/guide/keras/distributed_training">Distributed training with Keras (TensorFlow guide)</a></p>
