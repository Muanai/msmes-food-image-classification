# Architecture Overview

This repository implements a high-performance image classification pipeline developed for the **Action Data Mining (Kaggle)** competition, achieving **Top 10 / 131 teams** on the public leaderboard.

The system is designed around three core principles:
1. Architectural diversity over single-model dominance
2. Aggressive regularization to combat limited labeled data
3. Stability-first training under constrained compute (Kaggle T4)

---

## High-Level Pipeline

Raw Images  
→ Augmentation & Regularization  
→ Backbone Models (ConvNeXtV2, SwinV2)  
→ Semi-Supervised Expansion (Teacher–Student)  
→ Ensemble Inference (Weighted Geometric Mean + TTA)  
→ Final Predictions

---

## Data Characteristics

- **Task**: Multi-class image classification
- **Classes**: 15 Indonesian food categories
- **Dataset size**: ~4,200 images
- **Label availability**: Partial (semi-supervised setup)
- **Class balance**: Relatively balanced
- **Image resolution**: Highly variable (normalized during preprocessing)

The dataset size and label sparsity motivated strong regularization and pseudo-labeling strategies rather than scaling depth alone.

---

## Model Backbones

Two complementary architectures are used:

### ConvNeXtV2-Large (FCMAE pretraining)
- Pretrained on ImageNet-22K → fine-tuned on ImageNet-1K
- Strong local feature extraction
- Serves as both **standalone classifier** and **Teacher model**

### SwinV2-Large
- Window-based self-attention with dynamic resolution
- Pretrained on ImageNet-22K → fine-tuned on ImageNet-1K
- Strong at capturing global spatial context

These models were selected to maximize **inductive bias diversity** rather than marginal gains from similar architectures.

---

## Training Strategy

### Augmentation & Regularization
- **Mixup & CutMix**: Enabled across all models
- **RandAugment**: Applied to ConvNeXtV2 (num_ops=2, magnitude=9)
- **Standard augmentations**:
  - RandomResizedCrop
  - HorizontalFlip
  - ColorJitter
  - RandomErasing
  - GaussianBlur

### Loss Functions
- **Soft Target Cross Entropy** during Mixup/CutMix
- **Label Smoothing (ε = 0.1)** during standard training

Loss selection dynamically adapts to augmentation mode to stabilize gradients.

---

## Optimization

- **Optimizer**: AdamW (weight decay ≈ 0.01)
- **Schedulers**:
  - SwinV2: Custom cosine decay with warmup (LambdaLR)
  - ConvNeXtV2:
    - OneCycleLR for head stabilization
    - CosineAnnealingWarmRestarts for backbone fine-tuning
- **EMA**: Enabled via `ModelEmaV2` across all major models

Epoch ranges are intentionally short (12–25 epochs) to avoid overfitting under limited data.

---

## Semi-Supervised Learning

A **Teacher–Student framework** is employed:

1. ConvNeXtV2 Teacher trained using 3-fold cross-validation
2. Teacher generates pseudo-labels on unlabeled samples
3. Predictions with confidence ≥ **0.87** are accepted
4. Adaptive fallback threshold down to **0.75** if pseudo-label volume is insufficient
5. Student model trained on combined labeled + pseudo-labeled data

This approach improves generalization without introducing excessive label noise.

---

## Inference & Ensemble

### Ensemble Strategy
- **Weighted Geometric Mean** of probability outputs
- Implemented via log-space aggregation, where:
  `p_i` is the predicted probability from model *i*
  and `w_i` is the normalized ensemble weight
- Penalizes low-confidence predictions more effectively than arithmetic mean

### Test-Time Augmentation (4 views)
1. Standard resize + center crop
2. Horizontal flip
3. Color jitter (brightness/contrast)
4. RandomResizedCrop (scale 0.92–1.0)

TTA improves robustness under scale and illumination variance.

---

## Compute Constraints

- **Training environment**: Kaggle Notebook
- **GPU**: NVIDIA T4
- **Local hardware**: GTX 1650 (insufficient for large-scale training)

Notebook-based execution is a deliberate trade-off driven by hardware constraints, not a design preference.

---

## Summary

This architecture prioritizes **decision quality over architectural novelty**, focusing on:
- Stability under small data
- Controlled semi-supervised expansion
- Ensemble diversity with principled aggregation

The result is a robust and reproducible system capable of competing at the top tier under limited compute.
