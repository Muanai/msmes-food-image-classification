# Design Decisions & Trade-offs

This document outlines the key architectural and training decisions made during the development of this system, including alternatives that were tested and ultimately discarded.

The focus is on **why decisions were made**, not merely what was implemented.

---

## Why ConvNeXtV2 and SwinV2?

Rather than stacking similar CNNs, the final system combines:
- **ConvNeXtV2** → strong locality and texture bias
- **SwinV2** → hierarchical attention and global context

This pairing provides complementary failure modes, which is more valuable for ensembling than marginal gains from homogeneous architectures.

---

## Why EfficientNetV2-L Was Dropped

**EfficientNetV2-L (ImageNet-21K pretraining)** was initially evaluated but removed due to:

- Faster overfitting compared to ConvNeXtV2
- Higher sensitivity to aggressive augmentation
- Weaker marginal contribution to ensemble diversity

Despite competitive standalone accuracy, its inclusion reduced ensemble stability.

Decision: **Dropped in favor of robustness over peak single-model score.**

---

## Why Aggressive Mixup & CutMix Everywhere?

Given the relatively small labeled dataset (~4.2K images):
- Overfitting risk dominated underfitting risk
- Label interpolation proved more effective than deeper regularization stacks

Mixup/CutMix remained enabled across all architectures, with loss functions adapted accordingly.

---

## Loss Function Switching

Using a single loss across all training phases caused instability:
- Hard labels conflicted with Mixup targets
- Soft labels conflicted with non-augmented phases

Decision:
- **Soft Target Cross Entropy** during Mixup/CutMix
- **Label Smoothing (ε = 0.1)** otherwise

This reduced gradient noise and improved convergence consistency.

---

## Teacher–Student Instead of More Data Augmentation

Synthetic augmentation alone plateaued early.
Semi-supervised expansion provided:
- More effective decision boundary refinement
- Controlled risk via confidence thresholding

A confidence threshold of **0.87** was chosen empirically to balance:
- Label quality
- Sample coverage

Adaptive fallback to 0.75 prevents data starvation.

---

## Why Geometric Mean Ensemble?

Arithmetic mean (standard soft voting) was tested but discarded because:
- It over-rewards uncertain predictions
- It fails to penalize model disagreement

Geometric mean:
- Suppresses low-confidence outputs
- Rewards consensus
- Produces more calibrated probabilities

This choice improved leaderboard stability across submissions.

---

## Why Notebook-Based Training?

This is a constraint-driven decision:
- Kaggle T4 GPU enabled models infeasible on local GTX 1650
- Training time window: **Oct 8 – Nov 11**

Given these limits, reproducibility and iteration speed took priority over code modularity.

With more compute, this pipeline would be migrated to a script-based training framework.

---

## What Would Be Improved With More Time / Resources

- Convert training pipeline to `.py` modules
- Introduce k-fold ensemble across architectures
- Explore confidence-aware pseudo-label weighting
- Add calibration metrics (ECE) for ensemble outputs

---

## Closing Note

Every design choice in this project reflects a trade-off between:
- Performance
- Stability
- Compute constraints

The final system favors **reliability and reasoning** over architectural novelty — a principle that aligns closely with real-world ML engineering.
