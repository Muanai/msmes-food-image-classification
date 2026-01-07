# High-Performance Indonesian Food Image Classification (15 Classes)

> **A State-of-the-Art pipeline leveraging Vision Transformers (SwinV2), ConvNeXtV2, and EfficientNetV2 with Semi-Supervised Teacher-Student Learning.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

##  Overview

This repository contains the source code for my submission in a *Data Mining Competition. The goal of the project was to develop a robust computer vision model capable of classifying **15 types of Indonesian traditional cuisine*.

This solution was designed to assist *MSMEs (UMKM)* in digitizing their inventory and sales processes through automated image recognition. The final model achieved a *Leaderboard Score of 0.95222*, securing a top 10 position.

> ⚠️ *Disclaimer:* The dataset used in this competition is proprietary and protected by the competition rules. Therefore, the image data *cannot be shared publicly* in this repository. The notebook is also specifically made to be run on Kaggle Notebook provided by the organizer
## Key Architectures & Strategies

We employ a diverse set of backbones to maximize ensemble variance:

| Model Architecture | Variant | Training Strategy | Key Features |
| :--- | :--- | :--- | :--- |
| **Swin Transformer V2** | `swinv2_large_window12` | Supervised | Mixup, CutMix, EMA, AMP |
| **ConvNeXt V2** | `convnextv2_large.fcmae` | **Teacher-Student (SSL)** | Pseudo-Labeling, K-Fold, Masked Autoencoder Pretraining |
| **EfficientNet V2** | `efficientnetv2_l` | Supervised | Strong Augmentation, Label Smoothing |

### The Winning Strategy
After extensive experimentation with various backbones (including EfficientNetV2), the final top-performing ensemble prioritized a specific combination of Vision Transformers and ConvNets.

*Final Ensemble Composition:*
1.  *Swin Transformer V2 Large* (1 Model)
2.  *ConvNeXt V2 Large* (3 Student Models from different folds)

Note: While EfficientNetV2 was explored during development, it was excluded from the final submission as the Swin + ConvNeXt combination yielded better geometric alignment and higher accuracy.

### Advanced Techniques Used
* **Semi-Supervised Learning (SSL):** Implementation of a Teacher-Student framework where high-confidence predictions from Teacher models are used as pseudo-labels to train a robust Student model.
* **Geometric Ensemble:** Final predictions are aggregated using weighted geometric means with Temperature Scaling ($T=1.6$) to calibrate probabilities.
* **Test Time Augmentation (TTA):** 4-view inference (Resize, Flip, ColorJitter, RandomCrop) to stabilize predictions.
* **Regularization:** Label Smoothing (0.1), DropPath, and Weight Decay.

## 📂 Project Structure

```bash
.
├── train_swinv2_large.ipynb       # Training script for SwinV2 (Mixup/CutMix + EMA)
├── train_convnextv2_large.ipynb   # Full SSL Pipeline: Teacher Train -> Pseudo Label -> Student Train
├── train_efficientnetv2_l.ipynb   # Training script for EfficientNetV2
├── ensemble.ipynb                 # Geometric Ensemble & TTA Inference script
└── README.md
```

## Configuration & Hyperparameters
 * Image Size: 384x384 (Swin/ConvNeXt), 480x480 (EffNet)
 * Optimizer: AdamW (weight_decay=1e-2)
 * Schedulers: Cosine Annealing Warm Restarts & OneCycleLR
 * Augmentations:
   * Geometric: RandomResizedCrop, HorizontalFlip, Rotation, Perspective
   * Pixel-level: ColorJitter, GaussianBlur, RandomErasing
 
## Usage
1. **Training the Models**: Run the notebooks in the following order to generate the model weights:
   * Swin Transformer: Run train_swinv2_large.ipynb.
     * Output: swin_large_mixcut_ema_best.pth
   * EfficientNet: Run train_efficientnetv2_l.ipynb.
     * Output: efficientnetv2_l_best.pth
   * ConvNeXt V2 (Teacher-Student): Run train_convnextv2_large.ipynb.
     * Note: This script performs K-Fold training for teachers, generates pseudo-labels, and then trains the student models.
     * Output: models_student/best_model_fold_X.pth
2. **Inference (Ensemble)**: Run ensemble.ipynb. This script loads the best weights from all architectures and performs the geometric ensemble.
```bash
## Ensemble Weights (based on Validation Accuracy)
VAL_ACCS = [0.9190, 0.9118, 0.9201, 0.9239]
## Aggregation: Weighted Geometric Mean
```

## Dependencies
 * torch >= 2.0.0
 * torchvision
 * timm (PyTorch Image Models)
 * pandas, numpy, scikit-learn
 * Pillow, tqdm

## Author
[Muanai K. Revindo]
 * Informatics Engineering Student @ Universitas Sriwijaya
 * Interest: AI, Big Data, Financial Engineering
Feel free to use this repository if you find it useful!
