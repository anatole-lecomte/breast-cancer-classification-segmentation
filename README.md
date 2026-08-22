# Breast Cancer Classification & Segmentation from Ultrasound Images

Deep learning pipelines for classifying breast ultrasound images (benign / malignant / normal) and segmenting tumor regions, comparing two different pipeline orderings: classify-then-segment vs. segment-then-classify.

*University project — Applied AI in Biomedicine, Politecnico di Milano (2025–2026). Team of 3 students.*

## Overview

Breast cancer is the most commonly diagnosed cancer in women worldwide, and ultrasound is an increasingly used, accessible imaging modality — but one that is highly operator-dependent. This project explores how AI can assist by automating both the detection (classification) and localization (segmentation) of tumors from ultrasound images.

Rather than training a single end-to-end model, the project deliberately separates the two tasks — training and evaluating classification and segmentation models independently first — before comparing two ways of combining them into a full diagnostic pipeline:

1. **Classify → Segment**: classify each image first (benign / malignant / normal), then segment only the pathological ones. Mimics a doctor's diagnostic workflow.
2. **Segment → Classify**: segment every image first (as a region-proposal step), then classify benign vs. malignant using the masked region. Reduces the risk of a classifier missing a tumor outright.

**Dataset**: 1,531 ultrasound images from 968 patients (706 benign, 460 normal, 365 malignant), stratified 90/10 train/validation split.

## Key results

| Classification model | Accuracy | Balanced Accuracy | F1 (weighted) |
|---|---|---|---|
| CNN from scratch | 70.2% | 67.7% | 69.7% |
| ResNet50 (transfer learning) | **75.5%** | **75.2%** | **75.2%** |

| Segmentation model | Mean Dice | Mean IoU |
|---|---|---|
| U-Net (from scratch) | 0.55 | 0.47 |
| MaNet | 0.77 | 0.71 |
| **DeepLabV3** | **0.79** | **0.73** |
| SegFormer | 0.73 | 0.66 |

**Transfer learning (ResNet50) and DeepLabV3 were selected as the final models** for classification and segmentation respectively, based on consistently superior validation performance.

**Classify-first vs. segment-first**: the two pipelines trade off differently. Classify-first is more computationally efficient but creates a hard bottleneck — a missed tumor at the classification stage never reaches the segmenter. Segment-first is more robust to this specific failure mode (lower false-negative rate) since every image gets a segmentation attempt regardless of the classifier, but couples segmentation quality directly into the final classification, and loses the surrounding tissue context that classify-first pipelines can use. In a clinical context, minimizing false negatives is critical, which favors the segment-first ordering despite its added complexity.

## Method highlights

- **Preprocessing**: task- and model-specific resizing (128×128 for from-scratch classification, 224×224 for transfer learning / U-Net / DeepLabV3, 512×512 for MaNet / SegFormer), model-specific normalization (raw vs. ImageNet statistics), stratified splitting, and an Albumentations-based augmentation pipeline (horizontal flip, random translation/scale/rotation) applied identically to images and masks.
- **Classification**: a CNN built from scratch (with global average pooling to control overfitting) compared against ResNet50 transfer learning with a frozen backbone and a custom classification head.
- **Segmentation**: four architectures compared — a from-scratch U-Net, and pretrained MaNet, DeepLabV3, and SegFormer — evaluated with Dice and IoU, with morphological post-processing (hole filling, opening/closing) applied to clean predicted masks.
- **Training**: Adam optimizer with `ReduceLROnPlateau` scheduling and early stopping across all models, with class-weighted vs. unweighted loss compared for the classification task.
- **Two-stage pipeline comparison**: both pipeline orderings were implemented end-to-end and evaluated with 3×3 (or 2×2) confusion matrices to understand exactly where each pipeline fails.

## Tech stack

`PyTorch` · `torchvision` (ResNet50) · `segmentation_models_pytorch` (MaNet, SegFormer) · `DeepLabV3` (COCO-pretrained) · `Albumentations` · `scikit-learn` (metrics)

## Repository structure

```
.
├── README.md
├── Applied_AI_project.ipynb   # Full notebook: preprocessing, models, training, both pipelines
└── report/
    └── Applied_AI_Report.pdf   # Full write-up: methods, all metrics tables, confusion matrices
```

The notebook was built for Google Colab (Drive-mounted dataset, GPU runtime) and expects the dataset structure described in the assignment (`training_images/` + `training_metadata.xlsx`), which is not included in this repository as it is course-provided data.

## Authors
Politecnico di Milano - Applied AI in Biomedicine

- DAU Lara
- LECOMTE Anatole
- LUNEAU Nathan


Supervised by Prof. Valentina Corino, TA Meri Ferretti
