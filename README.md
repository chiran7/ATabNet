# ATabNet: Attention and Feature-Refined TabNet for Interpretable Hyperspectral Image Classification

This repository contains the implementation of:

**ATabNet: Attention and Feature-Refined TabNet for Interpretable Hyperspectral Image Classification**

📄 Manuscript (IEEE GRSL, under submission):  
Will be made available after acceptance.

📊 Code and datasets:  
To be released at: https://github.com/chiran7/ATabNet

---

## Overview

ATabNet is an attention-enhanced TabNet architecture designed for hyperspectral image (HSI) classification. It integrates spatial–spectral attention mechanisms and feature refinement modules to improve discriminative representation learning while maintaining interpretability.

The method extends TabNet with:
- Convolutional Block Attention Module (CBAM)
- Gated feature fusion mechanism
- Convolutional feature refinement
- Improved decision-step feature aggregation

---

## Architecture

The framework enhances TabNet by introducing:
- Attention-based mask refinement
- Feature transformer with convolutional enhancement
- Step-wise decision aggregation for final classification

(Architecture figures are provided in the paper.)

---

## Installation

```bash
pip install pytorch-tabnet
pip install torch torchvision
