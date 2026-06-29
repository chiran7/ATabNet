# ATabNet: Attention and Feature-Refined TabNet for Hyperspectral Image Classification

This repository contains the implementation of:

**ATabNet: Attention and Feature-Refined TabNet for Interpretable Hyperspectral Image Classification (IEEE GRSL, under submission)**

📌 Code and data: https://github.com/chiran7/ATabNet  
📌 Paper: To be updated upon publication in IEEE GRSL


## Overview
ATabNet is an attention-enhanced TabNet model for hyperspectral image (HSI) classification. It improves feature representation by integrating spatial–spectral attention and feature refinement modules while maintaining interpretability.

Key components:
- Convolutional Block Attention Module (CBAM)
- Gated feature fusion
- Convolutional feature refinement
- Enhanced decision-step feature aggregation

Training: 
python train_ATabNet_HSI.py

Citation

If you use this code, please cite:
@article{atabnet2026,
  title={ATabNet: Attention and Feature-Refined TabNet for Hyperspectral Image Classification},
  author={Shah, Chiranjibi and Du, Qian and others},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2026},
  note={Under submission}
}
## Installation
```bash
pip install pytorch-tabnet torch torchvision
