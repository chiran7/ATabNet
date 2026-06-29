# ATabNet: Attention and Feature-Refined TabNet for Hyperspectral Image Classification

**ATabNet (IEEE GRSL submission)**

## Overview
ATabNet is an attention-enhanced TabNet architecture for hyperspectral image (HSI) classification. It improves spatial–spectral feature learning using attention and feature refinement while preserving interpretability.

Main components:
- Convolutional Block Attention Module (CBAM)
- Gated feature fusion
- Convolutional refinement
- Enhanced decision-step aggregation


## Installation
```bash
pip install pytorch-tabnet torch torchvision

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
