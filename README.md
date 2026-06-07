# 🚧 Road Extraction using Deep Learning (DeepGlobe Challenge)

This project focuses on **semantic segmentation of road networks from satellite imagery**, inspired by the DeepGlobe Road Extraction Challenge. The goal is to accurately extract road structures from high-resolution satellite images using deep learning models.

---

## 📊 Project Overview

Satellite road extraction is a challenging task due to:

- Thin and disconnected road structures
- Severe class imbalance (road vs background)
- High-resolution images (1024×1024)

This project explores multiple architectures:
- U-Net (baseline)
- U-Net++

with ResNet34 encoder backbone.

---

## 🧠 Models Used

- U-Net (ResNet34 encoder)
- U-Net++ (improved skip connections)

All models use pretrained ImageNet encoders.

---

## 📦 Dataset

- DeepGlobe Road Extraction Dataset
- Images: 1024×1024 satellite RGB images
- Masks: Binary road masks
- Train/Validation split: 80/20

---

## ⚙️ Training Setup

- Image size: 256–1024 (patch-based training used initially)
- Optimizer: AdamW
- Loss function: BCEWithLogits + Dice Loss
- Data augmentation:
  - Random crop
  - Horizontal/vertical flips
  - Rotations
- Mixed precision training (AMP)

---

## 📈 Results

| Model        | Validation IoU | Dice Score |
|-------------|---------------|------------|
| U-Net       | ~0.60         | ~0.74      |
| U-Net++     | ~0.61         | ~0.75      |

🏆 Best achieved model:
- **U-Net++ with augmentation**
- **IoU: 0.612**
- **Dice: 0.757**

---
