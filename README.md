# 🧠 Glaucoma Detection using Deep Learning (PAPILA Dataset)

This project implements a deep learning pipeline to classify fundus images as normal or glaucomatous using models like ResNet50, EfficientNetB3, and Vision Transformer (ViT), trained and fine-tuned on the PAPILA dataset. The model also computes Cup-to-Disc Ratio (CDR) using mask contours.

---

## 🔍 Overview

- 📅 **Timeline**: Completed over 3 days
- 🧾 **Dataset**: [PAPILA](https://figshare.com/articles/dataset/PAPILA/14798004?file=35013982)
- 🏷 **Problem**: Classify fundus images and calculate vertical CDR for glaucoma assessment
- 🎯 **Accuracy Achieved**: **85%**

---

## 🚀 Features

- ✅ Data cleaning and preprocessing of fundus images
- ✅ Overlay generation using optic cup/disc contour masks
- ✅ Vertical and horizontal CDR computation via contour-based geometry
- ✅ Transfer learning using:
  - ResNet50
  - EfficientNetB3
  - ViT (Vision Transformer)
- ✅ Ensemble approach for improved prediction

---

## 🛠 Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib
- Streamlit
- Git & GitHub

---

## 🖼 Sample Visuals

<p align="center">
  <img src="assets/sample_overlay.jpg" width="350">
  <img src="assets/sample_gradcam.jpg" width="350">
</p>

---

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<somritaghosh76>/glaucoma-detection-ml.git
   cd glaucoma-detection-ml
