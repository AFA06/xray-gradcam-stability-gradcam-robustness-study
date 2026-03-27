# Investigating the Stability of Grad-CAM Explanations in Chest X-Ray Classification

## 📌 Overview
This project investigates the **stability and reliability of Grad-CAM explanations** in deep learning models for chest X-ray classification under controlled input perturbations.

While convolutional neural networks (CNNs) achieve high diagnostic accuracy, their interpretability remains a major challenge. This study evaluates whether Grad-CAM explanations remain consistent when input images undergo small, realistic transformations such as brightness, contrast, and rotation changes.

---

## 🎯 Objectives
- Train a multi-label classification model on chest X-ray data
- Generate Grad-CAM visual explanations for predictions
- Apply controlled perturbations to input images
- Measure explanation stability using quantitative metrics
- Compare **prediction stability vs explanation stability**
- Perform statistical analysis to validate findings

---

## 🧠 Key Findings
- Model predictions remain **stable** under small perturbations
- Grad-CAM explanations are **unstable**, especially under rotation
- Explanation stability ≠ Prediction stability
- Highlights limitations of gradient-based interpretability methods in medical AI

---

## 🗂️ Project Structure

```bash
chexpert/
│
├── data/                     # Dataset (images + CSV labels)
├── src/
│   ├── data/                 # Dataset class
│   ├── models/               # ResNet18 model
│   └── explainability/       # Grad-CAM implementation
│
├── scripts/                  # Experiment scripts
├── results/                  # Output results, figures, tables
├── gradcam_results/          # Generated heatmaps
├── checkpoints/              # Saved model weights
│
├── train.py                  # Training pipeline
├── README.md
