# Multi-label Attribute Classification Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## 📌 Project Overview
This repository contains a deep learning solution for a **multi-label classification** task. The objective is to identify four distinct attributes within a dataset of images while addressing real-world data challenges such as **missing labels (NA values)** and **skewed class distributions**.

### Key Deliverables:
1. **Model Weights:** `deep-model.pth`
2. **Loss Visualization:** `loss_curve.png`
3. **Inference Script:** `inference.py`
4. **Training Logic:** Handled within the provided notebook/script.

---

## 🛠️ Technical Implementation

### 1. Handling Missing Labels (NA)
A common challenge in this dataset was the presence of "NA" tags. To avoid losing valuable data, I implemented a **Masked Binary Cross-Entropy Loss**.
* **Mechanism:** Labels are mapped as `1` (Present), `0` (Absent), and `-1` (NA). 
* **Logic:** The custom loss function generates a binary mask where $Targets \geq 0$. During backpropagation, gradients from "NA" attributes are zeroed out, ensuring the model only learns from verified annotations.

### 2. Imbalance Mitigation
To prevent the model from biasing toward the majority class, I calculated **Positional Weights** for the `BCEWithLogitsLoss`.
* **Formula:** $\text{weight} = \frac{\text{negative\_samples}}{\text{positive\_samples}}$
* **Impact:** This increases the penalty for misclassifying minority attributes, improving the overall F1-score and recall across all four categories.

### 3. Architecture
- **Backbone:** ResNet50 (Pre-trained on ImageNet-1K).
- **Modification:** The final fully connected layer was replaced with a 4-unit linear layer to match the attribute count.
- **Optimization:** Adam Optimizer with a learning rate of $1 \times 10^{-4}$.



---

## 📂 Repository Structure
```text
├── images/               # Directory containing the dataset
├── labels.txt            # Original annotation file
├── deep-model.pth        # Trained weights (Deliverable)
├── loss_curve.png        # Training loss visualization (Deliverable)
├── inference.py          # Modular script for testing new images
└── README.md             # Project documentation
