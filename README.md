# Plant Disease Classification & Taxonomy Pipeline 🌿🧪

An end-to-end Computer Vision pipeline and Deep Learning classification framework built on the **PlantVillage** dataset. This project structures crop disease taxonomies, applies robust data preprocessing and augmentation strategies, and trains a custom Convolutional Neural Network (CNN) achieving **>96% validation accuracy** across multiple plant disease classes.

---

## 📌 Features & Highlights

* **Taxonomy Structuring:** Multi-level categorical hierarchy mapping plant species, target pathogens, and disease severity classes.
* **Data Curation Pipeline:** Automated dataset verification, corrupt image filtering, standardizing resolutions ($256 \times 256$), and split balancing ($70\%$ Train, $15\%$ Validation, $15\%$ Test).
* **Augmentation Pipeline:** Robust geometric and photometric transformations to eliminate spatial bias and prevent overfitting.
* **Custom Deep Convolutional Architecture:** Feature extraction via sequential Conv2D-ReLU-BatchNorm-MaxPool blocks with progressive Dropout regularizers.
* **Comprehensive Evaluation:** Detailed training/validation loss/accuracy curves, confusion matrix heatmaps, precision/recall/F1-score breakdowns, and benchmark comparisons.

---

## 📁 Taxonomy & Dataset Curation

The framework processes raw RGB leaf imagery and categorizes images into a standardized 3-level taxonomy:

$$\text{Plant Specimen} \longrightarrow \text{Pathogen / Condition Category} \longrightarrow \text{Class Label ID}$$

### Dataset Split Distribution

| Split | Percentage | Purpose |
| :--- | :--- | :--- |
| **Train** | $70\%$ | Optimization and weight updates via backpropagation |
| **Validation** | $15\%$ | Real-time loss/accuracy monitoring & hyperparameter tuning |
| **Test** | $15\%$ | Final un-biased performance evaluation |

### Online Data Augmentation Strategy

To ensure strong generalization across ambient lighting and leaf orientation variations, the training pipeline applies:
* **Random Horizontal & Vertical Flips** ($p=0.5$)
* **Rotation:** $\pm 20^\circ$ random rotations
* **Zoom / Scale:** $\pm 15\%$ scale jittering
* **Brightness & Contrast Adjustments:** $\pm 10\%$ photometric perturbation
* **Normalization:** Scaling pixel values from $[0, 255]$ to $[0.0, 1.0]$

---

## 🏗️ Model Architecture & Hyperparameters

The core classifier uses a custom multi-stage Convolutional Neural Network designed for optimal balance between feature capacity and spatial reduction:
