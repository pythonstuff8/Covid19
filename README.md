<div align="center">

# 🫁 COVID-19 Chest X-Ray Classification

### Deep Learning Multi-Class Pulmonary Disease Detection Using InceptionV3 Transfer Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen?style=for-the-badge)](paper/paper.md)
[![Loss](https://img.shields.io/badge/Loss-0.10-blue?style=for-the-badge)](paper/paper.md)

<br>

**A deep learning model that classifies chest X-ray images into 5 pulmonary disease categories with 96% accuracy using transfer learning from InceptionV3.**

[📄 Read the Paper](paper/paper.md) · [🔬 View Results](#results) · [🚀 Quick Start](#quick-start) · [📊 Figures](#visualizations)

</div>

---

## 📋 Abstract

This project presents a deep learning-based approach for **multi-class classification of chest X-ray images** into five clinically relevant categories:

| Class | Description |
|-------|-------------|
| 🦠 **COVID-19** | SARS-CoV-2 infection with ground-glass opacities |
| 🔴 **Marila** | Measles-related pulmonary involvement |
| ✅ **Normal** | Healthy chest radiograph |
| 🫁 **Pneumonia** | Bacterial/viral pneumonia (non-COVID) |
| 🔬 **Tuberculosis** | Pulmonary TB with upper-lobe infiltrates |

The model leverages **transfer learning from InceptionV3** (pre-trained on ImageNet) via TensorFlow Hub, achieving **96% classification accuracy** with a **categorical cross-entropy loss of 0.10** on the held-out test set.

---

## 🏗️ Architecture

<div align="center">

![Model Architecture](figures/architecture_diagram.png)

</div>

| Component | Details |
|-----------|---------|
| **Backbone** | InceptionV3 (TensorFlow Hub, frozen) |
| **Input Size** | 299 × 299 × 3 |
| **Classification Head** | Dense(256, ReLU) → Dropout(0.5) → Dense(5, Softmax) |
| **Total Parameters** | 22,328,613 |
| **Trainable Parameters** | 525,829 |
| **Optimizer** | Adam (lr=0.001) |
| **Loss** | Categorical Cross-Entropy |

---

## 📊 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.0%** |
| **Test Loss** | **0.10** |
| Precision (Macro) | 95.8% |
| Recall (Macro) | 95.6% |
| F1-Score (Macro) | 95.7% |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | AUC |
|-------|-----------|--------|----------|-----|
| COVID-19 | 0.97 | 0.96 | 0.965 | 0.985 |
| Marila | 0.94 | 0.93 | 0.935 | 0.970 |
| Normal | 0.98 | 0.98 | 0.980 | 0.993 |
| Pneumonia | 0.95 | 0.96 | 0.955 | 0.978 |
| Tuberculosis | 0.95 | 0.95 | 0.950 | 0.975 |

---

## 📈 Visualizations

### Training Curves
![Training Curves](figures/training_curves.png)

### Confusion Matrix
![Confusion Matrix](figures/confusion_matrix.png)

### ROC Curves
![ROC Curves](figures/roc_curves.png)

### Class Distribution
![Class Distribution](figures/class_distribution.png)

### Sample Predictions
![Sample Predictions](figures/sample_predictions.png)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
```

### Installation

```bash
# Clone the repository
git clone https://github.com/thayyilsuhaan/Covid19.git
cd Covid19

# Install dependencies
pip install -r requirements.txt
```

### Generate Figures

```bash
python notebooks/training_evaluation.py
```

### Run the Web Application

```bash
python app.py
# Navigate to http://localhost:5000
```

### Docker Deployment

```bash
docker-compose up --build
```

---

## 📁 Repository Structure

```
Covid19/
├── app.py                          # Flask web application for inference
├── forms.py                        # WTForms for file upload
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker containerization
├── docker-compose.yaml             # Docker Compose config
├── 96_ac_0.1_Loss_covidmodel.h5    # Trained model weights (~483 MB)
│
├── paper/
│   └── paper.md                    # Full research paper
│
├── notebooks/
│   └── training_evaluation.py      # Figure generation & evaluation script
│
├── figures/
│   ├── training_curves.png         # Training/validation accuracy & loss
│   ├── confusion_matrix.png        # 5×5 normalized confusion matrix
│   ├── roc_curves.png              # One-vs-rest ROC curves
│   ├── architecture_diagram.png    # Model architecture visualization
│   ├── class_distribution.png      # Dataset class distribution
│   └── sample_predictions.png      # Sample X-rays with predictions
│
├── templates/                      # Flask HTML templates
├── static/                         # Static assets (CSS, images)
├── LICENSE                         # MIT License
└── CITATION.cff                    # Citation metadata
```

---

## 🔬 Methodology

1. **Transfer Learning**: InceptionV3 pre-trained on ImageNet serves as the feature extraction backbone. All base layers are frozen; only the custom classification head is trained.

2. **Data Preprocessing**: Chest X-ray images are resized to 299×299 pixels (InceptionV3 input size), converted to arrays, and batch-formatted for inference.

3. **Training**: The classification head is trained for 25 epochs using the Adam optimizer with a learning rate of 0.001 and 50% dropout regularization.

4. **Deployment**: The trained model is served via a Flask web application, containerized with Docker, enabling real-time chest X-ray classification through any web browser.

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{thayyil2026covid19cxr,
  author       = {Thayyil, Suhaan},
  title        = {Deep Learning-Based Multi-Class Classification of Chest X-Ray
                  Images for COVID-19 and Pulmonary Disease Detection Using
                  Transfer Learning with InceptionV3},
  year         = {2026},
  url          = {https://github.com/thayyilsuhaan/Covid19},
  version      = {1.0.0}
}
```

---

## 📚 Key References

1. Szegedy, C. et al. (2016). "Rethinking the Inception Architecture for Computer Vision." *CVPR*.
2. Wang, L. & Wong, A. (2020). "COVID-Net: Detection of COVID-19 from Chest X-ray Images." *Scientific Reports*.
3. Rajpurkar, P. et al. (2017). "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-rays." *arXiv*.
4. Apostolopoulos, I.D. & Mpesiana, T.A. (2020). "COVID-19: Automatic Detection from X-ray Images Utilizing Transfer Learning." *Physical and Engineering Sciences in Medicine*.

---

## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

> This model is intended for **research and educational purposes only**. It is **NOT** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for medical decisions.

---

<div align="center">

**Made with ❤️ by [Suhaan Thayyil](mailto:thayyilsuhaan@gmail.com)**

</div>
