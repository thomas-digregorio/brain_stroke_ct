# Brain Stroke CT Classification System
> **Clinical-Grade AI for Rapid Ischemic & Hemorrhagic Stroke Detection**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Tracked-yellow.svg)](https://wandb.ai/)
[![Platform](https://img.shields.io/badge/Platform-WSL2%20|%20Vertex%20AI-blueviolet)](https://learn.microsoft.com/en-us/windows/wsl/)

## 🏥 Project Overview
This project implements a deep learning pipeline for the binary classification of brain CT scans (Normal vs. Stroke). Designed with a **clinical-first mindset**, it prioritizes high sensitivity (recall) to minimize missed diagnoses. The system features a robust training pipeline, automated experiment tracking, and a safety layer for Out-of-Distribution (OOD) detection.

**Key Objectives:**
*   **Clinical Literacy**: Primary metric is **F2 Score** (recall-weighted) rather than accuracy.
*   **High Performance**: Optimized for RTX 50-series GPUs using Mixed Precision (AMP) and CUDA 12.8.
*   **Safety**: Includes uncertainty estimation (Monte Carlo Dropout) and saturation checks to reject non-medical images.
*   **Scalability**: Designed for "Frontend-Backend" deployment on Google Cloud (Vertex AI + Cloud Run).

---

## ⚡ Tech Stack & Engineering
We leverage a modern MLOps stack to ensure reproducibility and performance:

*   **Core DL**: `PyTorch 2.5+`, `Torchvision`, `timm` (EfficientNet Backbones).
*   **Experiment Tracking**: `Weights & Biases (W&B)` for real-time loss curves, artifacts, and hyperparameter sweeps.
*   **Environment**: Developed on **WSL 2 (Ubuntu)** for native Linux kernel performance with direct GPU access.
*   **Visualization**: `JupyterLab` for extensive EDA (pixel histograms, class distribution analysis).
*   **Safety**: Custom OOD detectors in `utils/safety.py`.

---

## 🛠️ Usage

### 1. Environment Setup
Optimized for **NVIDIA RTX 5080** / **CUDA 12.9**.

```bash
# 1. Create Environment
conda create -n brain-stroke-dl python=3.10
conda activate brain-stroke-dl

# 2. Install PyTorch (Specific CUDA 12.8 Index)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install Dependencies
pip install -r requirements.txt
```

### 2. Training
Run the dedicated training script. This defaults to `EfficientNet-B4` and logs to W&B.

```bash
# Standard Training
python train_cls.py

# Arguments are handled via W&B Config or Hydra (in progress)
```

**Hardware Acceleration**:
The training loop uses `torch.amp.autocast('cuda')` for Automatic Mixed Precision. This reduces VRAM usage by ~40% and speeds up training on Tensor Cores without losing convergence stability.

### 3. Hyperparameter Sweeps (W&B)
We use Random Agent search to optimize Learning Rate, Batch Size, and Dropout.

```bash
# Initialize Sweep
wandb sweep configs/sweep.yaml

# Run Agent
wandb agent <SWEEP_ID>
```

### 4. Evaluation
Evaluate the clinically-relevant metrics (F2, Sensitivity, Specificity) on the held-out test set.

```bash
python eval_cls.py --model best_model.pth --split test
```

---

## 🛡️ Safety & Robustness
Medical AI must be trustworthy. This repo implements a "Safety Layer" (`utils/safety.py`) invoked before inference:

1.  **OOD Detection**: Rejects inputs with high channel variance (likely RGB natural images, not grayscale CTs).
2.  **Uncertainty Estimation**: Uses **Monte Carlo Dropout** inference (running multiple forward passes with dropout enabled) to quantify epistemic uncertainty.

---

## 🚀 Deployment Architecture
The production target is a microservices architecture:
*   **Inference Service**: Managed by **GCP Vertex AI** (Autoscaling GPU endpoints).
*   **Frontend**: A **Streamlit** dashboard (containerized via Docker) allowing clinicians to drag-and-drop scans.

*(Deployment modules located in `deploy/`)*

---

## 📂 Visualizations
Check `notebooks/eda.ipynb` for detailed analysis of the dataset:
*   Class Imbalance verification.
*   Pixel Intensity Histograms (Hounsfield Unit approximations).
*   Sample Grid Visualization.

---

**Author**: Thomas DiGregorio
**License**: MIT
