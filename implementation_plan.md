# Implementation Plan - Brain Stroke CT Classification

## Goal Description
Train a clinically literate binary classifier (Stroke vs No Stroke) on 2D CT slices. The system will use an EfficientNet backbone, prioritize sensitivity (recall), and use robust experiment tracking via W&B.
**Deployment**: A "Frontend-Backend" architecture where the model is hosted on Google Vertex AI for scalable inference, and a Streamlit UI (hosted on Cloud Run) consumes this endpoint.

## User Review Required
> [!IMPORTANT]
> **Dataset Status**: The workspace directory `c:\Users\thoma\Documents\Stroke_CT_DL` is currently empty. You will need to download/unzip the Kaggle dataset into this folder, or provide the path to where it is located.

> [!NOTE]
> **No Augmentation**: Per instructions, we will strictly avoid geometric (rotation, flip) and photometric (contrast) augmentations in the initial phase to establish a clean baseline.

## Security & Compliance
> [!CAUTION]
> **Credential Safety**:
> - **Git Ignore**: A strict `.gitignore` must be created immediately to exclude `*.key`, `.env`, `kaggle.json`, and `wandb/`.
> - **API Keys**: Keys for W&B and GCP must be env vars. Kaggle can use `kaggle.json` (must be gitignored).
> - **Verification**: Before pushing to GitHub, verify `git status` to ensure no sensitive files are tracked.

## Proposed Changes

### Project Structure
We will create a modular Python project structure.

#### [NEW] [notebooks/eda.ipynb](file:///c:/Users/thoma/Documents/Stroke_CT_DL/notebooks/eda.ipynb)
- **Visuals**: Class distribution (No Stroke vs Ischemia vs Bleeding), sample grid, pixel intensity histograms.
- **Verification**: Check class imbalance and mask consistency.

#### [NEW] [requirements.txt](file:///c:/Users/thoma/Documents/Stroke_CT_DL/requirements.txt)
- `torch>=2.5.1` (Requires CUDA 12.8+ wheel for RTX 5080 support).
- `torchvision>=0.20.1`, `timm>=1.0.11` (Modern backbones).
- `pandas`, `numpy`, `scikit-learn`, `opencv-python-headless`.
- `wandb>=0.18.0`, `streamlit>=1.40.0`.
- `jupyterlab`, `ipykernel` (For EDA).

#### [NEW] [utils/common.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/utils/common.py)
- **Reproducibility**: `seed_everything(seed=42)` locking `random`, `np`, `torch` (and `cuda` deterministic).

#### [NEW] [data.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/data.py)
- **Dataset Class**: Custom `Dataset` class (`StrokeDataset`).
    - **Label Logic**: Supports both Folder-based (Bleeding/Normal) AND CSV-based (`labels.csv` with `image_id`, `Stroke`) loading.
- **Preprocessing**:
    - **Resize**: None (Keep native 512x512).
    - Convert to FloatTensor.
    - Normalize to [0, 1].
    - Standardize using ImageNet mean/std.
- **Splitting Logic**:
    - **Strategy**: Stratified Shuffle Split (70/15/15).
    - *Note*: Patient IDs unavailable, so patient-level leakage is a known limitation.

#### [NEW] [models.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/models.py)
- **Backbone**: `timm.create_model('efficientnet_b4', features_only=True)`.
    - **Resolution**: Native 512x512 (No resizing, preserves all detail).
- **Classifier**: `StrokeClassifier` class.
    - Uses backbone features -> GAP -> Linear.
- **Segmentor**: `StrokeSegmentor` class (Future Phase).
    - Uses backbone features -> Decoder (U-Net style) -> Mask.

#### [NEW] [utils/metrics.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/utils/metrics.py)
- **Clinical Metrics**: Sensitivity, Specificity, F2, Calibration.
- **Seg Metrics**: Dice Coefficient, IoU (stubbed for later).

#### [NEW] [train_cls.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/train_cls.py)
- Dedicated training loop for Binary Classification.
- **Loss**: `BCEWithLogitsLoss` + **Label Smoothing** (0.1).
- **Logging**: Step-wise W&B logging (loss per batch) + Epoch-wise validation metrics.
- **Validation**: 5-Fold Cross-Validation support.

#### [NEW] [configs/sweep.yaml](file:///c:/Users/thoma/Documents/Stroke_CT_DL/configs/sweep.yaml)
- **W&B Sweep Config**: Automated hyperparameter tuning (**Random Search**).
- **Parameters**: Learning rate, Weight decay, Label smoothing factor, Batch size.

#### [NEW] [eval_cls.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/eval_cls.py)
- Dedicated evaluation for Classification.

### Safety & Robustness (Clinical "Trust" Layer)

#### [NEW] [utils/safety.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/utils/safety.py)
- **Uncertainty Estimation**: Monte Carlo Dropout wrapper. Run inference 10x with dropout on to get mean/std of prediction. High std = "Not sure".
- **Out-of-Distribution (OOD)**:
    - **Saturation Check**: Reject images with high mean color saturation (likely non-medical images).
    - *Note: Histogram/Resize checks omitted per design review.*

### Deployment Phase (GCP Vertex AI + Streamlit)

> [!TIP]
> **W&B + Vertex AI Integration**: W&B will be used for *Experiment Tracking*. Vertex AI is for *Serving*.
> 1. Train and identify best run in W&B.
> 2. Download `best_model.pth`.
> 3. Register to Vertex AI Model Registry.

#### [NEW] [deploy/handler.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/deploy/handler.py)
- Prediction handler for Vertex AI.
- **Includes OOD check** before model inference.

#### [NEW] [deploy/full_local_demo.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/deploy/full_local_demo.py)
- Alternative to Cloud: Runs Streamlit + Model locally on your RTX 5080.
- **Sample Gallery**: Sidebar with 10 clickable thumbnails (Stroke/Normal/OOD) to auto-load for easy testing.
- Useful for free development before deploying to paid GCP.

#### [NEW] [deploy/app.py](file:///c:/Users/thoma/Documents/Stroke_CT_DL/deploy/app.py)
- **Streamlit UI**:
    - Drag-and-drop file uploader.
    - Visualization of original image.
    - Request logic: Sends image to Vertex AI Endpoint.
    - Display: "Stroke Detected" vs "Normal" + Confidence Score.

#### [NEW] [deploy/Dockerfile](file:///c:/Users/thoma/Documents/Stroke_CT_DL/deploy/Dockerfile)
- Docker integration for both the Serving container (if custom) and the Streamlit frontend.

## Verification Plan

### Automated Tests
- **Shape Checks**: Verify tensor shapes passing through model.
- **Leakage Check**: Assert no image overlap between Train and Test sets.

### Manual Verification
- **Sanity Check**: Inspect a batch of preprocessed images before training (visualize normalized values).
- **Overfit Test**: Train on 1 batch to verify loss goes to ~0.
- **Clinical Sanity**: Verify that "background" or skull regions are not the primary focus of Grad-CAM heatmaps.
- **Deployment Test**:
    1. Local Docker run of the prediction container.
    2. cURL request to verify JSON output format.
    3. End-to-end test via Streamlit UI.
