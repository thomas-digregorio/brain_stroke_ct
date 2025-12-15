import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss

def calculate_metrics(y_true, y_probs, threshold=0.5):
    """
    Computes key clinical metrics for binary classification.
    Args:
        y_true (np.array): Ground truth labels (0 or 1)
        y_probs (np.array): Predicted probabilities [0, 1]
        threshold (float): Decision threshold
    Returns:
        dict: Metrics dictionary
    """
    # Thresholding
    y_pred = (y_probs >= threshold).astype(int)
    
    # 1. Base Counts
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 2. Clinical Metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F2 Score (Prioritizes Recall 2x over Precision)
    beta = 2
    f2_score = ((1 + beta**2) * precision * sensitivity) / ((beta**2 * precision) + sensitivity + 1e-8)
    
    # 3. Global Metrics
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except:
        roc_auc = 0.5 # Fail-safe for single class batches
        
    pr_auc = average_precision_score(y_true, y_probs)
    
    # 4. Calibration
    brier = brier_score_loss(y_true, y_probs)
    
    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f2_score': f2_score,
        'brier_score': brier,
        'confusion': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
    }
