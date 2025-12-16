import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from data import BrainStrokeDataset
from models import StrokeClassifier
from utils.metrics import calculate_metrics
import argparse
import os

def evaluate(model_path, split='test', batch_size=16):
    """
    Evaluates the model on the specified split.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 1. Load Data
    # Assuming splits.csv is in the current directory
    ds = BrainStrokeDataset('splits.csv', split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"[INFO] Loaded {len(ds)} images for {split} split.")

    # 2. Load Model
    model = StrokeClassifier().to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded weights from {model_path}")
    else:
        print(f"[WARNING] Model file {model_path} not found. Using random weights (Just for testing flow).")
    
    model.eval()
    
    # 3. Inference
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("[INFO] Running Inference...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # EfficientNet usually expects float inputs. 
            # Our transform does ToTensor -> Normalize.
            
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of Stroke
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_preds.extend((probs.cpu().numpy() > 0.5).astype(int))
            
    # 4. Metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    # Standard Metrics
    metrics = calculate_metrics(all_labels, all_probs)
    print("\n" + "="*30)
    print("       EVALUATION RESULTS       ")
    print("="*30)
    print(f"AUC:         {metrics['roc_auc']:.4f}")
    print(f"F2 Score:    {metrics['f2_score']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print("-" * 30)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Stroke']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate (val/test)")
    args = parser.parse_args()
    
    evaluate(args.model, args.split)
