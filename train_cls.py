import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from data import BrainStrokeDataset
from models import StrokeClassifier
from utils.metrics import calculate_metrics
from utils.common import seed_everything

# Initialize W&B with default config (can be overridden by Sweep)
# We use a standard argparse or just read from wandb.config directly
# For Sweeps, passing args via command line is standard, but wandb.init() picks them up.

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    epoch_loss = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # CEL expects LongTensor labels [B]
        images, labels = images.to(device), labels.to(device).long()
        
        optimizer.zero_grad()
        
        # Mixed Precision Training (AMP)
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Scale Loss to prevent underflow
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        # Step-wise Logging
        wandb.log({"train_batch_loss": loss.item()})
        
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            # CEL expects LongTensor labels [B]
            images, labels = images.to(device), labels.to(device).long()
            
            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Softmax for multi-class (2 classes: Normal, Stroke)
            probs = torch.softmax(logits, dim=1)[:, 1] # Take prob of class 1 (Stroke)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Metrics
    # Convert list of arrays to single numpy array
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['val_loss'] = val_loss / len(loader)
    
    return metrics

def main():
    seed_everything(42)
    
    # 1. Initialize W&B
    # project="brain-stroke-ct" should match what you want on dashboard
    run = wandb.init(project="brain-stroke-ct", job_type="train")
    config = wandb.config
    
    # Set defaults if not provided by Sweep
    # (These act as fallbacks if running script manually without sweep)
    if not hasattr(config, 'learning_rate'): config.learning_rate = 0.001
    if not hasattr(config, 'batch_size'): config.batch_size = 16
    if not hasattr(config, 'dropout_rate'): config.dropout_rate = 0.2
    if not hasattr(config, 'label_smoothing'): config.label_smoothing = 0.1
    if not hasattr(config, 'weight_decay'): config.weight_decay = 1e-4
    if not hasattr(config, 'epochs'): config.epochs = 5

    print(f"[INFO] Config: {dict(config)}")

    # 2. Data Loaders
    # Note: Default data path
    train_ds = BrainStrokeDataset('splits.csv', split='train')
    val_ds = BrainStrokeDataset('splits.csv', split='val')
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    
    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StrokeClassifier(dropout_rate=config.dropout_rate).to(device)
    
    # 4. Loss & Optimizer
    # CrossEntropyLoss supports label_smoothing natively in newer Torch.
    # We output 2 classes, so we use CrossEntropyLoss.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler('cuda') # Initialize Scaler for AMP
    
    # 5. Training Loop
    best_f2 = 0.0
    
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Logging
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
        }
        wandb.log(log_dict)
        
        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val F2: {val_metrics['f2_score']:.4f} | Val AUC: {val_metrics['roc_auc']:.4f}")
        
        # Save Best Model
        if val_metrics['f2_score'] > best_f2:
            best_f2 = val_metrics['f2_score']
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth") # Upload to Cloud
            
    print("[INFO] Training Complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training Failed: {e}")
        # Signal failure to W&B
        wandb.finish(exit_code=1)
        raise e
