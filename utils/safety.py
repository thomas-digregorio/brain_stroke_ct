import torch
import torch.nn.functional as F
import numpy as np

def check_saturation(image_tensor, threshold=0.5):
    """
    OOD Check: Rejects images that are too saturated (likely natural images, not CT).
    CT scans are grayscale, so R=G=B (Saturation ~ 0).
    Args:
        image_tensor (torch.Tensor): [C, H, W] Normalized tensor.
        threshold (float): Saturation threshold.
    Returns:
        bool: True if image is Safe (Low Saturation), False if OOD.
    """
    # Denormalize first to get back to roughly original RGB ratios
    # (Simplified check: Just check variance across channels)
    
    # Actually, efficient check: S = max(RGB) - min(RGB) / max(RGB)
    # If standard deviation across channels is high, it's colorful.
    
    # Calculate std dev across channel dimension
    channel_std = torch.std(image_tensor, dim=0).mean()
    
    if channel_std > threshold:
        return False # Too colorful (OOD)
    return True

def mc_dropout_predict(model, input_tensor, n_samples=10):
    """
    Monte Carlo Dropout Inference.
    Runs the model multiple times with Dropout enabled to estimate uncertainty.
    Args:
        model (nn.Module): The model.
        input_tensor (torch.Tensor): [B, C, H, W]
        n_samples (int): Number of forward passes.
    Returns:
        mean_prob (np.array): Average prediction.
        uncertainty (np.array): Standard deviation of predictions.
    """
    model.train() # Enable Dropout
    probs_list = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu().numpy())
            
    probs_stack = np.stack(probs_list)
    mean_prob = probs_stack.mean(axis=0)
    uncertainty = probs_stack.std(axis=0) # High std = High uncertainty
    
    model.eval() # Reset to eval mode
    return mean_prob, uncertainty
