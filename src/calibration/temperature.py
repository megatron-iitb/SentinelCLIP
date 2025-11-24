"""
Temperature scaling and calibration methods.
"""
import numpy as np
import torch
import torch.nn as nn
from src.utils.math_utils import softmax


class TemperatureScaler(nn.Module):
    """Temperature scaling module for probability calibration."""
    
    def __init__(self):
        super().__init__()
        # Use log-parameterization to ensure positivity
        self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def forward(self, logits):
        """Apply temperature scaling to logits."""
        # Ensure temperature is always positive and bounded
        temperature = torch.exp(self.log_temperature).clamp(min=0.01, max=100.0)
        return logits / temperature


def fit_temperature_optimizer(logits_val, labels_val, device="cuda", 
                              lr=0.01, max_epochs=500, patience=10):
    """
    Fit temperature using Adam optimizer.
    
    Args:
        logits_val: Validation logits (numpy array)
        labels_val: Validation labels (numpy array)
        device: Device for computation
        lr: Learning rate
        max_epochs: Maximum training epochs
        patience: Early stopping patience
    
    Returns:
        Optimized temperature value
    """
    logits_t = torch.tensor(logits_val, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels_val, dtype=torch.long, device=device)
    scaler = TemperatureScaler().to(device)
    
    optimizer = torch.optim.Adam([scaler.log_temperature], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        out = scaler(logits_t)
        loss = loss_fn(out, labels_t)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    with torch.no_grad():
        T = torch.exp(scaler.log_temperature).clamp(min=0.01, max=100.0).item()
    
    return T


def fit_temperature_grid(logits_val, labels_val, ts=None):
    """
    Grid-search for optimal temperature using NLL on validation set.
    This is robust and avoids optimizer boundary artifacts.
    
    Args:
        logits_val: Validation logits (numpy array)
        labels_val: Validation labels (numpy array)
        ts: Temperature values to search (None for default)
    
    Returns:
        Best temperature value
    """
    if ts is None:
        ts = np.logspace(-2, 2, 201)  # 0.01 to 100
    
    labels = np.array(labels_val, dtype=int)
    best_T = None
    best_loss = float('inf')
    
    for T in ts:
        probs = softmax(logits_val / float(T), axis=1)
        p_true = probs[np.arange(len(labels)), labels]
        loss = -np.log(np.clip(p_true, 1e-12, 1.0)).mean()
        
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    
    return best_T


def fit_temperature(logits_val, labels_val, device="cuda"):
    """
    Fit temperature with automatic fallback to grid search.
    
    Args:
        logits_val: Validation logits
        labels_val: Validation labels
        device: Computation device
    
    Returns:
        Optimal temperature value
    """
    print("Fitting temperature on val set...")
    T = fit_temperature_optimizer(logits_val, labels_val, device=device)
    
    # If we hit a clamp bound, fall back to grid search
    if T <= 0.010001 or T >= 99.999:
        print("Temperature hit clamp bounds during optimization; running grid-search fallback...")
        try:
            T_grid = fit_temperature_grid(logits_val, labels_val)
            if T_grid is not None:
                print(f"Grid-search found T = {T_grid:.4f}; using that value")
                T = T_grid
        except Exception as e:
            print(f"Grid search failed: {e}")
    
    print(f"Learned temperature T = {T:.4f}")
    
    # Sanity check
    if T <= 0 or T > 100:
        print(f"WARNING: Temperature {T} is outside reasonable range. Using T=1.0")
        T = 1.0
    
    return T


def apply_temperature(logits, T):
    """Apply temperature scaling to logits."""
    return softmax(logits / T, axis=1)
