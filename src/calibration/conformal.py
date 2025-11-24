"""
Conformal prediction for uncertainty quantification.
"""
import numpy as np
import math


def compute_conformal_threshold(val_probs, val_labels, alpha=0.10):
    """
    Compute conformal prediction threshold using split-conformal method.
    
    Args:
        val_probs: Validation probabilities (N x K)
        val_labels: Validation labels (N,)
        alpha: Miscoverage rate (1-alpha is target coverage)
    
    Returns:
        threshold: Probability threshold for prediction sets
        quantile: Score quantile used
    """
    # Compute conformity scores: 1 - p(true class)
    val_true_probs = val_probs[np.arange(len(val_labels)), val_labels]
    val_scores = 1.0 - val_true_probs
    
    # Compute quantile with finite-sample correction
    n_val = len(val_scores)
    quantile = np.quantile(
        np.append(val_scores, 1.0),
        math.ceil((n_val + 1) * (1 - alpha)) / (n_val + 1.0)
    )
    
    threshold = 1.0 - quantile
    return threshold, quantile


def compute_conformal_sets(probs, threshold):
    """
    Compute conformal prediction sets.
    
    Args:
        probs: Class probabilities (N x K)
        threshold: Probability threshold
    
    Returns:
        conformal_sets: Binary matrix indicating set membership (N x K)
        set_sizes: Size of each prediction set (N,)
    """
    conformal_sets = (probs >= threshold).astype(int)
    set_sizes = conformal_sets.sum(axis=1)
    return conformal_sets, set_sizes


def conformal_confidence(set_sizes, epsilon=1e-12):
    """
    Convert conformal set sizes to confidence scores.
    Smaller sets indicate higher confidence.
    
    Args:
        set_sizes: Prediction set sizes (N,)
        epsilon: Small value to prevent division by zero
    
    Returns:
        Confidence scores (N,)
    """
    return 1.0 / (set_sizes + epsilon)
