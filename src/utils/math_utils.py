"""
Utility functions for numerical operations and helpers.
"""
import numpy as np
import hashlib


def softmax(x, axis=-1):
    """Compute softmax values for array x."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sha256_bytes(b):
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def clip_and_log(x, epsilon=1e-12):
    """Safely clip values before taking log."""
    return np.log(np.clip(x, epsilon, 1.0))


def geometric_mean_stable(values, axis=1, epsilon=1e-8):
    """
    Compute geometric mean in log-space for numerical stability.
    
    Args:
        values: Array of values (should be in [0, 1])
        axis: Axis along which to compute mean
        epsilon: Small value to prevent log(0)
    
    Returns:
        Geometric mean of values
    """
    values_clipped = np.clip(values, epsilon, 1.0)
    return np.exp(np.mean(np.log(values_clipped), axis=axis))


def compute_entropy(probs, epsilon=1e-12):
    """
    Compute entropy of probability distribution.
    
    Args:
        probs: Probability distribution (sum to 1 along last axis)
        epsilon: Small value for numerical stability
    
    Returns:
        Entropy values
    """
    return -np.sum(probs * np.log(np.clip(probs, epsilon, 1.0)), axis=-1)
