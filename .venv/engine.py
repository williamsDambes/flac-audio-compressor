import numpy as np

def apply_mid_side(data):
    """Channel Decorrelation: Converts L/R to Mid/Side."""
    if len(data.shape) == 1: return data, None
    L, R = data[:, 0].astype(np.int32), data[:, 1].astype(np.int32)
    return (L + R) // 2, L - R

def get_residuals(block):
    """Prediction: Calculates the difference between samples."""
    res = np.zeros_like(block)
    res[0] = block[0]
    res[1:] = block[1:] - block[:-1]
    return res