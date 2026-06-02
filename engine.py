import numpy as np

def apply_mid_side(data):
    if len(data.shape) == 1: return data.astype(np.int32), None
    L = data[:, 0].astype(np.int32)
    R = data[:, 1].astype(np.int32)
    mid = (L + R) // 2
    side = L - R
    return mid, side

def undo_mid_side(mid, side):
    if side is None: return mid, None
    right = mid - (side // 2)
    left = right + side
    return left, right

def get_residuals(block, order):
    res = np.zeros_like(block, dtype=np.int32)
    if order == 0: return block.copy()
    res[0:order] = block[0:order]
    if order == 1:
        res[1:] = block[1:] - block[:-1]
    elif order == 2:
        res[2:] = block[2:] - (2*block[1:-1] - block[:-2])
    elif order == 3:
        res[3:] = block[3:] - (3*block[2:-1] - 3*block[1:-2] + block[:-3])
    elif order == 4:
        res[4:] = block[4:] - (4*block[3:-1] - 6*block[2:-2] + 4*block[1:-3] - block[:-4])
    return res

def undo_residuals(res, order):
    block = np.zeros(len(res), dtype=np.int32)
    block[0:order] = res[0:order]
    for n in range(order, len(res)):
        if order == 1:   block[n] = res[n] + block[n-1]
        elif order == 2: block[n] = res[n] + (2*block[n-1] - block[n-2])
        elif order == 3: block[n] = res[n] + (3*block[n-1] - 3*block[n-2] + block[n-3])
        elif order == 4: block[n] = res[n] + (4*block[n-1] - 6*block[n-2] + 4*block[n-3] - block[n-4])
        else: return res # Order 0
    return block