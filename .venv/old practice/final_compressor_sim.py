import numpy as np
from scipy.io import wavfile
import os

filename = 'test.wav'
samplerate, data = wavfile.read(filename)
original_size_bits = os.path.getsize(filename) * 8
data = data.astype(np.int32)

def rice_length(val, k):
    if val >= 0: v = 2 * val
    else: v = 2 * abs(val) - 1
    return (v >> k) + 1 + k

def simulate_compression(block_size):
    total_bits = 0
    if len(data.shape) > 1:
        mid = (data[:, 0] + data[:, 1]) // 2
        side = data[:, 0] - data[:, 1]
        channels = [mid, side]
    else:
        channels = [data]

    for channel in channels:
        for i in range(0, len(channel), block_size):
            block = channel[i : i + block_size]
            
            # Predict
            residuals = np.zeros_like(block)
            residuals[0] = block[0]
            for n in range(1, len(block)):
                residuals[n] = block[n] - block[n-1]
            
            # --- AUTO-OPTIMIZE K ---
            # We try k from 0 to 12 and pick the best one for THIS block
            best_block_bits = float('inf')
            for k_test in range(13):
                test_bits = sum(rice_length(r, k_test) for r in residuals)
                if test_bits < best_block_bits:
                    best_block_bits = test_bits
            
            total_bits += best_block_bits
                
    return total_bits

print("--- FLAC COMPRESSOR FINAL REPORT (OPTIMIZED) ---")
for b_size in [1024, 4096, 16384]:
    compressed_bits = simulate_compression(b_size)
    # Corrected formula for Reduction: (Original - Compressed) / Original
    reduction = ((original_size_bits - compressed_bits) / original_size_bits) * 100
    print(f"Block Size: {b_size:5} | Compressed: {compressed_bits:10} bits | Reduction: {reduction:.2f}%")