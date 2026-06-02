import numpy as np
from scipy.io import wavfile
import os

def calculate_rice_bits(value, k):
    """
    Calculates the number of bits required to store a value using Rice Coding.
    Includes ZigZag encoding for signed integers.
    """
    # 1. ZigZag Encoding: Maps signed integers to unsigned integers
    # Positive: n -> 2n | Negative: n -> 2|n| - 1
    if value >= 0:
        unsigned_val = 2 * value
    else:
        unsigned_val = 2 * abs(value) - 1
    
    # 2. Rice coding bit count logic:
    # Quotient (unary): unsigned_val // 2^k
    # Stop bit: 1 bit (0)
    # Remainder (binary): k bits
    quotient = unsigned_val >> k
    return quotient + 1 + k

def run_flac_simulation(audio_data, block_size, original_size_bits):
    """
    Simulates the compression pipeline and returns the total bit count.
    """
    total_compressed_bits = 0
    
    # STEP 1: Channel Decorrelation (Mid-Side)
    # Reduces redundancy between Left and Right channels
    if len(audio_data.shape) > 1:
        left = audio_data[:, 0].astype(np.int32)
        right = audio_data[:, 1].astype(np.int32)
        mid = (left + right) // 2
        side = left - right
        channels = [mid, side]
    else:
        channels = [audio_data.astype(np.int32)]

    # STEP 2: Processing each channel by Blocks
    for channel in channels:
        for i in range(0, len(channel), block_size):
            block = channel[i : i + block_size]
            
            # STEP 3: Polynomial Prediction (Order 1)
            # We store the 'residuals' (errors) instead of raw samples
            residuals = np.zeros_like(block)
            residuals[0] = block[0]
            for n in range(1, len(block)):
                residuals[n] = block[n] - block[n-1]
            
            # STEP 4: Adaptive Rice Parameter (k) Optimization
            # Find the optimal k (0-12) that yields the minimum bits for this block
            best_bits_for_block = float('inf')
            for k_candidate in range(13):
                current_bits = sum(calculate_rice_bits(r, k_candidate) for r in residuals)
                if current_bits < best_bits_for_block:
                    best_bits_for_block = current_bits
            
            total_compressed_bits += best_bits_for_block
                
    return total_compressed_bits

if __name__ == "__main__":
    # Configuration
    WAV_FILE = 'test.wav'
    
    if os.path.exists(WAV_FILE):
        sample_rate, pcm_data = wavfile.read(WAV_FILE)
        raw_size_bits = os.path.getsize(WAV_FILE) * 8
        
        print(f"--- FLAC-like Compressor Simulation ---")
        print(f"Input file: {WAV_FILE}")
        print(f"Original Size: {raw_size_bits} bits\n")

        # Investigate the influence of Block Size on compression ratio
        test_blocks = [1024, 4096, 16384]
        for b_size in test_blocks:
            comp_bits = run_flac_simulation(pcm_data, b_size, raw_size_bits)
            reduction = ((raw_size_bits - comp_bits) / raw_size_bits) * 100
            print(f"Block Size: {b_size:5} | Result: {comp_bits:10} bits | Reduction: {reduction:.2f}%")
    else:
        print(f"Error: {WAV_FILE} not found in the directory.")