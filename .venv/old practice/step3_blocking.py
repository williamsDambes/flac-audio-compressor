import numpy as np
from scipy.io import wavfile

# 1. Load the file
samplerate, data = wavfile.read('test.wav')

# Use only one channel (Left) to make the math clear
mono_audio = data[:, 0].astype(np.int32)

def calculate_bits_needed(audio_chunk):
    # This finds the biggest number in the block
    # and calculates how many bits are needed to store it.
    max_val = np.max(np.abs(audio_chunk))
    if max_val == 0: return 1
    return int(np.ceil(np.log2(max_val + 1))) + 1 # +1 for the plus/minus sign

def investigate_blocks(block_sizes):
    print(f"{'Block Size':<12} | {'Total Bits Needed':<20}")
    print("-" * 35)
    
    for size in block_sizes:
        total_bits = 0
        # Loop through the audio in steps of 'size'
        for i in range(0, len(mono_audio), size):
            block = mono_audio[i : i + size]
            bits_per_sample = calculate_bits_needed(block)
            total_bits += bits_per_sample * len(block)
        
        print(f"{size:<12} | {total_bits:<20}")

# Try different block sizes as required by your coursework
sizes_to_test = [128, 512, 1024, 4096, 16384, 65536]
investigate_blocks(sizes_to_test)