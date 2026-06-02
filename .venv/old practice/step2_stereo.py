import numpy as np
from scipy.io import wavfile

# 1. Load the data
samplerate, data = wavfile.read('test.wav')

# 2. Split the stereo columns into Left and Right
# We use .astype(np.int32) so we don't run out of room during math
left_channel = data[:10, 0].astype(np.int32)
right_channel = data[:10, 1].astype(np.int32)

# 3. DECORRELATION (The Mid-Side Trick)
mid = (left_channel + right_channel) // 2
side = left_channel - right_channel

print("--- ORIGINAL STEREO (L / R) ---")
for i in range(10):
    print(f"L: {left_channel[i]:4} | R: {right_channel[i]:4}")

print("\n--- DECORRELATED (MID / SIDE) ---")
for i in range(10):
    print(f"Mid: {mid[i]:4} | Side: {side[i]:4}")

print("\nNotice how the 'Side' numbers are often smaller than the original numbers?")