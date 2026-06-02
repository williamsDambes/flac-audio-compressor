import numpy as np
from scipy.io import wavfile

# 1. Load the file
samplerate, data = wavfile.read('test.wav')
# Just use a small chunk of mono audio
audio = data[1000:1010, 0].astype(np.int32) 

# --- METHOD A: Polynomial Prediction (Order 1) ---
# Guess: "The next number is the same as the current one"
poly_residuals = np.zeros_like(audio)
poly_residuals[0] = audio[0]
for i in range(1, len(audio)):
    poly_residuals[i] = audio[i] - audio[i-1]

# --- METHOD B: Simple Linear Prediction ---
# Guess: "The next number follows the trend of the last two"
# Math: Guess = 2*Previous - Second_Previous
lpc_residuals = np.zeros_like(audio)
lpc_residuals[0] = audio[0]
lpc_residuals[1] = audio[1]
for i in range(2, len(audio)):
    prediction = (2 * audio[i-1]) - audio[i-2]
    lpc_residuals[i] = audio[i] - prediction

print("--- ORIGINAL SAMPLES ---")
print(audio)

print("\n--- POLY RESIDUALS (Mistakes) ---")
print(poly_residuals)

print("\n--- LINEAR (LPC) RESIDUALS (Mistakes) ---")
print(lpc_residuals)

# Let's check which method made the numbers smaller
print(f"\nSum of Poly: {np.sum(np.abs(poly_residuals))}")
print(f"Sum of LPC:  {np.sum(np.abs(lpc_residuals))}")