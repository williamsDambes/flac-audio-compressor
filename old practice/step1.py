import numpy as np
from scipy.io import wavfile

# This function opens a .wav file
def start_here():
    try:
        sampling_rate, data = wavfile.read('test.wav')
        
        print("--- SUCCESS! ---")
        print(f"The song plays at {sampling_rate} samples per second.")
        print(f"The first 10 numbers (samples) in the file are:")
        print(data[:10]) 
        
    except FileNotFoundError:
        print("Error: You need to put a file named 'test.wav' in this folder!")

start_here()
