import struct
import numpy as np
from scipy.io import wavfile

class BitReader:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.byte = 0
        self.bits_left = 0

    def read_bit(self):
        if self.bits_left == 0:
            byte_data = self.file.read(1)
            if not byte_data:
                return None
            self.byte = byte_data[0]
            self.bits_left = 8
        self.bits_left -= 1
        return (self.byte >> self.bits_left) & 1

    def read_bits(self, n):
        val = 0
        for _ in range(n):
            bit = self.read_bit()
            if bit is None: return None
            val = (val << 1) | bit
        return val

    def read_rice(self, k):
        q = 0
        while True:
            bit = self.read_bit()
            if bit == 1: q += 1
            elif bit == 0: break
            else: return None
        r = self.read_bits(k)
        val = (q << k) | r
        # ZigZag Decode
        return (val >> 1) ^ -(val & 1)

# --- DECODING PROCESS ---
print("Decoding...")
br = BitReader('compressed.flite')
sig = br.file.read(5) # 'FLITE'
rate = struct.unpack('>I', br.file.read(4))[0]
total_samples = struct.unpack('>I', br.file.read(4))[0]

decoded_channels = []
# We have 2 channels (Mid and Side)
for c in range(2):
    channel_residuals = []
    for _ in range(total_samples):
        res = br.read_rice(k=4)
        channel_residuals.append(res)
    
    # Undo Prediction (Integration)
    reconstructed = np.zeros(total_samples, dtype=np.int32)
    reconstructed[0] = channel_residuals[0]
    for n in range(1, total_samples):
        reconstructed[n] = reconstructed[n-1] + channel_residuals[n]
    decoded_channels.append(reconstructed)

# Undo Mid-Side Decorrelation
mid = decoded_channels[0]
side = decoded_channels[1]
left = mid + (side // 2)
right = mid - (side // 2)

# Save as restored.wav
final_audio = np.vstack((left, right)).T.astype(np.int16)
wavfile.write('restored.wav', rate, final_audio)
print("Success! 'restored.wav' created.")