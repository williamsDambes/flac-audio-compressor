import struct
import numpy as np
from scipy.io import wavfile
from engine import apply_mid_side, get_residuals

class BitWriter:
    def __init__(self, filename):
        self.file = open(filename, 'wb')
        self.buffer = 0
        self.bits = 0

    def write_bits(self, val, n):
        for i in range(n - 1, -1, -1):
            bit = (val >> i) & 1
            self.buffer = (self.buffer << 1) | bit
            self.bits += 1
            if self.bits == 8:
                self.file.write(bytes([self.buffer]))
                self.buffer = 0
                self.bits = 0

    def write_rice(self, n, k):
        val = (n << 1) ^ (n >> 31)
        q, r = val >> k, val & ((1 << k) - 1)
        for _ in range(q): self.write_bits(1, 1)
        self.write_bits(0, 1)
        self.write_bits(r, k)

    def close(self):
        if self.bits > 0: self.write_bits(0, 8 - self.bits)
        self.file.close()

# --- MAIN EXECUTION ---
rate, data = wavfile.read('test.wav')
bw = BitWriter('compressed.flite')

# 1. Write Header: Signature + Rate + Total Samples (New!)
bw.file.write(b'FLITE') 
bw.file.write(struct.pack('>I', rate))
bw.file.write(struct.pack('>I', len(data))) # Save length so decoder knows when to stop

mid, side = apply_mid_side(data)
channels = [mid] if side is None else [mid, side]

print("Compressing...")
for ch in channels:
    for i in range(0, len(ch), 4096):
        block = ch[i:i+4096]
        residuals = get_residuals(block)
        for res in residuals:
            bw.write_rice(res, k=4)

bw.close()
print("Success! 'compressed.flite' created.")