import struct, numpy as np
from scipy.io import wavfile
from engine import apply_mid_side, get_residuals

class BitWriter:
    def __init__(self, filename):
        self.file = open(filename, 'wb')
        self.buffer, self.bits = 0, 0

    def write_bits(self, val, n):
        for i in range(n - 1, -1, -1):
            bit = (val >> i) & 1
            self.buffer = (self.buffer << 1) | bit
            self.bits += 1
            if self.bits == 8:
                self.file.write(bytes([self.buffer]))
                self.buffer, self.bits = 0, 0

    def write_rice(self, n, k):
        val = (n << 1) ^ (n >> 31)
        q, r = val >> k, val & ((1 << k) - 1)
        for _ in range(q): self.write_bits(1, 1)
        self.write_bits(0, 1)
        self.write_bits(r, k)

    def close(self):
        if self.bits > 0: self.file.write(bytes([self.buffer << (8 - self.bits)]))
        self.file.close()

rate, data = wavfile.read('test.wav')
bw = BitWriter('compressed.flite')
BLOCK_SIZE = 4096

bw.file.write(b'FLITE')
num_ch = 1 if len(data.shape) == 1 else 2
bw.file.write(struct.pack('>III', rate, len(data), num_ch))

mid, side = apply_mid_side(data)
channels = [mid, side] if side is not None else [mid]

for ch in channels:
    for i in range(0, len(ch), BLOCK_SIZE):
        block = ch[i : i + BLOCK_SIZE]
        best_bits, best_cfg = float('inf'), (1, 4)
        for order in [0, 1, 2, 3, 4]:
            res_t = get_residuals(block, order)
            zigzag = (res_t.astype(np.int64) << 1) ^ (res_t.astype(np.int64) >> 31)
            for k in range(13):
                bits = np.sum(zigzag >> k) + len(block) * (1 + k)
                if bits < best_bits:
                    best_bits, best_cfg = bits, (order, k)
        
        bw.write_bits(best_cfg[0], 4)
        bw.write_bits(best_cfg[1], 4)
        res = get_residuals(block, best_cfg[0])
        for val in res: bw.write_rice(val, best_cfg[1])

bw.close()
print("Compressed successfully.")