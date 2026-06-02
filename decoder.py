import struct, numpy as np
from scipy.io import wavfile
from engine import undo_mid_side, undo_residuals

class BitReader:
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.byte, self.bits_left = 0, 0

    def read_bit(self):
        if self.bits_left == 0:
            b = self.file.read(1)
            if not b: return None
            self.byte, self.bits_left = b[0], 8
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
        while self.read_bit() == 1: q += 1
        r = self.read_bits(k)
        val = (q << k) | r
        return (val >> 1) ^ -(val & 1)

br = BitReader('compressed.flite')
if br.file.read(5) == b'FLITE':
    rate, total_samples, num_ch = struct.unpack('>III', br.file.read(12))
    decoded_channels = []
    for _ in range(num_ch):
        channel_data, samples_done = [], 0
        while samples_done < total_samples:
            order, k = br.read_bits(4), br.read_bits(4)
            chunk_size = min(4096, total_samples - samples_done)
            res = [br.read_rice(k) for _ in range(chunk_size)]
            channel_data.extend(undo_residuals(np.array(res), order))
            samples_done += chunk_size
        decoded_channels.append(np.array(channel_data))

    if num_ch == 2:
        L, R = undo_mid_side(decoded_channels[0], decoded_channels[1])
        out = np.vstack((L, R)).T
    else: out = decoded_channels[0]
    
    wavfile.write('restored.wav', rate, out.astype(np.int16))
    print("Restored successfully.")