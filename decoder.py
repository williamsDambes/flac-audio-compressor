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
        while True:
            bit = self.read_bit()
            if bit == 1: q += 1
            elif bit == 0: break
            else: return None
        r = self.read_bits(k)
        val = (q << k) | r
        return (val >> 1) ^ -(val & 1)

# MAIN
print("Restoring...")
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

    L = decoded_channels[0]
    if num_ch == 2:
        L, R = undo_mid_side(decoded_channels[0], decoded_channels[1])
        out = np.vstack((L, R)).T
    else: out = L
    
    # CRITICAL: Clip and convert to int16
    out = np.clip(out, -32768, 32767).astype(np.int16)
    wavfile.write('restored.wav', rate, out)
    
    # Verification
    _, original = wavfile.read('test.wav')
    if np.array_equal(original[:len(out)], out):
        print("Success! Reconstruction is BIT-PERFECT.")
    else:
        print("Restored, but differences found.")