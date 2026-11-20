import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Compute power in 100-sample windows (100 us)
w = 100
p = np.mean(np.abs(iq[:(len(iq)//w)*w].reshape(-1, w))**2, axis=1)
threshold = np.mean(p) + 3 * np.std(p)

print(f"Mean power: {np.mean(p):.2f}, std: {np.std(p):.2f}, threshold: {threshold:.2f}")
loud_blocks = np.where(p > threshold)[0]

# Group consecutive blocks into bursts
bursts = []
if len(loud_blocks) > 0:
    start = loud_blocks[0]
    prev = loud_blocks[0]
    for b in loud_blocks[1:]:
        if b - prev > 50: # gap > 5 ms
            bursts.append((start * w, prev * w))
            start = b
        prev = b
    bursts.append((start * w, prev * w))

print(f"Found {len(bursts)} bursts:")
for i, (s, e) in enumerate(bursts):
    t_s = s / fs
    t_e = e / fs
    dur = (e - s) / fs
    print(f"Burst {i+1}: t = {t_s:.4f}s to {t_e:.4f}s (duration = {dur*1e3:.2f} ms)")
