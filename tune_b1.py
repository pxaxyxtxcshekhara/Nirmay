import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Burst 1: from t=0.095 to 0.175
t0 = int(0.095 * fs)
t1 = int(0.175 * fs)
iq_b1 = iq[t0:t1]

# We need the exact carrier frequency and symbol timing
# LoRa parameters: SF=7, BW=125 kHz
# Symbol duration Ts = 128 / 125000 = 1.024 ms = 1024 samples
sf = 7
bw = 125000.0
Ts = (2**sf) / bw
Ns = int(fs * Ts) # 1024

# Let's search across a grid of fc around -312.5 kHz (+- 20 kHz) and sample offset [0, Ns)
# To find preamble chirps
best_score = 0
best_fc = 0
best_offset = 0

t = np.arange(Ns) / fs
k = bw / Ts

# For a candidate fc:
# downchirp_base = exp(-j * 2*pi * (fc*t - bw/2*t + 0.5*k*t^2))
# But notice: if we shift iq_b1 by exp(-j * 2*pi * fc * t_total), then
# downchirp is at 0 Hz: exp(-j * 2*pi * (-bw/2*t + 0.5*k*t^2))

# Let's test a fine frequency grid around -312.5 kHz:
fcs = np.linspace(-320000, -305000, 301)
downchirp0 = np.exp(-1j * 2 * np.pi * (-bw/2 * t + 0.5 * k * t**2))

# In the preamble, 8 consecutive symbols should have the EXACT same frequency bin (symbol 0)!
# That means if we dechirp 8 consecutive blocks, the peak should be at the same bin and very strong!

scores = []
for fc in fcs[::5]: # coarse first
    shift = np.exp(-1j * 2 * np.pi * fc * np.arange(len(iq_b1)) / fs)
    sig = iq_b1 * shift
    # correlate with downchirp
    # Let's compute dechirped FFT for all offsets in steps of 16
    for off in range(0, Ns, 64):
        # check 4 consecutive symbols
        peaks = []
        for sym_i in range(5):
            idx = off + 10000 + sym_i * Ns # start around sample 10000
            if idx + Ns <= len(sig):
                chunk = sig[idx:idx+Ns] * downchirp0
                F = np.abs(np.fft.fft(chunk))
                peaks.append((np.max(F), np.argmax(F)))
        if len(peaks) == 5:
            # check if bins are identical
            bins = [p[1] for p in peaks]
            if len(set(bins)) == 1:
                sc = sum([p[0] for p in peaks])
                scores.append((sc, fc, off, bins[0]))

scores.sort(key=lambda x: x[0], reverse=True)
print(f"Top 5 candidates:")
for sc, fc, off, b in scores[:5]:
    print(f"Score: {sc:.1f}, fc: {fc:.1f} Hz, offset: {off}, bin: {b}")
