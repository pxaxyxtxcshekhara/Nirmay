import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# In part1_zoom.png, we shifted by +312.5 kHz.
# In part1_zoom.png, the chirps spanned from around -62.5 kHz to +62.5 kHz.
# That means center frequency is -312.5 kHz!
# Bandwidth is 125 kHz!
# Let's verify:
# Preamble chirps:
t_start = int(0.12 * fs)
t_end = int(0.24 * fs)
iq_p1 = iq[t_start:t_end] * np.exp(1j * 2 * np.pi * 312500.0 * np.arange(t_end - t_start) / fs)

# Let's test dechirping for various SF (7, 8, etc.)
# If SF=7, symbol period = 128 / 125000 = 1.024 ms = 1024 samples at 1 Msps.
# Let's do a spectrogram / dechirp with N = 1024 samples!
sf = 7
bw = 125000.0
N = int(fs * (2**sf) / bw)
print(f"N samples per symbol: {N}")

t_chirp = np.arange(N) / fs
k = bw / ((2**sf) / bw)
base_downchirp = np.exp(-1j * 2 * np.pi * (-bw/2 * t_chirp + 0.5 * k * t_chirp**2))

# Let's slide downchirp and find peak correlation
corr = []
for i in range(0, len(iq_p1) - N, 16):
    prod = iq_p1[i:i+N] * base_downchirp
    fft_prod = np.abs(np.fft.fft(prod))
    corr.append((i, np.max(fft_prod), np.argmax(fft_prod)))

corr = sorted(corr, key=lambda x: x[1], reverse=True)
print("Top peaks in part 1:")
for p in corr[:15]:
    print(f"sample {p[0]} (t={p[0]/fs:.4f}s): max_fft={p[1]:.1f}, bin={p[2]}")
