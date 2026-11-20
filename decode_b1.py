import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

sf = 7
bw = 125000.0
fc = -312060.0

# In Burst 1, sample rate 1 Msps, BW = 125 kHz.
# Decimate by 8 to get Fs = BW = 125 kHz!
t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

import scipy.signal as signal
iq_125k = signal.decimate(iq_base, 8, ftype='fir')
fs_new = 125000.0
N_sym = 2**sf # 128

# Burst 1 starts around t = 0.100s -> sample 0.100 * 125000 = 12500
n = np.arange(N_sym)
upchirp = np.exp(1j * 2 * np.pi * (-0.5 * n + 0.5 * (n**2) / N_sym))
downchirp = np.conj(upchirp)

# Find SFD in Burst 1
# In find_sfd.py, pos was 10257 in sig (which started at 0.100s, so total sample was 100000 + 10257 = 110257)
# In decimated domain: 110257 / 8 = 13782
center_sfd = int(110257 / 8)
best_p = 0
best_pk = 0
for p in range(center_sfd - 20, center_sfd + 20):
    chunk = iq_125k[p:p+N_sym] * upchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    if pk > best_pk:
        best_pk = pk
        best_p = p

print(f"Burst 1 SFD 0 pos: {best_p}, pk = {best_pk:.1f}")

# Check preamble:
chunk0 = iq_125k[best_p - 4*N_sym : best_p - 3*N_sym] * downchirp
b0 = np.argmax(np.abs(np.fft.fft(chunk0)))
print(f"Burst 1 preamble bin: {b0}")

# Extract payload symbols:
payload_start = best_p + int(2.25 * N_sym)

b1_symbols = []
for s_idx in range(50):
    pos = payload_start + s_idx * N_sym
    if pos + N_sym > len(iq_125k):
        break
    chunk = iq_125k[pos:pos+N_sym] * downchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    b = np.argmax(F)
    sym = (b - b0) % N_sym
    b1_symbols.append((sym, b, pk))
    if pk > 2000: # signal present
        print(f"B1 Sym {s_idx:2d}: sym={sym:3d} (bin={b:3d}, pk={pk:5.0f})")
    else:
        print(f"B1 Sym {s_idx:2d}: END OF SIGNAL (pk={pk:5.0f})")
        break
