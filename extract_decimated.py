import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

sf = 11
bw = 125000.0
# Tuned carrier frequency
fc = 230010.99 + (3.0 * fs / 16384)

# Shift to baseband 0 Hz
t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

# Decimate by 8
iq_125k = signal.decimate(iq_base, 8, ftype='fir')
fs_new = 125000.0
N_sym = 2**sf

# Base downchirp that sweeps from -BW/2 to +BW/2:
# In discrete time: n from 0 to N_sym-1
# f[n] = -bw/2 + k * n / fs = -bw/2 + (bw / N_sym) * n
# phi[n] = 2 * pi * (-0.5 * n + 0.5 * n^2 / N_sym)
n = np.arange(N_sym)
upchirp = np.exp(1j * 2 * np.pi * (-0.5 * n + 0.5 * (n**2) / N_sym))
downchirp = np.conj(upchirp)

best_p = 64248
# Let's check preamble with this centered downchirp:
# For preamble (symbol 0): chunk * downchirp should give bin 0!
chunk0 = iq_125k[best_p - 4*N_sym : best_p - 3*N_sym] * downchirp
F0 = np.abs(np.fft.fft(chunk0))
b0 = np.argmax(F0)
print(f"Preamble bin with centered downchirp: {b0}")

# Payload start:
payload_start = best_p + int(2.25 * N_sym)
print(f"Payload start index: {payload_start}")

# Extract symbols
raw_symbols = []
for s_idx in range(72):
    pos = payload_start + s_idx * N_sym
    chunk = iq_125k[pos:pos+N_sym] * downchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    b = np.argmax(F)
    # The symbol value is relative to preamble bin b0:
    sym = (b - b0) % N_sym
    raw_symbols.append((sym, pk))
    print(f"Sym {s_idx:2d}: sym = {sym:4d} (bin={b:4d}, pk={pk:6.0f})")
