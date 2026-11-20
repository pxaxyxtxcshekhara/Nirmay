import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

sf = 7
bw = 125000.0
fc = -312060.0

t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

iq_125k = signal.decimate(iq_base, 8, ftype='fir')
N_sym = 128

n = np.arange(N_sym)
upchirp = np.exp(1j * 2 * np.pi * (-0.5 * n + 0.5 * (n**2) / N_sym))
downchirp = np.conj(upchirp)

best_p = 13780
for i in range(-10, 3):
    pos = best_p + i * N_sym
    chunk = iq_125k[pos:pos+N_sym] * downchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    b = np.argmax(F)
    chunk_down = iq_125k[pos:pos+N_sym] * upchirp
    F_down = np.abs(np.fft.fft(chunk_down))
    pk_down = np.max(F_down)
    b_down = np.argmax(F_down)
    print(f"Index {i:2d}: up_pk={pk:5.0f} (bin={b:3d}) | down_pk={pk_down:5.0f} (bin={b_down:3d})")
