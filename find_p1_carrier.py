import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Loud part 1: t in [0.12, 0.22]
# Shift to baseband: let's precisely find the center frequency of Loud Part 1
t_start = int(0.12 * fs)
t_end = int(0.24 * fs)
iq_p1 = iq[t_start:t_end]

# Let's find exact frequency offset around -312.5 kHz
# A preamble upchirp: f(t) = fc - bw/2 + k*t
# If we multiply by downchirp, we get a constant frequency tone at fc!
sf = 7
bw = 125000.0
T = (2**sf) / bw
N = int(fs * T) # 1024
k = bw / T
t = np.arange(N) / fs
downchirp = np.exp(-1j * np.pi * k * t**2)

# Correlate across part 1 to find the preamble chirps
best_peak = 0
best_i = 0
for i in range(0, len(iq_p1) - N, 4):
    chunk = iq_p1[i:i+N] * downchirp
    # FFT
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    if pk > best_peak:
        best_peak = pk
        best_i = i

print(f"Best peak: {best_peak} at relative sample {best_i}")

# At best_i, let's see the peak frequency
chunk = iq_p1[best_i:best_i+N] * downchirp
F = np.abs(np.fft.fft(chunk))
peak_bin = np.argmax(F)
freq_bin = (peak_bin if peak_bin < N//2 else peak_bin - N) * (fs / N)
print(f"Peak bin: {peak_bin}, estimated carrier tone: {freq_bin} Hz")
