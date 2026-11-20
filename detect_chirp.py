import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Bandpass filter around 0 Hz: [-70 kHz, +70 kHz] for BW=125 kHz
# Or let's test directly on raw IQ or filtered IQ

def generate_chirp(sf, bw, fs, is_up=True):
    N = int(fs * (2**sf) / bw)
    t = np.arange(N) / fs
    T = (2**sf) / bw
    k = bw / T # chirp rate
    if is_up:
        phase = 2 * np.pi * (-bw/2 * t + 0.5 * k * t**2)
    else:
        phase = 2 * np.pi * (bw/2 * t - 0.5 * k * t**2)
    return np.exp(1j * phase)

# Let's test cross-correlation with upchirp for SF 7 through 12, BW=125 kHz
for bw in [125000.0, 250000.0]:
    for sf in [7, 8, 9, 10, 11, 12]:
        ref = generate_chirp(sf, bw, fs, is_up=True)
        # We can do matched filtering via FFT convolution
        # To avoid edge effects, correlate with iq
        corr = signal.fftconvolve(iq, np.conj(ref[::-1]), mode='valid')
        corr_mag = np.abs(corr)
        max_val = np.max(corr_mag)
        mean_val = np.mean(corr_mag)
        std_val = np.std(corr_mag)
        ratio = (max_val - mean_val) / std_val
        print(f"BW={bw/1e3:.0f}k, SF={sf:2d}: max/std = {ratio:.2f}, max_val = {max_val:.1f}")
