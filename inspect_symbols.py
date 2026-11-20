import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

t0 = int(0.100 * fs)
t1 = int(0.160 * fs)
sig = iq[t0:t1]

sf = 7
bw = 125000.0
Ts = (2**sf) / bw
Ns = int(fs * Ts) # 1024
k = bw / Ts

t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
downchirp = np.conj(upchirp)

fc = -312060.0
sig = sig * np.exp(-1j * 2 * np.pi * fc * np.arange(len(sig)) / fs)

# Let's inspect symbol energy and peaks for symbols starting from pos = 10257 - 10*1024
# First downchirp is at pos, let's see if pos - 1024 is upchirp or downchirp
for offset in range(-12, 45):
    p = 10257 + offset * Ns
    if p >= 0 and p + Ns <= len(sig):
        chunk = sig[p:p+Ns]
        # Dechirp with downchirp (to find modulated upchirps)
        F_up = np.abs(np.fft.fft(chunk * downchirp))
        # Dechirp with upchirp (to find modulated downchirps)
        F_down = np.abs(np.fft.fft(chunk * upchirp))
        
        peak_up = np.max(F_up)
        bin_up = np.argmax(F_up)
        peak_down = np.max(F_down)
        bin_down = np.argmax(F_down)
        
        # In LoRa, the bin maps to symbol:
        # Since Fs = 1 Msps and BW = 125 kHz, there are 8 samples per chip!
        # The FFT has length 1024. The 128 LoRa symbol bins are spaced by 8 FFT bins!
        sym_up = bin_up / 8.0
        
        print(f"sym {offset:3d} (sample {p:5d}): up=(pk={peak_up:5.0f}, bin={bin_up:4d}, sym={sym_up:5.1f}) | down=(pk={peak_down:5.0f}, bin={bin_down:4d})")
