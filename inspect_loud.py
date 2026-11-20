import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Loud part 1: t in [0.12, 0.22], around -312.5 kHz
# Let's inspect its spectrogram
t_start = int(0.12 * fs)
t_end = int(0.24 * fs)
iq_part1 = iq[t_start:t_end]

# Frequency shift to baseband: let's see center frequency
f_shift = 312500.0 # shift by +312.5 kHz
t = np.arange(len(iq_part1)) / fs
iq_part1_shifted = iq_part1 * np.exp(1j * 2 * np.pi * f_shift * t)

# Spectrogram of loud part 1
f, tt, Zxx = signal.spectrogram(iq_part1_shifted, fs=fs, nperseg=256, noverlap=128, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)

print("Part 1 shape:", Zxx.shape)

# Let's check instantaneous frequency or phase
# Is part 1 LoRa chirps?
# In LoRa, we see diagonal lines in spectrogram!
# Let's save a zoomed spectrogram of part 1
import matplotlib.pyplot as plt
import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\PRADYOTH C\Desktop\NIRMAY\mpl_tmp'

plt.figure(figsize=(10, 5))
plt.imshow(10*np.log10(np.abs(Zxx)+1e-12), aspect='auto', origin='lower',
           extent=[tt[0], tt[-1], f[0], f[-1]], cmap='viridis')
plt.ylim([-100000, 100000])
plt.title("Loud Part 1 (shifted by +312.5 kHz)")
plt.savefig("part1_zoom.png")
print("Saved part1_zoom.png")
