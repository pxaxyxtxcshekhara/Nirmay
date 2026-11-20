import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Part 2: around t=1.88 to 1.95, freq ~ 400 kHz
t_start = int(1.88 * fs)
t_end = int(1.95 * fs)
iq_part2 = iq[t_start:t_end]

f_shift = -400000.0 # shift by -400 kHz
t = np.arange(len(iq_part2)) / fs
iq_part2_shifted = iq_part2 * np.exp(1j * 2 * np.pi * f_shift * t)

f, tt, Zxx = signal.spectrogram(iq_part2_shifted, fs=fs, nperseg=256, noverlap=128, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)

plt.figure(figsize=(10, 5))
plt.imshow(10*np.log10(np.abs(Zxx)+1e-12), aspect='auto', origin='lower',
           extent=[tt[0], tt[-1], f[0], f[-1]], cmap='viridis')
plt.ylim([-100000, 100000])
plt.title("Loud Part 2 (shifted by -400 kHz)")
plt.savefig("part2_zoom.png")
print("Saved part2_zoom.png")
