import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Slice burst 2
t0 = int(1.895 * fs)
t1 = int(1.915 * fs)
sig = iq[t0:t1] * np.exp(-1j * 2 * np.pi * 400000.0 * np.arange(t1 - t0) / fs)

f, t, Zxx = signal.spectrogram(sig, fs=fs, nperseg=128, noverlap=112, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)

plt.figure(figsize=(15, 6))
plt.imshow(10*np.log10(np.abs(Zxx)+1e-12), aspect='auto', origin='lower',
           extent=[t[0], t[-1], f[0]/1e3, f[-1]/1e3], cmap='viridis')
plt.ylim([-60, 60])
plt.xlabel("Time [s]")
plt.ylabel("Freq [kHz]")
plt.title("Burst 2 zoomed")
plt.savefig("burst2_hires.png")
print("Saved burst2_hires.png")
