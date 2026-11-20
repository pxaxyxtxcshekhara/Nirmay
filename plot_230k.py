import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

fc = 230000.0 # +230 kHz
# Slice from t = 0.3s to 1.8s
t0 = int(0.3 * fs)
t1 = int(1.8 * fs)
sig = iq[t0:t1] * np.exp(-1j * 2 * np.pi * fc * np.arange(t1 - t0) / fs)

# Let's filter to [-100 kHz, +100 kHz] to see it cleanly
# STFT
f, t, Zxx = signal.spectrogram(sig, fs=fs, nperseg=1024, noverlap=512, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)

plt.figure(figsize=(16, 6))
plt.imshow(10*np.log10(np.abs(Zxx)+1e-12), aspect='auto', origin='lower',
           extent=[0.3 + t[0], 0.3 + t[-1], f[0]/1e3, f[-1]/1e3], cmap='viridis',
           vmin=-50, vmax=-15)
plt.ylim([-80, 80])
plt.xlabel("Time [s]")
plt.ylabel("Freq [kHz]")
plt.title("Signal at +230 kHz")
plt.colorbar(label="dB")
plt.savefig("signal_230k.png")
print("Saved signal_230k.png")
