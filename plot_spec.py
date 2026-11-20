import os
os.environ['MPLCONFIGDIR'] = r'C:\Users\PRADYOTH C\Desktop\NIRMAY\mpl_tmp'
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)

fs = 1000000.0 # 1 Msps

# Spectrogram
# Let's do STFT with nperseg=1024 or 2048
f, t, Zxx = signal.spectrogram(iq, fs=fs, nperseg=1024, noverlap=512, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)
Zxx_db = 10 * np.log10(np.abs(Zxx) + 1e-12)

print(f"Freq range: {f[0]} to {f[-1]}")
print(f"Time range: {t[0]} to {t[-1]}")
print(f"Zxx_db min: {np.min(Zxx_db):.1f}, max: {np.max(Zxx_db):.1f}, mean: {np.mean(Zxx_db):.1f}")

# Plot and save
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f / 1e3, Zxx_db, shading='gouraud', cmap='viridis')
plt.ylabel('Frequency [kHz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram (868.1 MHz center)')
plt.colorbar(label='dB')
plt.savefig('spectrogram_full.png', dpi=150)
print("Saved spectrogram_full.png")
