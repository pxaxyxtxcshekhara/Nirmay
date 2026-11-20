import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# First, check DC: mean of iq
mean_iq = np.mean(iq)
print(f"Mean IQ (DC bias): {mean_iq.real:.4f} + {mean_iq.imag:.4f}j, mag = {np.abs(mean_iq):.4f}")

# Let's inspect narrow band around 0 Hz: [-50 kHz, +50 kHz]
# Also let's inspect the entire frequency band by looking at spectrogram with high frequency resolution and high time resolution
# Let's subtract DC bias
iq_no_dc = iq - mean_iq

# Let's do STFT on iq_no_dc around 0 Hz
f, t, Zxx = signal.spectrogram(iq_no_dc, fs=fs, nperseg=4096, noverlap=2048, return_onesided=False)
f = np.fft.fftshift(f)
Zxx = np.fft.fftshift(Zxx, axes=0)
Zxx_db = 10 * np.log10(np.abs(Zxx) + 1e-12)

# Check frequencies between -100 kHz and +100 kHz
mask = (f >= -100000) & (f <= 100000)
f_sub = f[mask]
Zxx_sub = Zxx_db[mask, :]

print(f"Sub-band Zxx_db: min={np.min(Zxx_sub):.2f}, max={np.max(Zxx_sub):.2f}, mean={np.mean(Zxx_sub):.2f}")

# Plot sub-band
plt.figure(figsize=(12, 6))
plt.imshow(Zxx_sub, aspect='auto', origin='lower', extent=[t[0], t[-1], f_sub[0]/1e3, f_sub[-1]/1e3], cmap='viridis')
plt.title("Spectrogram around 0 Hz (DC removed)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (kHz)")
plt.colorbar(label="dB")
plt.savefig("center_band_zoom.png")
print("Saved center_band_zoom.png")
