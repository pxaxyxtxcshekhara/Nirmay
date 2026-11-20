import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Overall spectrum
fft_len = 4096
# Average spectrum
psd = np.zeros(fft_len)
n_slices = len(iq) // fft_len
for i in range(n_slices):
    chunk = iq[i*fft_len : (i+1)*fft_len] * np.blackman(fft_len)
    psd += np.abs(np.fft.fftshift(np.fft.fft(chunk)))**2
psd /= n_slices

freqs = np.fft.fftshift(np.fft.fftfreq(fft_len, 1/fs))
# Find top peaks in psd
psd_db = 10 * np.log10(psd + 1e-12)
print("PSD min/max/mean (dB):", np.min(psd_db), np.max(psd_db), np.mean(psd_db))

# Check for signals around DC or other frequencies
# 125 kHz LoRa at DC would span [-62.5 kHz, +62.5 kHz]
# Let's print PSD in sub-bands of 50 kHz across [-500k, 500k]
for f_start in range(-500, 500, 50):
    mask = (freqs >= f_start*1e3) & (freqs < (f_start+50)*1e3)
    print(f"[{f_start:+4d} kHz, {f_start+50:+4d} kHz]: max = {np.max(psd_db[mask]):.2f} dB, mean = {np.mean(psd_db[mask]):.2f} dB")
