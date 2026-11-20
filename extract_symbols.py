import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

sf = 11
bw = 125000.0
Ts = (2**sf) / bw
Ns = int(fs * Ts) # 16384
k = bw / Ts

downchirp0_start = 513862
fc = 230010.99 + (3.0 * fs / Ns) # Adjust by 3 bins (183 Hz) so preamble bin is exactly 0!

t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))

# Payload start:
payload_start = downchirp0_start + int(2.25 * Ns)
print(f"Payload start sample: {payload_start} (t = {payload_start / fs:.4f}s)")

# How many symbols until the end of the recording?
total_syms = (len(iq_base) - payload_start) // Ns
print(f"Total possible payload symbols in capture: {total_syms}")

symbols = []
mags = []

for s_idx in range(total_syms):
    pos = payload_start + s_idx * Ns
    chunk = iq_base[pos:pos+Ns] * np.conj(upchirp)
    F = np.fft.fft(chunk)
    mag = np.abs(F)
    peak_bin = np.argmax(mag)
    
    # In LoRa, symbol bin is peak_bin / 8
    # Since FFT is length 16384 (8 samples per chip),
    # Let's check how sharp the peak is
    peak_val = mag[peak_bin]
    mean_val = np.mean(mag)
    snr_est = peak_val / mean_val
    
    # Symbol value in [0, 2047]
    # FFT bin maps to frequency 0..Fs:
    # 0..BW corresponds to bins 0..2047*8
    # If bin > 2048*8, it's outside BW (negative frequencies)
    sym_val = int(round(peak_bin / 8.0)) % 2048
    symbols.append(sym_val)
    mags.append((peak_val, snr_est, peak_bin % 8))
    
    print(f"Sym {s_idx:2d} (t={pos/fs:.3f}s): sym={sym_val:4d} (bin={peak_bin:5d}, pk={peak_val:6.0f}, snr={snr_est:4.1f}, rem={peak_bin%8})")
