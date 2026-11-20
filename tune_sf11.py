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

print(f"Ns = {Ns}, Ts = {Ts*1e3:.3f} ms, k = {k}")

# In inspect_sf11:
# sym 8 and sym 9 were downchirps at bin 3739.
# The downchirps are at start_idx + 8*Ns = 383000 + 8*16384 = 514072.
# Let's precisely find the start of the SFD (downchirps)
# A downchirp dechirped with upchirp:
t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))

# Search for the exact sample index of downchirp 1 around 514072 +- 2000 samples
center_search = 383000 + 8 * Ns
best_pk = 0
best_pos = 0
best_freq = 0

# We can search pos in steps of 1 sample, and frequency offset
# First let's find the coarse peak by taking FFT
for pos in range(center_search - 2000, center_search + 2000, 32):
    chunk = iq[pos:pos+Ns] * upchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    if pk > best_pk:
        best_pk = pk
        best_pos = pos

print(f"Coarse downchirp pos: {best_pos}, pk = {best_pk}")

# Refine pos by +- 64 samples
best_pk2 = 0
best_pos2 = 0
for pos in range(best_pos - 64, best_pos + 64):
    chunk = iq[pos:pos+Ns] * upchirp
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    if pk > best_pk2:
        best_pk2 = pk
        best_pos2 = pos

print(f"Refined downchirp pos: {best_pos2}, pk = {best_pk2}")

# Let's find the exact frequency of that tone
chunk = iq[best_pos2:best_pos2+Ns] * upchirp
F = np.fft.fft(chunk)
peak_bin = np.argmax(np.abs(F))
print(f"Peak bin: {peak_bin} (FFT length {Ns})")

# Frequency corresponding to peak_bin
# Since bin = peak_bin:
freq_offset = (peak_bin if peak_bin < Ns//2 else peak_bin - Ns) * (fs / Ns)
print(f"Frequency offset of tone: {freq_offset} Hz")
