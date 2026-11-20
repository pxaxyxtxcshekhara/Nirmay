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

# Exact parameters
downchirp0_start = 513862
fc = 230010.99 # Hz

# Frequency shift the entire IQ by fc so the carrier is at 0 Hz!
t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

# Base upchirp and downchirp centered at 0 Hz
t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
downchirp = np.exp(1j * 2 * np.pi * (bw/2 * t_sym - 0.5 * k * t_sym**2))

# In baseband, dechirping an upchirp with downchirp template:
# conj(upchirp) = downchirp!
# So multiplying by conj(upchirp) gives a tone at the symbol frequency!
# In LoRa, a symbol s in {0, ..., 2^SF - 1} produces a chirp shifted in frequency by s * (BW / 2^SF).
# So the dechirped tone will be at frequency:
# f_tone = (s / 2^SF) * BW
# In the FFT of size Ns (sample rate Fs = 1 Msps):
# bin = f_tone / (Fs / Ns) = (s / 2^SF * BW) / (Fs / (2^SF * Fs / BW)) = s * (BW/Fs)^2 ? No!
# Let's check:
# Fs / Ns = Fs / (2^SF * Fs / BW) = BW / 2^SF!
# EXACT MATCH!
# One LoRa symbol bin = EXACTLY one FFT bin if FFT size is Ns / (Fs/BW) = 2^SF = 2048!
# Since Ns = 16384 and 2^SF = 2048:
# Fs / BW = 8 samples per chip.
# So in an FFT of length Ns (16384), the frequency bin of symbol s is:
# bin = s * 8!
# Symbol s is simply: round(bin / 8) % 2048!

print("Testing preamble symbols (should all be symbol 0):")
for i in range(-10, 0):
    pos = downchirp0_start + i * Ns
    chunk = iq_base[pos:pos+Ns] * np.conj(upchirp)
    F = np.fft.fft(chunk)
    mag = np.abs(F)
    peak_bin = np.argmax(mag)
    sym = (peak_bin / 8.0)
    print(f"Index {i:2d}: peak_mag = {mag[peak_bin]:7.1f}, bin = {peak_bin:5d}, sym = {sym:6.2f}")

print("\nTesting SFD downchirps (should be downchirps):")
for i in range(0, 2):
    pos = downchirp0_start + i * Ns
    chunk = iq_base[pos:pos+Ns] * np.conj(downchirp)
    F = np.fft.fft(chunk)
    mag = np.abs(F)
    peak_bin = np.argmax(mag)
    print(f"SFD {i}: peak_mag = {mag[peak_bin]:7.1f}, bin = {peak_bin:5d}")
