import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

sf = 11
bw = 125000.0
# Tuned carrier frequency
fc = 230010.99 + (3.0 * fs / 16384)

# Shift to baseband 0 Hz
t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

# Lowpass filter to [-62.5 kHz, +62.5 kHz] and decimate by 8
# Using scipy.signal.resample_poly or decimate
# FIR lowpass filter with cutoff at 62.5 kHz
iq_125k = signal.decimate(iq_base, 8, ftype='fir')
fs_new = 125000.0
print(f"Decimated IQ length: {len(iq_125k)}, sample rate: {fs_new}")

# Symbol length in decimated domain:
# N_sym = 2^11 = 2048 samples!
N_sym = 2**sf

# Preamble / SFD location in decimated domain:
# downchirp0_start was 513862 in 1 Msps -> in 125k it is 513862 / 8 = 64232.75
# Let's precisely find downchirp0 in the decimated domain:
t_sym = np.arange(N_sym) / fs_new
k = bw**2 / (2**sf)
# Base downchirp: sweeps from +bw/2 to -bw/2 or starts at 0?
# In discrete time at Fs = BW:
# f(n) = n * BW / N_sym = n * k / Fs
# Phase of base upchirp: phi[n] = pi * n^2 / N_sym
# Base upchirp: exp(1j * pi * (np.arange(N_sym)**2) / N_sym)
# Base downchirp: exp(-1j * pi * (np.arange(N_sym)**2) / N_sym)
upchirp_dec = np.exp(1j * np.pi * (np.arange(N_sym)**2) / N_sym)
downchirp_dec = np.conj(upchirp_dec)

# Refine downchirp0 position
center_sfd = int(513862 / 8)
best_p = 0
best_pk = 0
for p in range(center_sfd - 20, center_sfd + 20):
    chunk = iq_125k[p:p+N_sym] * upchirp_dec # multiply downchirp by upchirp -> tone
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    if pk > best_pk:
        best_pk = pk
        best_p = p

print(f"Decimated SFD 0 pos: {best_p}, peak = {best_pk:.1f}")

# Check preamble symbols (should be bin 0):
for i in range(-10, 0):
    pos = best_p + i * N_sym
    chunk = iq_125k[pos:pos+N_sym] * downchirp_dec
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    b = np.argmax(F)
    print(f"Preamble {i:2d}: bin = {b:4d}, pk = {pk:7.1f}")

# Check SFD downchirps:
for i in range(0, 2):
    pos = best_p + i * N_sym
    chunk = iq_125k[pos:pos+N_sym] * upchirp_dec
    F = np.abs(np.fft.fft(chunk))
    pk = np.max(F)
    b = np.argmax(F)
    print(f"SFD {i}: bin = {b:4d}, pk = {pk:7.1f}")
