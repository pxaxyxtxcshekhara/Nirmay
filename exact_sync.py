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

t_sym = np.arange(Ns) / fs
# Upchirp: sweeps from -bw/2 to +bw/2
# Downchirp: sweeps from +bw/2 to -bw/2
# Let's write them explicitly:
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
downchirp = np.exp(1j * 2 * np.pi * (bw/2 * t_sym - 0.5 * k * t_sym**2))

# In tune_sf11:
# Downchirp 1 was around sample 513850
# Downchirp 2 is around sample 513850 + Ns = 530234
# Preamble upchirps were at pos = 513850 - (SFD_offset) * Ns
# In standard LoRa:
# Preamble: 8 upchirps (symbols -10 to -3)
# Sync word: 2 upchirps (symbols -2, -1)
# SFD: 2.25 downchirps (starts at symbol 0: downchirp 0, downchirp 1, quarter downchirp)
# So downchirp 0 is at sample 513850!
# Then preamble upchirp 7 (the one right before sync word) is at 513850 - 3 * Ns = 464698!

pos_down = 513850
pos_up = pos_down - 3 * Ns # a preamble chirp

# Let's dechirp downchirp with upchirp:
# (downchirp in signal * conj(downchirp template) = tone)
# Since downchirp template has phase (bw/2 t - 0.5 k t^2), conj(downchirp template) = exp(-j 2*pi*(bw/2 t - 0.5 k t^2))
chunk_down = iq[pos_down : pos_down + Ns] * np.conj(downchirp)
F_down = np.fft.fft(chunk_down)
bin_down = np.argmax(np.abs(F_down))
f_down = (bin_down if bin_down < Ns//2 else bin_down - Ns) * (fs / Ns)

chunk_up = iq[pos_up : pos_up + Ns] * np.conj(upchirp)
F_up = np.fft.fft(chunk_up)
bin_up = np.argmax(np.abs(F_up))
f_up = (bin_up if bin_up < Ns//2 else bin_up - Ns) * (fs / Ns)

print(f"f_down: {f_down:.2f} Hz (bin {bin_down})")
print(f"f_up:   {f_up:.2f} Hz (bin {bin_up})")

fc = (f_up + f_down) / 2.0
delta_f = (f_up - f_down) / 2.0
tau = delta_f / k
tau_samples = tau * fs

print(f"Carrier frequency fc: {fc:.2f} Hz")
print(f"Timing error tau: {tau*1e6:.2f} us ({tau_samples:.2f} samples)")
print(f"Exact start of downchirp 0: {pos_down + tau_samples:.2f}")
