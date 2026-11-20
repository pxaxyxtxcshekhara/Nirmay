import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# In burst 1:
# We know SF=7, BW=125000.
# Let's find the exact symbol timing and carrier frequency.
# First, isolate burst 1:
t0 = int(0.100 * fs)
t1 = int(0.160 * fs)
sig = iq[t0:t1]

sf = 7
bw = 125000.0
Ts = (2**sf) / bw
Ns = int(fs * Ts) # 1024
k = bw / Ts # chirp slope

# In burst1_hires.png, the SFD (downchirps) was around t = 0.010 to 0.012 (relative to t0).
# Downchirps have slope -k.
# If we multiply the signal by an upchirp of slope +k, a downchirp becomes a pure tone!
# Let's search for the downchirps (SFD):
t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
downchirp = np.conj(upchirp)

# Search across time in sig for maximum correlation with upchirp (which detects downchirps!)
# Also over carrier frequency:
# In burst1_hires.png, fc was roughly -312.5 kHz.
# Let's test fc from -315 kHz to -310 kHz
best_val = 0
best_fc = 0
best_pos = 0

for fc in np.linspace(-313500, -311500, 201):
    shift = np.exp(-1j * 2 * np.pi * fc * np.arange(len(sig)) / fs)
    sig_shift = sig * shift
    # Check around t = 0.010 to 0.015 (sample 10000 to 15000)
    for pos in range(10000, 15000, 16):
        chunk = sig_shift[pos:pos+Ns] * upchirp
        pk = np.abs(np.sum(chunk))
        if pk > best_val:
            best_val = pk
            best_fc = fc
            best_pos = pos

print(f"SFD search: best_val={best_val:.1f}, best_fc={best_fc:.2f} Hz, best_pos={best_pos}")

# Now refine around best_pos and best_fc
fine_fcs = np.linspace(best_fc - 200, best_fc + 200, 81)
best_val2 = 0
best_fc2 = 0
best_pos2 = 0
for fc in fine_fcs:
    shift = np.exp(-1j * 2 * np.pi * fc * np.arange(len(sig)) / fs)
    sig_shift = sig * shift
    for pos in range(best_pos - 32, best_pos + 32):
        chunk = sig_shift[pos:pos+Ns] * upchirp
        pk = np.abs(np.sum(chunk))
        if pk > best_val2:
            best_val2 = pk
            best_fc2 = fc
            best_pos2 = pos

print(f"Refined SFD: best_val={best_val2:.1f}, fc={best_fc2:.2f} Hz, pos={best_pos2}")
