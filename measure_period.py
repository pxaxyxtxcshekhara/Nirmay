import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

fc = 230000.0
downchirp0_start = 513862

# Let's inspect the preamble: e.g. from downchirp0_start - 5*16384 to downchirp0_start
# Let's compute instantaneous frequency:
# f_inst(t) = angle(x[n] * conj(x[n-1])) * fs / (2*pi)
# But signal is below noise floor!
# How do we measure f_inst of a weak signal?
# By correlating with chirps of different BW and duration!

# Let's check:
# If a single chirp repeats, how often does it repeat?
# In check_symbols.py:
# Index -10, -9, -8, -7, -6, -5, -4, -3 were spaced by 16384 samples!
# Did it repeat every 16384 samples, or did it repeat every 1024, 2048, 4096, 8192 samples?
chunk = iq[downchirp0_start - 8*16384 : downchirp0_start - 2*16384]

for lag in [512, 1024, 2048, 4096, 8192, 16384]:
    # Autocorrelation at lag
    corr = np.mean(chunk[lag:] * np.conj(chunk[:-lag]))
    print(f"Lag {lag:5d} ({lag/fs*1e3:6.3f} ms): mag = {np.abs(corr):.2f}")
