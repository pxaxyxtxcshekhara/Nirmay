import numpy as np
import scipy.signal as signal
import os

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"

raw = np.fromfile(data_path, dtype=np.int8)
print(f"Total raw bytes: {len(raw)}")
# ci8 format: real, imag, real, imag...
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
print(f"Total IQ samples: {len(iq)}")

# Power over time
power = np.abs(iq)**2
mean_power = np.mean(power)
max_power = np.max(power)
print(f"Mean power: {mean_power:.2f}, Max power: {max_power:.2f}")

# Let's inspect power in chunks of, say, 1000 samples (1 ms)
chunk_size = 1000
chunks = len(iq) // chunk_size
power_profile = np.mean(power[:chunks*chunk_size].reshape(chunks, chunk_size), axis=1)

print(f"Power profile length (ms): {len(power_profile)}")
# Find loud regions
loud_threshold = np.percentile(power_profile, 90)
print(f"90th percentile power: {loud_threshold:.2f}, median power: {np.median(power_profile):.2f}, min power: {np.min(power_profile):.2f}")

# Look at top power regions
top_indices = np.argsort(power_profile)[-20:]
print(f"Top 20 ms indices by power: {sorted(top_indices)}")
