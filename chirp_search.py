import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Let's inspect the center band [-100 kHz, 100 kHz]
# Filter to center band and check power or do STFT with high resolution
# In LoRa, BW is typically 125 kHz, 250 kHz, or 500 kHz.
# Common SF (spreading factors) are 7, 8, 9, 10, 11, 12.
# Let's check chirp duration:
# T_sym = 2^SF / BW
# For BW = 125 kHz:
# SF7:  2^7  / 125000 = 128  / 125000 = 1.024 ms
# SF8:  2^8  / 125000 = 256  / 125000 = 2.048 ms
# SF9:  2^9  / 125000 = 512  / 125000 = 4.096 ms
# SF10: 2^10 / 125000 = 1024 / 125000 = 8.192 ms
# SF11: 2^11 / 125000 = 2048 / 125000 = 16.384 ms
# SF12: 2^12 / 125000 = 4096 / 125000 = 32.768 ms

# Preamble in LoRa is typically 8 (or 6 to 12) identical upchirps, followed by 2 downchirps (sync / SFD).

# Let's test matched filtering with upchirps for various BW and SF, centered at various frequencies (e.g. 0 Hz, or anywhere).
print("Writing chirp detector...")
