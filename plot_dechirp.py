import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

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
fc = 230010.99 + (3.0 * fs / Ns) # tuned fc

t_all = np.arange(len(iq)) / fs
iq_base = iq * np.exp(-1j * 2 * np.pi * fc * t_all)

# Let's create a matrix of symbols starting from preamble to end of packet
start_sym = downchirp0_start - 10 * Ns
num_syms = 90

t_sym = np.arange(Ns) / fs
upchirp = np.exp(1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))

waterfall = []
for i in range(num_syms):
    pos = start_sym + i * Ns
    chunk = iq_base[pos:pos+Ns] * np.conj(upchirp)
    F = np.abs(np.fft.fft(chunk))
    # Downsample or keep 2048 bins (each symbol is 8 bins)
    # The 2048 bins corresponding to 0..bw:
    # bins 0 to 2048*8: let's pick the max in each 8-bin window or take 2048-point FFT of decimate
    # Let's take the FFT bins 0..16384:
    waterfall.append(F)

waterfall = np.array(waterfall) # shape (num_syms, 16384)

plt.figure(figsize=(14, 8))
plt.imshow(waterfall[:, :2048*8:8], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Symbol value (0..2047)')
plt.ylabel('Symbol index (from -10)')
plt.title('Dechirped LoRa Waterfall')
plt.savefig('dechirped_waterfall.png')
print("Saved dechirped_waterfall.png")
