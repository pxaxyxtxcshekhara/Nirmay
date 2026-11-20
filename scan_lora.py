import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# For a given BW and SF:
# chirp rate mu = BW^2 / 2^SF
# downchirp(t) = exp(-j * pi * mu * t^2)
# When we multiply iq(t) by downchirp(t), any upchirp of rate mu becomes a tone!

for bw in [125000.0, 250000.0]:
    for sf in [7, 8, 9, 10, 11, 12]:
        T = (2**sf) / bw
        N = int(fs * T)
        mu = bw / T
        
        # Test: generate full downchirp repeating or continuous?
        # A continuous downchirp phase = -pi * mu * (t % T)^2
        t_sym = (np.arange(N) / fs)
        downchirp_sym = np.exp(-1j * np.pi * mu * (t_sym - T/2)**2)
        
        # We can test chunks of length N.
        # But symbol boundaries are unknown. If we step by N/4 or N/8:
        step = N // 4
        max_peaks = []
        for i in range(0, len(iq) - N, step):
            chunk = iq[i:i+N] * downchirp_sym
            # FFT of chunk
            F = np.abs(np.fft.fft(chunk))
            max_peaks.append((np.max(F), i))
        
        max_peaks.sort(key=lambda x: x[0], reverse=True)
        top5 = max_peaks[:5]
        mean_p = np.mean([p[0] for p in max_peaks])
        print(f"BW={bw/1e3:3.0f}k SF={sf:2d} (N={N:5d}): max={top5[0][0]:.1f}, mean={mean_p:.1f}, ratio={top5[0][0]/mean_p:.2f} at sample {top5[0][1]} (t={top5[0][1]/fs:.3f}s)")
