import numpy as np

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Normalize / remove DC
iq = iq - np.mean(iq)

# For each candidate symbol length T_s:
# We compute prod(t) = iq[t] * conj(iq[t - T_s])
# Then integrate over a window of length T_s (moving sum)
# Then normalize by energy: |sum(prod)|^2 / (sum(|iq[t]|^2) * sum(|iq[t-T_s]|^2))
# This is the classic Schmidl-Cox / preamble autocorrelation metric!
# Value ranges from 0 to 1! For noise, it is ~ 1 / T_s.
# For a periodic preamble, it is close to 1 (or SNR / (SNR + 1))!

candidate_Ts = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

for Ts in candidate_Ts:
    # Product
    prod = iq[Ts:] * np.conj(iq[:-Ts])
    power = np.abs(iq)**2
    power1 = power[Ts:]
    power2 = power[:-Ts]
    
    # Moving sum of length Ts via cumulative sum
    # To be fast, use box filter (uniform_filter1d or cumsum)
    cumsum_prod = np.cumsum(np.concatenate(([0], prod)))
    win_prod = cumsum_prod[Ts:] - cumsum_prod[:-Ts]
    
    cumsum_p1 = np.cumsum(np.concatenate(([0], power1)))
    win_p1 = cumsum_p1[Ts:] - cumsum_p1[:-Ts]
    
    cumsum_p2 = np.cumsum(np.concatenate(([0], power2)))
    win_p2 = cumsum_p2[Ts:] - cumsum_p2[:-Ts]
    
    denom = np.sqrt(win_p1 * win_p2) + 1e-12
    metric = np.abs(win_prod) / denom
    
    max_idx = np.argmax(metric)
    max_val = metric[max_idx]
    
    # Also find all peaks above 0.15 (excluding the loud parts if needed)
    # Let's see:
    print(f"Ts={Ts:5d}: max metric = {max_val:.4f} at t = {max_idx / fs:.4f}s")
    
    # Check if there are other peaks outside t in [0.1, 0.25] and [1.85, 1.95]
    mask = np.ones(len(metric), dtype=bool)
    mask[int(0.1*fs):int(0.25*fs)] = False
    mask[int(1.85*fs):int(1.95*fs)] = False
    max_outside = np.max(metric[mask])
    max_outside_idx = np.where(mask)[0][np.argmax(metric[mask])]
    print(f"         outside loud parts: max = {max_outside:.4f} at t = {max_outside_idx / fs:.4f}s")
