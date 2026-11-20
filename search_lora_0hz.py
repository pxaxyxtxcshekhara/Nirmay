import numpy as np
import scipy.signal as signal

data_path = r"C:\Users\PRADYOTH C\.gemini\antigravity\brain\bf7ca5c1-0efa-4637-aef8-1b7407d7f424\.user_uploaded\media_1795084227158_0741d785"
raw = np.fromfile(data_path, dtype=np.int8)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
fs = 1000000.0

# Direct dechirp on raw IQ

bw = 125000.0

print("Testing LoRa at 0 Hz (868.1 MHz) for SF 7 to 12...")

for sf in [7, 8, 9, 10, 11, 12]:
    Ts = (2**sf) / bw
    Ns = int(fs * Ts)
    k = bw / Ts
    
    t_sym = np.arange(Ns) / fs
    # Base downchirp:
    downchirp = np.exp(-1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
    
    # Let's test a fine search over time offset within [0, Ns)
    # We step by Ns // 8
    step = Ns // 8
    
    # We want to detect a preamble: 8 consecutive symbols at bin ~0 (or bin corresponding to carrier offset)
    # Let's slide through the whole recording
    n_blocks = (len(iq) - Ns) // step
    
    # To be very sensitive: for each offset, compute the FFT
    # In LoRa, the preamble chirps all produce a peak at the SAME bin for 8 consecutive symbols!
    # Let's compute the spectrogram of dechirped signal:
    # chunk_i = iq[i*Ns : (i+1)*Ns] * downchirp
    # Let's test 8 different timing phases (phase 0, 1, ..., 7)
    best_consec_score = 0
    best_phase = 0
    best_bin = 0
    best_start = 0
    
    for ph in range(8):
        offset = ph * step
        num_syms = (len(iq) - offset) // Ns
        if num_syms < 8:
            continue
        
        # Dechirp all symbols in this phase
        # Reshape into (num_syms, Ns)
        chunks = iq[offset : offset + num_syms * Ns].reshape(num_syms, Ns)
        dechirped = chunks * downchirp
        
        # FFT along axis 1
        spec = np.abs(np.fft.fft(dechirped, axis=1)) # (num_syms, Ns)
        
        # Look for 6+ consecutive symbols with a peak at the same bin
        # Peak bin for each symbol
        peaks = np.max(spec, axis=1)
        peak_bins = np.argmax(spec, axis=1)
        mean_noise = np.mean(spec)
        
        # Check running agreement of peak_bins
        # In a preamble, peak_bin is constant!
        for s_idx in range(num_syms - 6):
            b_cand = peak_bins[s_idx]
            # Check how many of the next 6 symbols have peak near b_cand (+- 2 bins)
            matches = [abs(peak_bins[s_idx + j] - b_cand) <= 2 or abs(abs(peak_bins[s_idx + j] - b_cand) - Ns) <= 2 for j in range(6)]
            if sum(matches) >= 5:
                # Calculate average peak to noise ratio
                p_ratio = np.mean(peaks[s_idx:s_idx+6]) / mean_noise
                if p_ratio > best_consec_score:
                    best_consec_score = p_ratio
                    best_phase = ph
                    best_bin = b_cand
                    best_start = offset + s_idx * Ns
    
    print(f"SF={sf:2d}: best consec score = {best_consec_score:.2f}, bin = {best_bin}, start_t = {best_start / fs:.4f}s")
