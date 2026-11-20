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
downchirp = np.exp(-1j * 2 * np.pi * (-bw/2 * t_sym + 0.5 * k * t_sym**2))
upchirp = np.conj(downchirp)

# Let's inspect symbols around start_t = 0.3830s
start_idx = int(0.3830 * fs)
print(f"Inspecting around start_idx = {start_idx}...")

for sym_i in range(-3, 15):
    idx = start_idx + sym_i * Ns
    if idx >= 0 and idx + Ns <= len(iq):
        chunk = iq[idx:idx+Ns]
        F_down = np.abs(np.fft.fft(chunk * downchirp))
        F_up = np.abs(np.fft.fft(chunk * upchirp))
        
        pk_down = np.max(F_down)
        bin_down = np.argmax(F_down)
        pk_up = np.max(F_up)
        bin_up = np.argmax(F_up)
        
        print(f"sym {sym_i:2d} (t={idx/fs:.4f}s): down pk={pk_down:7.1f} bin={bin_down:5d} | up pk={pk_up:7.1f} bin={bin_up:5d}")
