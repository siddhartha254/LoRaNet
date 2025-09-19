#!/usr/bin/env python3
import os
import subprocess
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 1) PICK BEST DEVICE
# ────────────────────────────────────────────────────────────────────────────────
def get_best_gpu():
    out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits"
    ]).decode().splitlines()
    free = [int(x) for x in out]
    return max(range(len(free)), key=lambda i: free[i])

if torch.cuda.is_available():
    best_gpu = get_best_gpu()
    device = torch.device(f"cuda:{best_gpu}")
    print(f"→ Using GPU {best_gpu}")
else:
    device = torch.device("cpu")
    print("→ No CUDA available, using CPU")

# ────────────────────────────────────────────────────────────────────────────────
# 2) PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────
Fs = 1_000_000               # 1 MHz sampling
minT, maxT = 0.05, 1.0       # signal duration range (s)
train_n, val_n = 4000, 1000  # #samples per class for train & val
test_bins = [-30, -20, -10, 0, 10]
test_per_bin = 500           # per class per bin

# ────────────────────────────────────────────────────────────────────────────────
# 3) DIRECTORY SETUP
# ────────────────────────────────────────────────────────────────────────────────
def makedirs(path):
    os.makedirs(path, exist_ok=True)

for split in ['train','val']:
    for cls in ['LoRa','Non-LoRa']:
        makedirs(f"data/{split}/{cls}")
for snr in test_bins:
    for cls in ['LoRa','Non-LoRa']:
        makedirs(f"data/test/{snr}dB/{cls}")

# ────────────────────────────────────────────────────────────────────────────────
# 4) SIGNAL GENERATORS (NumPy)
# ────────────────────────────────────────────────────────────────────────────────
def generate_lora(T):
    SF = np.random.randint(7, 13)
    BW = np.random.choice([125e3, 250e3])
    T_sym = (2**SF) / BW
    n_sym = max(1, int(T / T_sym))
    t_sym = np.arange(0, T_sym, 1/Fs)
    base = np.exp(1j * np.pi * (BW/T_sym) * t_sym**2)
    sig = np.zeros(int(T*Fs), dtype=complex)
    ptr = 0
    for _ in range(n_sym):
        symbol = np.random.randint(0, 2**SF)
        phase = np.exp(1j * 2*np.pi * symbol * (t_sym/T_sym) * (BW/(2**SF)))
        chunk = base * phase
        n = len(chunk)
        if ptr + n <= sig.size:
            sig[ptr:ptr+n] = chunk
        else:
            sig[ptr:] = chunk[:sig.size-ptr]
        ptr += n
    return np.real(sig)

def generate_nonlora(T):
    f0 = np.random.uniform(50e3, 200e3)
    t = np.arange(0, T, 1/Fs)
    return np.sin(2 * np.pi * f0 * t)

# ────────────────────────────────────────────────────────────────────────────────
# 5) GPU NOISE & SPECTROGRAM
# ────────────────────────────────────────────────────────────────────────────────
def add_noise(x, snr_db):
    p_signal = torch.mean(x**2)
    noise_p = p_signal / (10**(snr_db/10))
    noise = torch.sqrt(noise_p) * torch.randn_like(x)
    return x + noise

def save_spec(x, path):
    # x: 1D torch.Tensor already on 'device'
    n_fft = 256
    hop = n_fft - 200
    # <-- removed invalid device= argument
    S = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        return_complex=True
    )
    Sxx = (S.abs()**2).cpu().numpy()

    plt.figure(figsize=(2,2), dpi=150)
    plt.pcolormesh(
        np.linspace(0, x.size(0)/Fs, Sxx.shape[1]),
        np.linspace(0, Fs/2,    Sxx.shape[0]),
        10*np.log10(Sxx + 1e-10),
        shading='gouraud'
    )
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ────────────────────────────────────────────────────────────────────────────────
# 6) CORE LOOPS: train & val, then test
# ────────────────────────────────────────────────────────────────────────────────
for split, N in [('train', train_n), ('val', val_n)]:
    for cls, gen in [('LoRa', generate_lora), ('Non-LoRa', generate_nonlora)]:
        desc = f"{split:5s} {cls:8s}"
        out_dir = f"data/{split}/{cls}"
        for i in tqdm(range(1, N+1), desc=desc):
            T = np.random.uniform(minT, maxT)
            sig_np = gen(T)
            sig = torch.from_numpy(sig_np).float().to(device)
            snr = np.random.uniform(-30, 10)
            noisy = add_noise(sig, snr)
            fname = f"{cls.lower().replace('-','')}_snr_{snr:+05.1f}dB_{i:05d}.png"
            save_spec(noisy, os.path.join(out_dir, fname))

for snr in test_bins:
    for cls, gen in [('LoRa', generate_lora), ('Non-LoRa', generate_nonlora)]:
        desc = f"test {snr:+03d}dB {cls}"
        out_dir = f"data/test/{snr}dB/{cls}"
        for i in tqdm(range(1, test_per_bin+1), desc=desc):
            T = np.random.uniform(minT, maxT)
            sig_np = gen(T)
            sig = torch.from_numpy(sig_np).float().to(device)
            noisy = add_noise(sig, snr)
            fname = f"{cls.lower().replace('-','')}_snr_{snr:+04d}dB_{i:04d}.png"
            save_spec(noisy, os.path.join(out_dir, fname))

print("✔️  Dataset generation complete!")
