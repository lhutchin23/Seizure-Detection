import os
import glob
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join('data', 'Bonn')
fs = 173.61  # sampling rate in Hz

def plot_file(path, out_dir='plots', n_samples=1000):
    os.makedirs(out_dir, exist_ok=True)
    signal = np.loadtxt(path)
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time[:n_samples], signal[:n_samples])
    plt.title(f'Bonn EEG - {os.path.basename(path)}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage (μV)')
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, os.path.basename(path).replace('.txt', '.png'))
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path} (loaded {len(signal)} samples, duration: {len(signal)/fs:.1f} seconds)")

if __name__ == '__main__':
    files = glob.glob(os.path.join(DATA_DIR, '*.txt'))
    if not files:
        print(f'No files found in {DATA_DIR}. Place the Bonn data there.')
    for f in files[:10]:  # plot up to first 10 files
        plot_file(f)