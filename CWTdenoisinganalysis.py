import os

import pywt
import matplotlib.pyplot as plt
import numpy as np

SeizureDataPath = "DATA/seizure.txt"
NonSeizureDataPath = "DATA/nonseizure.txt"
SeizureData = []
NonSeizureData = []
with open(SeizureDataPath, "r") as f:
    for line in f:
        SeizureData.append(list(map(int, line.strip().split(","))))
    f.close()
with open(NonSeizureDataPath, "r") as f:
    for line in f:
        NonSeizureData.append(list(map(int, line.strip().split(","))))
    f.close()
SeizureData = np.array(SeizureData)
NonSeizureData = np.array(NonSeizureData)

def cwt_denoising(data, wavelet="morl", scales=np.arange(1, 128)):
    if data.ndim > 1:
        return np.array([cwt_denoising(row, wavelet, scales) for row in data])
    
    coeffs, freqs = pywt.cwt(data, scales, wavelet)
    # Estimate noise sigma from the first scale (highest frequency) only
    sigma = np.median(np.abs(coeffs[0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    
    magnitude = np.abs(coeffs)
    with np.errstate(divide='ignore', invalid='ignore'):
        scale_factor = np.maximum(0, 1 - threshold / magnitude)
    coeffs_denoised = coeffs * scale_factor
    
    reconstructed = np.sum(np.real(coeffs_denoised) / np.sqrt(scales)[:, None], axis=0)
    
    if np.std(reconstructed) > 0:
        reconstructed *= (np.std(data) / np.std(reconstructed))
        
    return reconstructed

def plot_spectrogram(signal, ax, title, wavelet="morl"):
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    
    im = ax.imshow(np.abs(coeffs), aspect='auto', cmap='jet', 
               extent=[0, len(signal), scales[-1], scales[0]])
    plt.colorbar(im, ax=ax, label='Magnitude')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Scale')

# Prepare signals
ns_signal = NonSeizureData[0] if NonSeizureData.ndim > 1 else NonSeizureData
ns_denoised = cwt_denoising(ns_signal)

s_signal = SeizureData[0] if SeizureData.ndim > 1 else SeizureData
s_denoised = cwt_denoising(s_signal)

# Plot 4 images side by side
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle("Morlet denoising", fontsize=16)

# Left side: Non-Seizure (Raw, Denoised)
plot_spectrogram(ns_signal, axes[0], "Raw Non-Seizure")
plot_spectrogram(ns_denoised, axes[1], "Denoised Non-Seizure")

# Right side: Seizure (Raw, Denoised)
plot_spectrogram(s_signal, axes[2], "Raw Seizure")
plot_spectrogram(s_denoised, axes[3], "Denoised Seizure")

plt.tight_layout()
plt.show()

print ("finished")









