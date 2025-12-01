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

def dwt_denoising(data, wavelet="sym4", level=None):
    if data.ndim > 1:
        return np.array([dwt_denoising(row, wavelet, level) for row in data])
    
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
    
    denoised_data = pywt.waverec(new_coeffs, wavelet)
    
    if len(denoised_data) > len(data):
        denoised_data = denoised_data[:len(data)]
    return denoised_data

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
ns_denoised = dwt_denoising(ns_signal)

s_signal = SeizureData[0] if SeizureData.ndim > 1 else SeizureData
s_denoised = dwt_denoising(s_signal)

# Plot 4 images side by side
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
fig.suptitle("sym4 denoising", fontsize=16)

# Left side: Non-Seizure (Raw, Denoised)
plot_spectrogram(ns_signal, axes[0], "Raw Non-Seizure")
plot_spectrogram(ns_denoised, axes[1], "Denoised Non-Seizure (DWT)")

# Right side: Seizure (Raw, Denoised)
plot_spectrogram(s_signal, axes[2], "Raw Seizure")
plot_spectrogram(s_denoised, axes[3], "Denoised Seizure (DWT)")

plt.tight_layout()
plt.show()




