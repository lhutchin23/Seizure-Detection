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

def plot_denoised_spectrogram(data, wavelet="sym4", title="Spectrogram"):
    if data.ndim > 1:
        signal = data[0]
    else:
        signal = data

    denoised_signal = dwt_denoising(signal, wavelet=wavelet)
    
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(denoised_signal, scales, "morl")
    
    plt.figure(figsize=(12, 6))
    plt.imshow(np.abs(coeffs), aspect='auto', cmap='jet', 
               extent=[0, len(signal), scales[-1], scales[0]])
    plt.colorbar(label='Magnitude')
    plt.title(f'Denoised Spectrogram (sym4 denoised) - {title}')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.show()

plot_denoised_spectrogram(SeizureData, title="Seizure Data")
plot_denoised_spectrogram(NonSeizureData, title="Non-Seizure Data")

DenoisedSeizureData = dwt_denoising(SeizureData)
DenoisedNonSeizureData = dwt_denoising(NonSeizureData)






