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
    
    def process_one(signal):
        # 1. Forward CWT
        # coeffs shape: (Scales, Time)
        coeffs, _ = pywt.cwt(signal, scales, wavelet)
        
        # 2. Denoising (Soft Thresholding)
        # Estimate noise sigma
        sigma = np.median(np.abs(coeffs)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        # Apply Soft Thresholding to the coefficients
        coeffs_denoised = pywt.threshold(coeffs, threshold, mode='soft')
        
        # 3. Return the Magnitude of the Clean Coefficients (The Image)
        # We return the absolute value because CNNs need real numbers
        return np.abs(coeffs_denoised)

    # Main Logic: Handle 1D or 2D input
    if data.ndim == 1:
        return process_one(data)
    elif data.ndim == 2:
        
        return np.array([process_one(row) for row in data])
    else:
        raise ValueError("Input data must be 1D or 2D array.")



    
    
print ("fertig")














