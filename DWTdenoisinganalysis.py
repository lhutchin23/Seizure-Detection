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

    if level is None:
        level = pywt.dwt_max_level(data.shape[-1], pywt.Wavelet(wavelet).dec_len)

    def process_one(signal):
        # 1. Decompose
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # 2. Denoise (Thresholding Detail Coefficients)
        # We usually keep the approximation (coeffs[0]) and threshold details (coeffs[1:])
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        
        denoised_coeffs = [coeffs[0]] # Keep approximation
        for detail in coeffs[1:]:
            denoised_coeffs.append(pywt.threshold(detail, threshold, mode='soft'))
            
        # 3. Resize and Stack to make a 2D Image
        target_length = len(signal)
        image_rows = []
        
        for c in denoised_coeffs:
            # Resize to match original signal length
            c_resized = np.interp(
                np.linspace(0, 1, target_length),
                np.linspace(0, 1, len(c)),
                c
            )
            image_rows.append(c_resized)
            
        return np.abs(np.vstack(image_rows))

    if data.ndim == 1:
        return process_one(data)
    elif data.ndim == 2:
        return np.array([process_one(x) for x in data])



print ("fertig")

