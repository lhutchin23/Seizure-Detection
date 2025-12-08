import numpy as np
import pywt
import matplotlib.pyplot as plt

'''
CWT using 4 different thresholding functions.

Control: No denoising
SQWT: Square-root of log thresholding, constant across all data.
RIGRSURE: Risk minimalizaiton using Stein's Unbiased Risk Estimate, varies by data
HEURESURE: Heuristic SURE-based thresholding, combination of bvoth sqwt and rigrsure
'''

def sqwtolog(detail_coeff):
    coeff_flat = detail_coeff.flatten()
    sigma = np.median(np.abs(coeff_flat)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(coeff_flat)))
    return threshold

def rigrsure(detail_coeff):
    coeff_flat = detail_coeff.flatten()
    s = np.sort(coeff_flat ** 2)
    n = len(coeff_flat)
    
    risks = np.zeros(n)
    for i in range(n):
        risks[i] = (n - 2 * (i + 1) + np.sum(s[:i+1]) + (n - i - 1) * s[i]) / n
    
    if np.all(risks == risks[0]):
        lambda_sure = 0.0
    else:
        idx_min = np.argmin(risks)
        lambda_sure = np.sqrt(s[idx_min])
    
    return lambda_sure

def heuresure(detail_coeff):
    coeff_flat = detail_coeff.flatten()
    n = len(coeff_flat)

    sigma = np.median(np.abs(coeff_flat)) / 0.6745
    thresh_universal = sigma * np.sqrt(2 * np.log(n))
    
    s = np.sort(coeff_flat ** 2)
    risks = np.zeros(n)
    for i in range(n):
        risks[i] = (n - 2 * (i + 1) + np.sum(s[:i+1]) + (n - i - 1) * s[i]) / n
    
    if np.all(risks == risks[0]):
        thresh_sure = 0.0
    else:
        idx_min = np.argmin(risks)
        thresh_sure = np.sqrt(s[idx_min])
    
    eta = np.sum(coeff_flat ** 2) / n - sigma ** 2
    if eta < 0:
        return thresh_universal
    else:
        return min(thresh_universal, thresh_sure)

def cwt_denoising(data, wavelet="morl", scales=None, threshold_func=sqwtolog, 
                  threshold_type='scale_dependent', mode='soft'):
    
    #control group, no denoising, raw scalogram
    
    if threshold_func == 'no denoise':
        if scales is None:
            scales = np.arange(1, 65)
        
        # Process batch directly
        batch_coeffs = []
        for signal in data:
            coeffs, _ = pywt.cwt(signal, scales, wavelet)
            batch_coeffs.append(np.abs(coeffs))
            
        return np.array(batch_coeffs)
    
    print("denoising for" + threshold_func.__name__ + "\n")

    #denoising function for other 3 thresholding methods
    def process_signal(signal):
   
        if scales is None:
            scales_used = np.arange(1, 65)
        else:
            scales_used = scales
        
        # Forward CWT
        coeffs, frequencies = pywt.cwt(signal, scales_used, wavelet)
        
        # Apply Thresholding
        if threshold_type == 'global':
            threshold = threshold_func(coeffs)
            coeffs_denoised = pywt.threshold(coeffs, threshold, mode=mode)
            
        elif threshold_type == 'scale_dependent':
            coeffs_denoised = np.zeros_like(coeffs)
            for i, scale_coeff in enumerate(coeffs):
                threshold = threshold_func(scale_coeff)
                coeffs_denoised[i] = pywt.threshold(scale_coeff, threshold, mode=mode)
                
        elif threshold_type == 'location_dependent':
            coeffs_denoised = np.zeros_like(coeffs)
            for t in range(coeffs.shape[1]):
                time_coeff = coeffs[:, t]
                threshold = threshold_func(time_coeff)
                coeffs_denoised[:, t] = pywt.threshold(time_coeff, threshold, mode=mode)
        else:
            raise ValueError("threshold_type must be 'global', 'scale_dependent', or 'location_dependent'")
        
      
        

        return np.abs(coeffs_denoised)
    
    batch_coeffs = []
    for signal in data:
        denoised_coeffs = process_signal(signal)   
        batch_coeffs.append(denoised_coeffs)
    
    return np.array(batch_coeffs)
    







