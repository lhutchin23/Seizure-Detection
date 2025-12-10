"""
We compare our method of a custom CWT layer vs other thresholding standards.
input: 1D array (input-signal), N is the length of the signal (178)
output:
denoised np array of magnitudes of coeffs with shape: (N,S,T)

standard normalized batches.
wavelet basis: morlet, db8
# of scales is 1 through 127.
cwt transform for the following thresholds:
None : No threshold
VisuShrink: sigma * sqrt(2log(N))
Sqtwolog: sqrt(2log(N))
Rigrsure: SURE risk minimization
Heuresure: Rigsure + Sqtwolog to minimize SNR and SURE estimate becomes noisy
where sigma is the estimated noise standard deviation. We use the standard practice of median absolute deviation:
median of magnitude of coeffs / 0.6745

Future steps: Using DB8 for wavelet basis
https://www.sciencedirect.com/science/article/pii/S1665642313715244
"""

import numpy as np
import pywt


def sigma_mad(coeffs):
    """
    noise estimation by median absolute deviation
    """
    sigma = np.median(np.abs(coeffs)) / 0.6745
    return sigma


def universal_threshold(coeffs, N):
    """
    VisuShrink threshold
    sigma * sqrt(2 ln(N))"""
    sigma = sigma_mad(coeffs)
    return sigma * np.sqrt(2 * np.log(N))


def sqtwolog(coeffs):
    """
    sqtwolog threshold
    sqrt(2ln(N))
    """
    return np.sqrt(2 * np.log(len(coeffs.flatten())))


def minimax(coeffs, N):
    sigma = sigma_mad(coeffs)
    if N > 32:
        return sigma * (0.3936 + 0.1829 * np.log2(N))
    return 0


def rigr_sure(coeffs, N):
    """
    rigr_sure thresholding, we choose the threshold which minimizes SURE. Our approach is the one found at pyyawt.denoising, except we estimate sigma using MAD rather than letting sigma = 1.
    """
    sigma = sigma_mad(coeffs)
    coeffs_norm = coeffs / sigma
    s = np.sort(coeffs_norm**2)
    risks = np.zeros(N)
    for i in range(N):
        risks[i] = (N - 2 * (i + 1) + np.sum(s[: i + 1]) + (N - i - 1) * s[i]) / N
    if np.all(risks == risks[0]):
        threshold = 0.0
    else:
        idx_min = np.argmin(risks)
        threshold = np.sqrt(s[idx_min])
    return sigma * threshold


def heuresure(coeffs, N):
    """
    this is an implementation of heuresure which takes the minimum between virushrink and rigrsure
    """
    thresh_sure = rigr_sure(coeffs, N)
    thresh_univ = universal_threshold(coeffs, N)
    sigma = sigma_mad(coeffs)
    eta = np.sum(coeffs**2) / N - sigma**2
    if eta < 0:
        return thresh_univ
    else:
        return min(thresh_univ, thresh_sure)


def apply_threshold(coeffs, threshold, mode="soft"):
    if mode == "soft":
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    return coeffs * (np.abs(coeffs) > threshold)


def cwt_denoise(
    signal,
    wavelet="db8",
    scales=None,
    method="none",
    mode="soft",
):
    """applies cwt transform with denoising"""
    if scales is None:
        scales = np.arange(1, 65)

    #z-score normalize
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > 0:
        signal = (signal - signal_mean) / signal_std
    else:
        signal = signal - signal_mean

    coeffs, freqs = pywt.cwt(signal, scales, wavelet)
    N = len(signal)

    if method == "none":
        return np.abs(coeffs)

    threshold_funcs = {
        "universal_threshold": lambda c: universal_threshold(c, N),
        "sqtwolog": lambda c: sqtwolog(c),
        "minimax": lambda c: minimax(c, N),
        "rigrsure": lambda c: rigr_sure(c, N),
        "heuresure": lambda c: heuresure(c, N),
    }

    if method not in threshold_funcs:
        raise ValueError(f"Unknown method: {method}")

    threshold_func = threshold_funcs[method]

    coeffs_denoised = np.zeros_like(coeffs)
    for i, scale_coeff in enumerate(coeffs):
        threshold = threshold_func(scale_coeff)
        coeffs_denoised[i] = apply_threshold(scale_coeff, threshold, mode)

    return np.abs(coeffs_denoised)


def batch_transform(
    signals,
    wavelet="morl",
    scales=None,
    method="none",
    mode="soft",
):
    if signals.ndim == 1:
        return cwt_denoise(signals, wavelet, scales, method, mode)

    return np.array(
        [cwt_denoise(sig, wavelet, scales, method, mode) for sig in signals]
    )
