import torch
import torch.nn as nn
import torch.nn.functional as F

"""
for the morlet wavelet we use:
bandwidth_freq is the bandwidth frequency parameter
centre_freq is the centre frequency parameter
"""


class LearnableWaveletTransform(nn.Module):
    """
    Learnable wavelet transform:
    Basis: Morlet
    We do soft denoising.
    Learnable params:
    Threshold, Scales, Centre_freq, bandwith_freq
    """

    def __init__(self, num_scales=64, signal_length=178):
        super().__init__()

        # uses log based parameters to ensure > 0
        initial_scales = torch.linspace(1.0, 64.0, num_scales)
        self.log_scales = nn.Parameter(torch.log(initial_scales))

        self.log_centre_freq = nn.Parameter(torch.log(torch.tensor(6.0)))
        self.log_bandwidth_freq = nn.Parameter(torch.log(torch.tensor(1.0)))

        # learnable denoise threshold - we do soft denoising
        self.log_threshold = nn.Parameter(torch.log(torch.tensor(0.1)))

        self.signal_length = signal_length
        self.register_buffer("t", torch.linspace(-4, 4, signal_length))

    # converts back from log scale
    @property
    def scales(self):
        return torch.exp(self.log_scales)

    @property
    def centre_freq(self):
        return torch.exp(self.log_centre_freq)

    @property
    def bandwidth_freq(self):
        return torch.exp(self.log_bandwidth_freq)

    @property
    def threshold(self):
        return torch.exp(self.log_threshold)

    def morlet_wavelet(self, scale, centre_freq, bandwidth_freq):
        """
        complex wavelet definition according to pywavelets:
        1/sqrt(bandwidth_freq * pi) * exp(-t^2 / bandwidth_freq) * exp(i * 2 * pi * centre_freq * t)
        where:
        - centre_freq is the center frequency
        - bandwidth_freq is the bandwidth parameter
        - scale controls the dilation
        pywavelets reference:
        https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#morlet-wavelet
        """
        t_scaled = self.t / scale

        norm = 1.0 / torch.sqrt(bandwidth_freq * torch.pi * scale)

        gaussian = torch.exp(-(t_scaled**2) / bandwidth_freq)

        wave_real = torch.cos(2 * torch.pi * centre_freq * t_scaled)
        wave_imag = torch.sin(2 * torch.pi * centre_freq * t_scaled)

        wavelet_real = norm * gaussian * wave_real
        wavelet_imag = norm * gaussian * wave_imag

        return wavelet_real, wavelet_imag

    def forward(self, x):
        batch_size = x.shape[0]
        cwt_real_list = []
        cwt_imag_list = []

        scales = self.scales
        centre_freq = self.centre_freq
        bandwidth_freq = self.bandwidth_freq

        for scale in scales:
            wavelet_real, wavelet_imag = self.morlet_wavelet(
                scale, centre_freq, bandwidth_freq
            )

            conv_real = F.conv1d(
                x.unsqueeze(1),
                wavelet_real.flip(0).view(1, 1, -1),
                padding=self.signal_length // 2,
            )[:, :, : self.signal_length]

            conv_imag = F.conv1d(
                x.unsqueeze(1),
                wavelet_imag.flip(0).view(1, 1, -1),
                padding=self.signal_length // 2,
            )[:, :, : self.signal_length]

            cwt_real_list.append(conv_real)
            cwt_imag_list.append(conv_imag)

        cwt_real = torch.cat(cwt_real_list, dim=1)
        cwt_imag = torch.cat(cwt_imag_list, dim=1)

        magnitude = torch.sqrt(cwt_real**2 + cwt_imag**2 + 1e-10)

        # Apply soft thresholding for denoising
        threshold = self.threshold
        sharpness = 10.0
        gate = torch.sigmoid(sharpness * (magnitude - threshold))
        magnitude = magnitude * gate

        return magnitude.unsqueeze(1)


class EEG_CNN_Learnable(nn.Module):
    def __init__(self, num_scales=64, signal_length=178, dropout_rate=0.3):
        super().__init__()

        self.wavelet_transform = LearnableWaveletTransform(num_scales, signal_length)

        conv_out_height = num_scales // 8
        conv_out_width = signal_length // 8
        flattened_size = 256 * conv_out_height * conv_out_width

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.wavelet_transform(x)
        x = self.cnn(x)
        return x

    def get_learned_params(self):
        return {
            # scales needs to be put into CPU to access from Numpy
            "scales": self.wavelet_transform.scales.data.cpu().numpy(),
            "centre_freq": self.wavelet_transform.centre_freq.item(),
            "bandwidth_freq": self.wavelet_transform.bandwidth_freq.item(),
            "threshold": self.wavelet_transform.threshold.item(),
        }
