import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pywt


'''

Model architecture:

3 Convolutional layers with ReLU activations and MaxPooling

Flatten the kernal and then apply 2 Fully Connected layers with Dropout in between with softmax at the end
'''

class EEG_CNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
        
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
            nn.Linear(256 * 8 * 22, 512), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.net(x)







