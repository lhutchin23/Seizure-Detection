'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pywt
from scipy import signal

#basic CNN model, not much commentary needed here

class EEG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(1, 64, 3, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),               
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),             
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),              
            
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(45056, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)               \
            

            
      
      
        )
        
    def forward(self, x):
        return self.net(x)
        


'''