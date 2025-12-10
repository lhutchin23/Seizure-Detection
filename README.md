# Seizure-Detection
Programming a CNN model to identify seizures from Bonn dataset, the data benchmark for seizure recognition. 

Inspired by the study: https://arxiv.org/html/2508.08602v1#S10

HKIS juniors: Landon Hutchinson and Yongzhen Cheng
271296@hkis.edu.hk

Ultimately we achieved a 98.87% accuracy using a custom CNN model which incorporates a learnable wavelet denoising layer using the morlet basis.


# Purpose
The purpose of this project is to compare the impacts of different levels of wavelet denoising on the accuracy of a standard CNN.
We compared:
RigrSure
VisuShrink
HeureSure
Sqtwolog
No thresholding
Additionally we also incorporated a learnable wavelet denoising layer with the morlet wavelet as our basis:
Learnable parameters
Denoising Threshold (we do soft denoising), Scales, Centre_freq, and Bandwith_freq

# Standard CNN Model
conv2D 1

Batch norm 1

Relu 1

Maxpool 1

conv 2D 2

Batch norm 2

Relu 2

Maxpool 2

Dropout 1

FCL 1

Relu 3

Dropout 2

Relu 4

2 unit softmax

 
