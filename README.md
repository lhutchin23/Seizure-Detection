# Seizure-Detection
Programming a CNN model to identify seizures from Bonn dataset, the data benchmark for seizure recognition. 

Attempting to replicate the 90%+ success rate achieved by: https://arxiv.org/html/2508.08602v1#S10

HKIS juniors: Landon Hutchinson and Yongzhen Cheng


Model architecture: 
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
