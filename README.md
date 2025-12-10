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


Test run 1: resulting scalograms in results_12_09_25. 

Results: 

No Denoise:
f1 accuracy: 0.9800
Scalogram:
![No denoise Confusion Matrix](results_12_09_25/confusion_matrix_nodenoise.png)

SQWTOLOG: 
f1 accuracy: 0.9739
Scalogram:
![SQWTOLOG Confusion Matrix](results_12_09_25/confusion_matrix_sqwtolog.png)

RIGRSURE: 
f1 accuracy: 0.9791
Scalogram:
![RIGRSURE Confusion Matrix](results_12_09_25/confusion_matrix_rigrsure.png)

HEURSURE: 
f1 accuracy: 0.9661
Scalogram:
![HEURSURE Confusion Matrix](results_12_09_25/confusion_matrix_heuresure.png)

VISURHINK: 
f1 accuracy: 0.9739
Scalogram: 
![VISURHINK Confusion Matrix](results_12_09_25/confusion_matrix_visushrink.png)
