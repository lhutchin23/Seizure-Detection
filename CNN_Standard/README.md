

Model architecture: 
We used a dropout of 0.3

Conv2D 1

BatchNorm 1

ReLU 1

MaxPool 1



Conv2D 2

BatchNorm 2

ReLU 2

MaxPool 2



Conv2D 3

BatchNorm 3

ReLU 3

MaxPool 3



Flatten

Dropout 1

FCL 1

ReLU 4

Dropout 2

FCL 2

 



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


