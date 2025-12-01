# Seizure-Detection
Programming a CNN model to identify seizures from Bonn dataset, the data benchmark for seizure recognition. 

Attempting to replicate the 90%+ success rate achieved by: https://arxiv.org/html/2508.08602v1#S10

HKIS juniors: Landon Hutchinson and Yongzhen Cheng

From raw image ---------> Wavelet transformed with thresholds determined by deep networks

Denoised image -----------> Seizure detection(Yes seizure or no seizure), binary classification.

Model architecture: 

From raw image ----> Wavelet

              Firstly, we use the 5 most common wavelet denoising techniques, Rigrsure,  Sqtwolog, Heursure, Minimax. 
              And then we plug that into our CNN to find which gives the most accurate results. 

Denoised image -----> Seizure detection:
     
              With our denoised image, this will be our model architecture: 

              
