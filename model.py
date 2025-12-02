import random
import normalize

import matplotlib as plt
import numpy as np
import pandas as pd
import dataloader
from scipy import signal

#80, 10, 10 train, val, test split

X_train = np.array([], dtype=np.float64)
X_val = np.array([], dtype=np.float64)
X_test = np.array([], dtype=np.float64)
Y_train = np.array([], dtype=np.int64)
Y_val = np.array([], dtype=np.int64)
Y_test = np.array([], dtype=np.int64)

X_train, Y_train, X_val, Y_val, X_test, Y_test = dataloader.preprocess()


firstconv = convlayer.forward()
secondconv = convlayer.forward(firstconv)
thirdconv = convlayer.forward(secondconv)
dropout1 = dropout.forward(thirdconv)
FCL1 = FCLayer.forward(dropout1)
dropout2 = dropout.forward(FCL1)
FCL2 = FCLayer.forward(dropout2)
dropout3 = dropout.forward(FCL2)











