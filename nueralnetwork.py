import random
import normalize

import matplotlib as plt
import numpy as np
import pandas as pd
import dataloader

X_train = np.array([], dtype=np.float64)
X_val = np.array([], dtype=np.float64)
X_test = np.array([], dtype=np.float64)
Y_train = np.array([], dtype=np.int64)
Y_val = np.array([], dtype=np.int64)
Y_test = np.array([], dtype=np.int64)

X_train, Y_train, X_val, Y_val, X_test, Y_test = dataloader.preprocess()

