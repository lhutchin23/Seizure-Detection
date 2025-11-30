import random
import normalize

import matplotlib as plt
import numpy as np
import pandas as pd
def preprocess():
    SeizureDataPath = "DATA/seizure.txt"
    NonSeizureDataPath = "DATA/nonseizure.txt"
    SeizureData = []
    NonSeizureData = []
    #Stratified 80:10:10 train,dev,validate split
    with open(SeizureDataPath, "r") as f:
        for line in f:
            lst = list(map(int, line.strip().split(','))) 
            SeizureData.append(lst)
        f.close()
        
    print("Seizure Loaded \n")
    with open(NonSeizureDataPath, "r") as f:
        for line in f:
            lst = (list(map(int, line.strip().split(','))))
            NonSeizureData.append(lst)
        f.close()
    print("NonSeizureData Loaded \n")
    
    X = np.vstack([
        np.array(SeizureData, dtype=np.float64),
        np.array(NonSeizureData, dtype=np.float64)
    ])
    Y = np.hstack([
        np.ones(len(SeizureData), dtype=np.int64),
        np.zeros(len(NonSeizureData), dtype=np.int64)
    ])
    seizure_indices = np.array(range(0,2300))
    non_seizure_indices=np.array(range(2300,11500))
    
    print(str(seizure_indices)+" :seizure_indices \n")
    print(str(non_seizure_indices)+" :non_seizure_indices \n")
    np.random.shuffle(seizure_indices)
    np.random.shuffle(non_seizure_indices)
    
    blocks = []
    num_blocks = len(seizure_indices)
    for i in range(num_blocks):
        seizure_idx = seizure_indices[i]
        non_seizure_idxs = non_seizure_indices[i*4 : (i+1)*4]
        block = np.concatenate(([seizure_idx], non_seizure_idxs))
        blocks.append(block)
        
    random.shuffle(blocks)
    
    #80:10:10 split
    train_size = int(0.8 * num_blocks)
    val_size = int(0.1 * num_blocks)
    
    train_blocks = blocks[:train_size]
    val_blocks = blocks[train_size:train_size + val_size]
    test_blocks = blocks[train_size + val_size:]
    
    train_indices = np.concatenate(train_blocks).astype(int) 
    val_indices = np.concatenate(val_blocks).astype(int) 
    test_indices = np.concatenate(test_blocks).astype(int)
    
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    X_train, Y_train = X[train_indices], Y[train_indices]
    X_val, Y_val = X[val_indices], Y[val_indices]
    X_test, Y_test = X[test_indices], Y[test_indices]
    #Z-score normalization
    X_train = normalize.Normalize(X_train)
    X_val = normalize.Normalize(X_val)
    X_test = normalize.Normalize(X_test)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
