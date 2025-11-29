import numpy as np

def Normalize(lst):
    """
    takes in a list of data and returns Z-score normalization.
    """
    mean = sum(lst)/len(lst)
    variance = 0
    for x in lst:
        variance += (x - mean)**2
    variance /= len(lst)
    variance = variance**0.5
    normalized_lst = [(x-mean) / variance for x in lst]
    return normalized_lst
