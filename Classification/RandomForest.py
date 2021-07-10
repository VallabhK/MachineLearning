#This file contains the implementation of Random Forest algorithm

#Import all the required packages
from collections import Counter
import numpy as np

#Import the decision tree implemented previously
from .decision_tree import DecisionTree

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
