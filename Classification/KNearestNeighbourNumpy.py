#KNN Implementation using numpy

import numpy as np
from collections import Counter

def EuclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k;

    def fit(self, X, y):
    """Function used to define the relation between dependent and independent variables
    Args:
        X: Independent variable
        y: Dependent variable
    Returns:
        None
    """
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
    """Function used to predict 
    Args:
        X: Independent variable
    Returns:
        None
    """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        #Computing the distances
        distances = [EuclideanDistance(x, x_train) for x_train in self.X_train]
        #Compute the nearest neighbours
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        #Majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
