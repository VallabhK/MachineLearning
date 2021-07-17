#This file contains implementation of Principal Component Analysis
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        #Compute mean
        self.mean = np.mean(X, axis = 0)
        X = X- self.mean
        #Compute Covariance
        cov  = np.cov(X.T)
        #Eigen vectors and eigenvalues
        eigenvalues, eigenvectors = np. linalg.eig(cov)
        #Sort Eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        #Store first n in eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transfor(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)