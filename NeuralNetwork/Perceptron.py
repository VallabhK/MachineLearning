#A Perceptron is basic block of code which mimics the actual 
#neuron present in human brain
#Limitation: The perceptron can only be used in cases where 
#the classes are linearly separable

import numpy as np

class Perceptron:

    def __init__(self, learning_rate =0.01, n_iters = 1000):
    """Function used to initialize the variables
    Args:
        learning_rate: Variables used to define step size
        n_iters: Number of iterations
    Returns:
        None
    """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
    """Function used to define the relation between dependent and independent variables
    Args:
        X: Independent variable
        y: Dependent variable
    Returns:
        None
    """
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
    """Function used to predict 
    Args:
        X: Independent variable
    Returns:
        y: predicted value
    """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def _unit_step_function(self, X):
    """Function used to determine the final class for an input  
    Args:
        X: Independent variable
    Returns:
        Class of the independent variable
    """
        return np.where(X>=0, 1, 0)