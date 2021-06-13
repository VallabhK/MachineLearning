#Logistic Regression using Numpy

import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
    """Init Function used to initialize variables

    Args:
        lr: learning rate
        n_iters: Number of iterations

    Returns:
        None

    """
        self.lr = lr
        self.n_iters = n_iters
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
        #Initialize the weights and biases
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot( X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
    
    def predict(self, X):
    """Function used to predict the values of the dependent variable

    Args:
        X: Independent variable

    Returns:
        Class of the predicted value

    """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_class
        

    def _sigmoid(self, x):
        return (1/(1+ np.exp(-x)))
