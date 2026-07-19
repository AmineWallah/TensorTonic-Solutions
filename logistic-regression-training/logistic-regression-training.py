import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def predict(X, w, b):
    return _sigmoid(X*w + b)
    
def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = _sigmoid(X @ w + b)
        err = p - y
        w -= lr * (X.T @ err) / n
        b -= lr * err.mean()

    return (w,b)
    
    