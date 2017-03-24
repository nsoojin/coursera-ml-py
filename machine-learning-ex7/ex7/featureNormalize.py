import numpy as np


def feature_normalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0, ddof=1)
    X_norm = (X - mu) / sigma

    return X_norm, mu, sigma
