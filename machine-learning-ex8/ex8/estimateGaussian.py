import numpy as np


def estimate_gaussian(X):
    # Useful variables
    m, n = X.shape

    # You should return these values correctly
    mu = np.zeros(n)
    sigma2 = np.zeros(n)

    # ===================== Your Code Here =====================
    # Instructions: Compute the mean of the data and the variances
    #               In particular, mu[i] should contain the mean of
    #               the data for the i-th feature and sigma2[i]
    #               should contain variance of the i-th feature
    #


    # ==========================================================

    return mu, sigma2
