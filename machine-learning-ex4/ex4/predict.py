import numpy as np
from sigmoid import *


def predict(theta1, theta2, x):
    m = x.shape[0]

    x = np.c_[np.ones(m), x]
    h1 = sigmoid(np.dot(x, theta1.T))
    h1 = np.c_[np.ones(h1.shape[0]), h1]
    h2 = sigmoid(np.dot(h1, theta2.T))
    p = np.argmax(h2, axis=1) + 1

    return p


