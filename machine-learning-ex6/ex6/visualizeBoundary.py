import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets


def visualize_boundary(clf, X, x_min, x_max, y_min, y_max):
    h = .02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='r')
