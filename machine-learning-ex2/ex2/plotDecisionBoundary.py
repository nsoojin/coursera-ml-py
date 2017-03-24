import matplotlib.pyplot as plt
import numpy as np
from plotData import *
from mapFeature import *


def plot_decision_boundary(theta, X, y):
    plot_data(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need two points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])

        plt.plot(plot_x, plot_y)

        plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc=1)
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))

        # Evaluate z = theta*x over the grid
        for i in range(0, u.size):
            for j in range(0, v.size):
                z[i, j] = np.dot(map_feature(u[i], v[j]), theta)

        z = z.T

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        cs = plt.contour(u, v, z, levels=[0], colors='r', label='Decision Boundary')
        plt.legend([cs.collections[0]], ['Decision Boundary'])
