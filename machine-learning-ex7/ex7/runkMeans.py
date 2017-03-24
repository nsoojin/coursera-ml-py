import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import findClosestCentroids as fc
import computeCentroids as cc


def run_kmeans(X, initial_centroids, max_iters, plot):
    if plot:
        plt.figure()

    # Initialize values
    (m, n) = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print('K-Means iteration {}/{}'.format((i + 1), max_iters))

        # For each example in X, assign it to the closest centroid
        idx = fc.find_closest_centroids(X, centroids)

        # Optionally plot progress
        if plot:
            plot_progress(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press ENTER to continue')

        # Given the memberships, compute new centroids
        centroids = cc.compute_centroids(X, idx, K)

    return centroids, idx


def plot_progress(X, centroids, previous, idx, K, i):
    plt.scatter(X[:, 0], X[:, 1], c=idx, s=15)

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=25)

    for j in range(centroids.shape[0]):
        draw_line(centroids[j], previous[j])

    plt.title('Iteration number {}'.format(i + 1))


def draw_line(p1, p2):
    plt.plot(np.array([p1[0], p2[0]]), np.array([p1[1], p2[1]]), c='black', linewidth=1)
