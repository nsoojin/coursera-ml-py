import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from skimage import io
from skimage import img_as_float

import runkMeans as km
import findClosestCentroids as fc
import computeCentroids as cc
import kMeansInitCentroids as kmic

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Find Closest Centroids =====================
# To help you implement K-means, we have divided the learning algorithm
# into two functions -- find_closest_centroids and compute_centroids. In this
# part, you should complete the code in the findClosestCentroids.py
#

print('Finding closest centroids.')

# Load an example dataset that we will be using
data = scio.loadmat('ex7data2.mat')
X = data['X']

# Select an initial set of centroids
k = 3  # Three centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = fc.find_closest_centroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print('{}'.format(idx[0:3]))
print('(the closest centroids should be 0, 2, 1 respectively)')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Compute Means =====================
# After implementing the closest centroids function, you should now
# complete the compute_centroids function.
#

print('Computing centroids means.')

# Compute means based on the closest centroids found in the previous part.
centroids = cc.compute_centroids(X, idx, k)

print('Centroids computed after initial finding of closest centroids: \n{}'.format(centroids))
print('the centroids should be')
print('[[ 2.428301 3.157924 ]')
print(' [ 5.813503 2.633656 ]')
print(' [ 7.119387 3.616684 ]]')

input('Program paused. Press ENTER to continue')

# ===================== Part 3: K-Means Clustering =====================
# After you have completed the two functions compute_centroids and
# find_closest_centroids, you will have all the necessary pieces to run the
# kMeans algorithm. In this part, you will run the K-Means algorithm on
# the example dataset we have provided.
#
print('Running K-Means Clustering on example dataset.')

# Load an example dataset
data = scio.loadmat('ex7data2.mat')
X = data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = km.run_kmeans(X, initial_centroids, max_iters, True)
print('K-Means Done.')

input('Program paused. Press ENTER to continue')

# ===================== Part 4: K-Means Clustering on Pixels =====================
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel onto its closest centroid.
#
#  You should now complete the code in kMeansInitCentroids.m
#
print('Running K-Means clustering on pixels from an image')

# Load an image of a bird
image = io.imread('bird_small.png')
image = img_as_float(image)

# Size of the image
img_shape = image.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.

X = image.reshape(img_shape[0] * img_shape[1], 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly.
# You should complete the code in kMeansInitCentroids.py before proceeding
initial_centroids = kmic.kmeans_init_centroids(X, K)

# Run K-Means
centroids, idx = km.run_kmeans(X, initial_centroids, max_iters, False)
print('K-Means Done.')

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Image Compression =====================
# In this part of the exercise, you will use the clusters of K-Means to
# compress an image. To do this, we first find the closest clusters for
# each example.
print('Applying K-Means to compress an image.')

# Find closest cluster members
idx = fc.find_closest_centroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by its index in idx) to the centroid value
X_recovered = centroids[idx]

# Reshape the recovered image into proper dimensions
X_recovered = np.reshape(X_recovered, (img_shape[0], img_shape[1], 3))

plt.subplot(2, 1, 1)
plt.imshow(image)
plt.title('Original')

plt.subplot(2, 1, 2)
plt.imshow(X_recovered)
plt.title('Compressed, with {} colors'.format(K))

input('ex7 Finished. Press ENTER to exit')
