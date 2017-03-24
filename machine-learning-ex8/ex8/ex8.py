import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import estimateGaussian as eg
import multivariateGaussian as mvg
import visualizeFit as vf
import selectThreshold as st

plt.ion()
# np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Load Example Dataset =====================
# We start this exercise by using a small dataset that is easy to visualize.
#
# Our example case consists of two network server statistics across
# several machines: the latency and throughput of each machine.
# This exercise will help us find possibly faulty (or very fast) machines
#

print('Visualizing example dataset for outlier detection.')

#  The following command loads the dataset. You should now have the
#  variables X, Xval, yval in your environment.
data = scio.loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].flatten()

# Visualize the example dataset
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='b', marker='x', s=15, linewidth=1)
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Estimate the dataset statistics =====================
# For this exercise, we assume a Gaussian distribution for the dataset.
#
# We first estimate the parameters of our assumed Gaussian distribution,
# then compute the probabilities for each of the points and then visualize
# both the overall distribution and where each of the points falls in
# terms of that distribution
#
print('Visualizing Gaussian fit.')

# Estimate mu and sigma2
mu, sigma2 = eg.estimate_gaussian(X)

# Returns the density of the multivariate normal at each data point(row) of X
p = mvg.multivariate_gaussian(X, mu, sigma2)

# Visualize the fit
vf.visualize_fit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s')

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Find Outliers =====================
# Now you will find a good epsilon threshold using a cross-validation set
# probabilities given the estimated Gaussian distribution
#
pval = mvg.multivariate_gaussian(Xval, mu, sigma2)

epsilon, f1 = st.select_threshold(yval, pval)
print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('(you should see a value epsilon of about 8.99e-05 and F1 of about 0.875)')

# Find outliers in the training set and plot
outliers = np.where(p < epsilon)
plt.scatter(X[outliers, 0], X[outliers, 1], marker='o', facecolors='none', edgecolors='r')

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Multidimensional Outliers =====================
# We will now use the code from the previous part and apply it to a
# harder problem in which more features describe each datapoint and only
# some features indicate whether a point is an outlier.
#

# Loads the second dataset.
data = scio.loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval'].flatten()

# Apply the same steps to the larger dataset
mu, sigma2 = eg.estimate_gaussian(X)

# Training set
p = mvg.multivariate_gaussian(X, mu, sigma2)

# Cross Validation set
pval = mvg.multivariate_gaussian(Xval, mu, sigma2)

# Find the best threshold
epsilon, f1 = st.select_threshold(yval, pval)

print('Best epsilon found using cross-validation: {:0.4e}'.format(epsilon))
print('Best F1 on Cross Validation Set: {:0.6f}'.format(f1))
print('# Outliers found: {}'.format(np.sum(np.less(p, epsilon))))
print('(you should see a value epsilon of about 1.38e-18, F1 of about 0.615, and 117 outliers)')

input('ex8 Finished. Press ENTER to exit')
