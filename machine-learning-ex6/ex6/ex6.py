import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from sklearn import svm
import plotData as pd
import visualizeBoundary as vb
import gaussianKernel as gk

plt.ion()
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and
# plot the data

print('Loading and Visualizing data ... ')

# Load from ex6data1:
data = scio.loadmat('ex6data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Training Linear SVM =====================
# The following code will train a linear SVM on the dataset and plot the
# decision boundary learned
#

print('Training Linear SVM')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)

c = 1000
clf = svm.SVC(c, kernel='linear', tol=1e-3)
clf.fit(X, y)

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, 0, 4.5, 1.5, 5)

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Implementing Gaussian Kernel =====================
# You will now implement the Gaussian kernel to use
# with the SVM. You should now complete the code in gaussianKernel.py
#

print('Evaluating the Gaussian Kernel')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gk.gaussian_kernel(x1, x2, sigma)

print('Gaussian kernel between x1 = [1, 2, 1], x2 = [0, 4, -1], sigma = {} : {:0.6f}\n'
      '(for sigma = 2, this value should be about 0.324652'.format(sigma, sim))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Visualizing Dataset 2 =====================
# The following code will load the next dataset into your environment and
# plot the data
#

print('Loading and Visualizing Data ...')

# Load from ex6data1:
data = scio.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Training SVM with RBF Kernel (Dataset 2) =====================
# After you have implemented the kernel, we can now use it to train the
# SVM classifier
#
print('Training SVM with RFB(Gaussian) Kernel (this may take 1 to 2 minutes) ...')

c = 1
sigma = 0.1


def gaussian_kernel(x_1, x_2):
    n1 = x_1.shape[0]
    n2 = x_2.shape[0]
    result = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            result[i, j] = gk.gaussian_kernel(x_1[i], x_2[j], sigma)

    return result

# clf = svm.SVC(c, kernel=gaussian_kernel)
clf = svm.SVC(c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)

print('Training complete!')

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, 0, 1, .4, 1.0)

input('Program paused. Press ENTER to continue')

# ===================== Part 6: Visualizing Dataset 3 =====================
# The following code will load the next dataset into your environment and
# plot the data
#

print('Loading and Visualizing Data ...')

# Load from ex6data3:
data = scio.loadmat('ex6data3.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Plot training data
pd.plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 7: Visualizing Dataset 3 =====================

clf = svm.SVC(c, kernel='rbf', gamma=np.power(sigma, -2))
clf.fit(X, y)

pd.plot_data(X, y)
vb.visualize_boundary(clf, X, -.5, .3, -.8, .6)

input('ex6 Finished. Press ENTER to exit')
