import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import displayData as dd
import lrCostFunction as lCF
import oneVsAll as ova
import predictOneVsAll as pova

plt.ion()

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
num_labels = 10         # 10 labels, from 0 to 9
                        # Note that we have mapped "0" to label 10


# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scio.loadmat('ex3data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

dd.display_data(selected)

input('Program paused. Press ENTER to continue')

# ===================== Part 2-a: Vectorize Logistic Regression =====================
# In this part of the exercise, you will reuse your logistic regression
# code from the last exercise. Your task here is to make sure that your
# regularized logistic regression implementation is vectorized. After
# that, you will implement one-vs-all classification for the handwritten
# digit dataset
#

# Test case for lrCostFunction
print('Testing lrCostFunction()')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((3, 5)).T/10]
y_t = np.array([1, 0, 1, 0, 1])
lmda_t = 3
cost, grad = lCF.lr_cost_function(theta_t, X_t, y_t, lmda_t)

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})
print('Cost: {:0.7f}'.format(cost))
print('Expected cost: 2.534819')
print('Gradients:\n{}'.format(grad))
print('Expected gradients:\n[ 0.146561 -0.548558 0.724722 1.398003]')

input('Program paused. Press ENTER to continue')

# ===================== Part 2-b: One-vs-All Training =====================
print('Training One-vs-All Logistic Regression ...')

lmd = 0.1
all_theta = ova.one_vs_all(X, y, num_labels, lmd)

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Predict for One-Vs-All =====================

pred = pova.predict_one_vs_all(all_theta, X)

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

input('ex3 Finished. Press ENTER to exit')
