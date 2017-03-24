import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt
import displayData as dd
import nncostfunction as ncf
import sigmoidgradient as sg
import randInitializeWeights as rinit
import checkNNGradients as cng
import predict as pd

plt.ion()

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
hidden_layer_size = 25  # 25 hidden layers
num_labels = 10         # 10 labels, from 0 to 9
                        # Note that we have mapped "0" to label 10


# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scio.loadmat('ex4data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

dd.display_data(selected)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Loading Parameters =====================
# In this part of the exercise, we load some pre-initiated
# neural network parameters

print('Loading Saved Neural Network Parameters ...')

data = scio.loadmat('ex4weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']

nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

# ===================== Part 3: Compute Cost (Feedforward) =====================
# To the neural network, you should first start by implementing the
# feedforward part of the neural network that returns the cost only. You
# should complete the code in nncostfunction.py to return cost. After
# implementing the feedforward to compute the cost, you can verify that
# your implementation is correct by verifying that you get the same cost
# as us for the fixed debugging parameters.
#
# We suggest implementing the feedforward cost *without* regularization
# first so that it will be easier for you to debug. Later, in part 4, you
# will get to implement the regularized cost.
#

print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
lmd = 0

cost, grad = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.287629)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Implement Regularization =====================
# Once your cost function implementation is correct, you should now
# continue to implement the regularization with the cost.
#

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
lmd = 1

cost, grad = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.383770)'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 5: Sigmoid Gradient =====================
# Before you start implementing the neural network, you will first
# implement the gradient for the sigmoid function. You should complete the
# code in the sigmoidGradient.py file
#

print('Evaluating sigmoid gradient ...')

g = sg.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))

print('Sigmoid gradient evaluated at [-1  -0.5  0  0.5  1]:\n{}'.format(g))

input('Program paused. Press ENTER to continue')

# ===================== Part 6: Initializing Parameters =====================
# In this part of the exercise, you will be starting to implement a two
# layer neural network that classifies digits. You will start by
# implementing a function to initialize the weights of the neural network
# (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')

initial_theta1 = rinit.rand_initialization(input_layer_size, hidden_layer_size)
initial_theta2 = rinit.rand_initialization(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_theta1.flatten(), initial_theta2.flatten()])

# ===================== Part 7: Implement Backpropagation =====================
# Once your cost matches up with ours, you should proceed to implement the
# backpropagation algorithm for the neural network. You should add to the
# code you've written in nncostfunction.py to return the partial
# derivatives of the parameters.
#

print('Checking Backpropagation ... ')

# Check gradients by running check_nn_gradients()

lmd = 0
cng.check_nn_gradients(lmd)

input('Program paused. Press ENTER to continue')

# ===================== Part 8: Implement Regularization =====================
# Once your backpropagation implementation is correct, you should now
# continue to implement the regularization with the cost and gradient.
#

print('Checking Backpropagation (w/ Regularization) ...')

lmd = 3
cng.check_nn_gradients(lmd)

# Also output the cost_function debugging values
debug_cost, _ = ncf.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:0.6f}\n(for lambda = 3, this value should be about 0.576051)'.format(lmd, debug_cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 9: Training NN =====================
# You have now implemented all the code necessary to train a neural
# network. To train your neural network, we will now use 'opt.fmin_cg'.
#

print('Training Neural Network ... ')

lmd = 1


def cost_func(p):
    return ncf.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[0]


def grad_func(p):
    return ncf.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[1]

nn_params, *unused = opt.fmin_cg(cost_func, fprime=grad_func, x0=nn_params, maxiter=400, disp=True, full_output=True)

# Obtain theta1 and theta2 back from nn_params
theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

input('Program paused. Press ENTER to continue')

# ===================== Part 10: Visualize Weights =====================
# You can now 'visualize' what the neural network is learning by
# displaying the hidden units to see what features they are capturing in
# the data

print('Visualizing Neural Network...')

dd.display_data(theta1[:, 1:])

input('Program paused. Press ENTER to continue')

# ===================== Part 11: Implement Predict =====================
# After the training the neural network, we would like to use it to predict
# the labels. You will now implement the 'predict' function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

pred = pd.predict(theta1, theta2, X)

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

input('ex4 Finished. Press ENTER to exit')
