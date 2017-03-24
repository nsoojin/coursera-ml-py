import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

import displayData as dd
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

data = scio.loadmat('ex3data1.mat')
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

data = scio.loadmat('ex3weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']

# ===================== Part 3: Implement Predict =====================
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

pred = pd.predict(theta1, theta2, X)

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

input('Program paused. Press ENTER to continue')

# To give you an idea of the network's output, you can also run
# thru the examples one at a time to see what it is predicting


def getch():
    import termios
    import sys, tty

    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    return _getch()

# Randomly permute examples
rp = np.random.permutation(range(m))
for i in range(m):
    print('Displaying Example image')
    example = X[rp[i]]
    example = example.reshape((1, example.size))
    dd.display_data(example)

    pred = pd.predict(theta1, theta2, example)
    print('Neural network prediction: {} (digit {})'.format(pred, np.mod(pred, 10)))

    s = input('Paused - press ENTER to continue, q + ENTER to exit: ')
    if s == 'q':
        break
