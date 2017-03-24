import scipy.optimize as opt
import lrCostFunction as lCF
from sigmoid import *


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))
        # ===================== Your Code Here =====================
        # Instructions : You should complete the following code to train num_labels
        #                logistic regression classifiers with regularization
        #                parameter lambda
        #
        #
        # Hint: you can use y == c to obtain a vector of True(1)'s and False(0)'s that tell you
        #       whether the ground truth is true/false for this class
        #
        # Note: For this assignment, we recommend using opt.fmin_cg to optimize the cost
        #       function. It is okay to use a for-loop (for c in range(num_labels) to
        #       loop over the different classes
        #



        # ============================================================    
        print('Done')

    return all_theta
