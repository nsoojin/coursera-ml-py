import numpy as np
import computeNumericalGradient as cng
import cofiCostFunction as ccf

def check_cost_function(lmd):

    # Create small problem
    x_t = np.random.rand(4, 3)
    theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(x_t, theta_t.T)  # 4x5
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    x = np.random.randn(x_t.shape[0], x_t.shape[1])
    theta = np.random.randn(theta_t.shape[0], theta_t.shape[1])
    num_users = Y.shape[1]  #5
    num_movies = Y.shape[0]  #4
    num_features = theta_t.shape[1] #3

    def cost_func(p):
        return ccf.cofi_cost_function(p, Y, R, num_users, num_movies, num_features, lmd)

    numgrad = cng.compute_numerial_gradient(cost_func, np.concatenate((x.flatten(), theta.flatten())))

    cost, grad = ccf.cofi_cost_function(np.concatenate((x.flatten(), theta.flatten())), Y, R, num_users, num_movies, num_features, lmd)

    print(np.c_[numgrad, grad])
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient')

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If you backpropagation implementation is correct, then\n'
          'the relative difference will be small (less than 1e-9).\n'
          'Relative Difference: {:0.3e}'.format(diff))
