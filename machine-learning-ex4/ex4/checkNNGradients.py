import numpy as np
import debugInitializeWeights as diw
import nncostfunction as ncf
import computeNumericalGradient as cng


def check_nn_gradients(lmd):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generatesome 'random' test data
    theta1 = diw.debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = diw.debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to genete X
    X = diw.debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m + 1), num_labels)

    # Unroll parameters
    nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

    def cost_func(p):
        return ncf.nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

    cost, grad = cost_func(nn_params)
    numgrad = cng.compute_numerial_gradient(cost_func, nn_params)

    print(np.c_[grad, numgrad])

