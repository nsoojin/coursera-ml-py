import numpy as np
import linearRegCostFunction as lrcf
import scipy.optimize as opt


def train_linear_reg(x, y, lmd):
    initial_theta = np.ones(x.shape[1])

    def cost_func(t):
        return lrcf.linear_reg_cost_function(t, x, y, lmd)[0]

    def grad_func(t):
        return lrcf.linear_reg_cost_function(t, x, y, lmd)[1]

    theta, *unused = opt.fmin_cg(cost_func, initial_theta, grad_func, maxiter=200, disp=False,
                                     full_output=True)

    return theta
