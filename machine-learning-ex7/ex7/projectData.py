import numpy as np


def project_data(X, U, K):
    # You need to return the following variables correctly.
    Z = np.zeros((X.shape[0], K))

    # ===================== Your Code Here =====================
    # Instructions: Compute the projection of the data using only the top K
    #               eigenvectors in U (first K columns).
    #               For the i-th example X[i], the projection on to the k-th
    #               eigenvector is given as follows:
    #                   x = X(i, :)';
    #                   projection_k = x' * U(:, k);
    #                   (above is octave code)
    #


    # ==========================================================

    return Z
