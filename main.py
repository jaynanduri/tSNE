# importing libraries
from typing import Tuple, Any, Union

import numpy as np
from sklearn.decomposition import PCA
import pylab

np.random.seed(42)


class TSNE:
    """
    This class represents implementation of a different set of dimensionality reduction techniques that have no
    geometric constraints. Main focus is on preserving the distribution of higher dimension in lower dimensions.
    """

    def __init__(self, ndims: int, pca_dims: int, use_momentum: bool, X: np.ndarray) -> None:
        """
        Initialising params of the model.
        :param ndims: final dims after reduction
        :param pca_dims: no of dimensions to project original data in PCA
        :param use_momentum: use momentum to perform SGD
        :param X: original dataset
        """
        self.ndims = ndims
        self.pca_dims = pca_dims
        self.use_momentum = use_momentum
        self.X = X

    def h_beta(self, D: np.ndarray, beta=1.0) -> Tuple[Union[float, Any], Union[float, Any]]:
        """
        Compute P-row and corresponding perplexity
        :param D: pairwise distance matrix row
        :param beta: variance
        :return:
        """
        P = np.exp(-D.copy() * beta)
        sumP = sum(P)
        H = np.log(sumP) + beta * np.sum(D * P) / sumP
        P = P / sumP
        return H, P

    def compute_pij(self, perplexity: int) -> np.ndarray:
        """
        Compute the similarities pij for each row i. This is a one time operation and remains constant
        :param perplexity No of neighbours around i
        :return: similarities matrix
        """
        PCA_X = PCA(n_components=self.pca_dims).fit_transform(self.X)

        (n, d) = PCA_X.shape
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        # Loop over all datapoints
        for i in range(n):

            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf
            betamax = np.inf
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
            (H, thisP) = self.h_beta(Di, beta[i])

            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > 1e-5 and tries < 50:

                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy()
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].copy()
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.

                # Recompute the values
                (H, thisP) = self.h_beta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        # Return final P-matrix
        print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
        return P

    def run(self, max_iter: int, epsilon: int, min_gain: float) -> np.ndarray:
        """
        Finding the optimal y projection in lower dimension using stochastic gradient with momentum.
        :param max_iter: total number iterations for SGD
        :param epsilon: learning rate
        :param min_gain: minimum gain at each iteration
        :return: return solution vector
        """
        (n, d) = X.shape
        initial_momentum = 0.5
        final_momentum = 0.8
        Y = np.random.randn(n, self.ndims)
        dY = np.zeros_like(Y)
        iY = np.zeros_like(Y)
        gains = np.ones_like(Y)

        # Compute P-values
        P = self.compute_pij(30)
        P = P + np.transpose(P)
        P = P / np.sum(P)
        P = P * 4.  # early exaggeration
        P = np.maximum(P, 1e-12)

        # Run iterations
        for iter in range(max_iter):

            # Compute pairwise affinities
            sum_Y = np.sum(np.square(Y), 1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Compute gradient
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (self.ndims, 1)).T * (Y[i, :] - Y), 0)

            # Perform the update
            if iter < 20:
                momentum = initial_momentum
            else:
                momentum = final_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                    (gains * 0.8) * ((dY > 0.) == (iY > 0.))
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - epsilon * (gains * dY)
            Y = Y + iY
            Y = Y - np.tile(np.mean(Y, 0), (n, 1))

            # Compute current value of cost function
            if (iter + 1) % 10 == 0:
                C = np.sum(P * np.log(P / Q))
                print("Epoch %d: error is %f" % (iter + 1, C))

            # Stop lying about P-values
            if iter == 100:
                P = P / 4.

        # Return solution
        return Y


if __name__ == "__main__":
    X = np.loadtxt("data/mnist2500_X.txt")
    labels = np.loadtxt("data/mnist2500_labels.txt")
    n = X.shape[0]
    X = X - np.tile(np.mean(X, 0), (n, 1))
    tsne = TSNE(2, 50, True, X)
    y = tsne.run(400, 500, 0.01)
    pylab.scatter(y[:, 0], y[:, 1], 20, labels)
    pylab.show()

