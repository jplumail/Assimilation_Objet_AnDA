import numpy as np
from utils import broadcasted_prod


def compute_weights(distances, l):
    """Compute weights with a gaussian kernel
    Parameters:
        distances : (..., k) array
        l : (...) array
    Returns:
        weights : (..., k) array
    """
    kernels = np.exp(-(distances/l[...,np.newaxis])**2)
    weights = kernels / np.sum(kernels, axis=-1)[...,np.newaxis]
    return weights

def locally_constant_mean(x, neighbors, successors, weights):
    """locally constant mean operator
    Parameters:
        x : inital state, (..., p) array where p%5 == 0
        neighbors : nearest neighbors of x in the catalog, (..., k, p) array
        successors : neighbors' successors, (..., k, p) array
        weights : weights of the neighbors, (..., k) array
    Returns:
        the next state of x according to the locally constant mean operator, (..., p) array
    """
    return np.sum(successors * weights[...,np.newaxis], axis=-2)

def locally_incremental_mean(x, neighbors, successors, weights):
    """locally incremental mean operator
    Parameters:
        x : inital state, (..., p) array where p%5 == 0
        neighbors : nearest neighbors of x in the catalog, (..., k, p) array
        successors : neighbors' successors, (..., k, p) array
        weights : weights of the neighbors, (..., k) array
    Returns:
        the next state of x according to the locally incremental mean operator, (..., p) array
    """
    return x + np.sum((successors - neighbors) * weights[...,np.newaxis], axis=-2)

def locally_linear_mean(x, neighbors, successors, weights):
    """locally_linear_mean operator
    Parameters:
        x : inital state, (..., p) array where p%5 == 0
        neighbors : nearest neighbors of x in the catalog, (..., k, p) array
        successors : neighbors' successors, (..., k, p) array
        weights : weights of the neighbors, (..., k) array
    Returns:
        y : the next state of x according to the locally linear mean operator, (..., p) array
    """
    W = weights[...,np.newaxis] * np.eye(weights.shape[-1]) # shape (...,k,k)
    X = neighbors
    Y = successors
    X_mean = np.sum(X*weights[...,np.newaxis], axis=-2) # shape (...,p)
    Y_mean = np.sum(Y*weights[...,np.newaxis], axis=-2) # shape (...,p)
    X_centered = X - X_mean[...,np.newaxis,:] # shape (...,k,p)
    Y_centered = Y - Y_mean[...,np.newaxis,:] # shape (...,k,p)
    cov_X = broadcasted_prod(
        broadcasted_prod(np.swapaxes(X_centered, -1, -2), W),
        X_centered
    ) # shape (...,p,p)
    cov_X_inv = np.linalg.pinv(cov_X) # shape (...,p,p)
    cov_YX = broadcasted_prod(
        broadcasted_prod(np.swapaxes(Y_centered, -1, -2), W),
        X_centered
    ) # shape (...,p,p)
    y = broadcasted_prod(broadcasted_prod(cov_YX, cov_X_inv), x - X_mean) # shape (...,p,p)
    return y