import numpy as np
from distance import wasserstein


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
    cov_X = np.swapaxes(X_centered, -1, -2) @ W @ X_centered # shape (...,p,p)
    cov_X_inv = np.linalg.pinv(cov_X) # shape (...,p,p)
    cov_YX = np.swapaxes(Y_centered, -1, -2) @ W @ X_centered # shape (...,p,p)
    y = cov_YX @ cov_X_inv @ (x - X_mean)
    return


def forecast_step(catalogues, observations, operator, k, distance):
    """forecasting the next state of several observations
    Parameters:
        catalogues: (...,N,2,p) array, p%5==0
            If catalogues.ndim > 3, catalogues is a stack of catalogues.
            In this case, (...) must be broadcastable with observations (...). It becomes the shape of the output.
        observations: (...,p) array
        operator: an operator taken from forecasting.py
        k: number of analogs, int
            k < N
        distance: a distance.GaussianDistance object
    Returns:
        predictions: (...,p) array"""
    
    predecessors = catalogues[...,:,0,:] # shape (...,N,p)
    # Compute distances
    distances = distance(observations[...,np.newaxis,:], predecessors) # shape (...,N), broadcast between observations and predecessors
    broadcasted_shape = distances.shape[:-1]

    # find k nearest analogs and their successors
    indices_analogs = np.argpartition(distances, k, axis=-1)[...,:k] # indices of the k nearest analogs, shape (...,k)
    ind = [[True]*(catalogues.shape[i]) for i in range(catalogues.ndim-2)]
    ind = list(np.ix_(*ind))
    ind[-1] = indices_analogs
    analogs = catalogues[tuple(ind+[0])] # shape (...,k,p)
    successors = catalogues[tuple(ind+[1])] # successors of the analogs, shape (...,k,p)

    # Distances of analogs
    ind = [[True]*(distances.shape[i]) for i in range(distances.ndim)]
    ind = list(np.ix_(*ind))
    ind[-1] = indices_analogs
    distance_analogs = distances[tuple(ind)] # shape (...,k)

    # Compute weights and apply the operator
    weights = compute_weights(distance_analogs, np.median(distance_analogs, axis=-1))
    predictions = operator(observations, analogs, successors, weights)

    return predictions



def forecast(catalogues, observations, operator, T, k=50, distance=wasserstein):
    """forecasting the next P states
    Parameters:
        catalogues: (...,N,2,p) array, p%5==0
            If catalogues.ndim > 3, catalogues is a stack of catalogues.
            In this case, (...) must be broadcastable with observations (...). It becomes the shape of the output.
        observations: (...,p) array
        operator: an operator from operator.py
        T: number of predictions to do, int
        k: number of analogs to consider
        distance: distance.GaussianDistance object
    Returns:
        predictions: (..., p, T) array"""
    final_shape = np.broadcast_shapes(catalogues.shape[:-3], observations.shape[:-1])
    p = observations.shape[-1]
    predictions = np.empty(final_shape+tuple([p, T]))
    next_obs = observations
    for j in range(T):
        next_obs = forecast_step(catalogue, next_obs, operator, k, distance)
        predictions[...,j] = next_obs
    return predictions