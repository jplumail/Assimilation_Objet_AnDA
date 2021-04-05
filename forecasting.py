import numpy as np
from distance import wasserstein
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
    cov_X = np.swapaxes(X_centered, -1, -2) @ W @ X_centered # shape (...,p,p)
    cov_X_inv = np.linalg.pinv(cov_X) # shape (...,p,p)
    cov_YX = np.swapaxes(Y_centered, -1, -2) @ W @ X_centered # shape (...,p,p)
    y = broadcasted_prod(broadcasted_prod(cov_YX, cov_X_inv), x - X_mean) # shape (...,p,p)
    return


def forecast_step(catalogue, observations, operator, k, distance):
    """forecasting the next state of several observations
    Parameters:
        catalogue: (N,2,p) array, p%5==0 #TODO make the catalogue and the observations broadcastable
        observations: (...,p) array
        operator: an operator from operator.py
        k: number of analogs
        distance: distance.GaussianDistance object
    Returns:
        predictions: (...,p) array"""
    
    # Flatten observations
    observations_flat = observations.reshape(-1,observations.shape[-1]) # shape (M,p)
    predecessors = catalogue[:,0,:] # shape (N,p)
    M = observations_flat.shape[0]
    N = predecessors.shape[0]

    # Compute distances and find k nearest analogs and their successors
    distances = distance(observations_flat[:,np.newaxis,:], predecessors) # shape (M,N)
    indices_wt = np.argpartition(distances, k, axis=-1)[:,:k] # indices of the k nearest analogs, shape (M,k)
    analogs = predecessors[indices_wt] # shape (M,k,p)
    successors = catalogue[indices_wt,0] # successors of the analogs, shape (M,k,p)
    distance_analogs = np.array([distances[i, indices_wt[i]] for i in range(distances.shape[0])]) # shape (M,k)

    # Compute weights and apply the operator
    weights = compute_weights(distance_analogs, np.median(distance_analogs, axis=1))
    predictions = operator(observations_flat, analogs, successors, weights)

    # reshape the predictions
    predictions = predictions.reshape(observations.shape)

    return predictions



def forecast(catalogue, observations, operator, P, k=50, distance=wasserstein):
    """forecasting the next P states
    Parameters:
        catalogue: (N,2,p) array, p%5==0
        observations: (...,p) array
        operator: an operator from operator.py
        P: number of predictions to do, int
        k: number of analogs to consider
        distance: distance.GaussianDistance object
    Returns:
        predictions: (observations.shape, P) array"""
    predictions = np.empty(observations.shape+tuple([P]))
    next_obs = observations
    for j in range(P):
        next_obs = forecast_step(catalogue, next_obs, operator, k, distance)
        predictions[...,j] = next_obs
    return predictions