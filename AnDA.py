from distance import wasserstein
from forecasting_operators import compute_weights
import numpy as np


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