import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.patches as pat
from matplotlib import cm
import os


def det_22(M):
    """M is a stack of positive definite matrices of size 2x2 (...,2,2)
    returns their determinant"""
    delta = M[...,0,0]*M[...,1,1] - M[...,1,0]*M[...,0,1]
    return delta

def sqrtm_22(M):
    """M is a stack of positive definite matrices of size 2x2 (...,2,2)
    returns square root"""
    tau = np.trace(M, axis1=-2, axis2=-1)
    delta = det_22(M)
    s = np.sqrt(delta)
    t = np.sqrt(tau + 2*s)
    return ((M + s[...,np.newaxis,np.newaxis]*np.eye(2)).T / t.T).T

def inv_22(M):
    """M is a stack of positive definite matrices of size 2x2 (...,2,2)
    returns their inverse"""
    invM = np.empty(M.shape)
    delta = det_22(M)
    invM[...,0,0], invM[...,1,1] = M[...,1,1], M[...,0,0]
    invM[...,1,0] = -M[...,1,0]
    invM[...,0,1] = -M[...,0,1]
    invM = invM.T / delta.T
    return invM.T

def broadcasted_prod(A, B):
    """
    A array of shape (...,p,q)
    B array of shape (...,q,r)
    A and B must be broadcastable
    compute a matrix product in a broadcast way between A and B"""
    return np.einsum('...ik,...kj->...ij', A, B)

def gaussian_repr(x):
    """Transform a vector x of shape (...,5) into its gaussian representation
    returns mean (...,2) and covMat (...,2,2)"""
    mean = x[..., :2]
    cov_shape = tuple(list(x.shape[:-1])+[2,2])
    covMat = np.stack([x[..., 2], x[..., 3], x[..., 3], x[..., 4]], axis=-1).reshape(cov_shape)
    return mean, covMat

def vect_repr(mu, covMat):
    """Transform a gaussian into its vector representation
    returns array of shape (...,5)"""
    c = np.stack([covMat[...,0,0],covMat[...,0,1],covMat[...,1,1]], axis=-1)
    return np.concatenate([mu, c], axis=-1)


def compute_weights(distances, l=1.):
    kernels = np.exp(-(distances/l)**2)
    return kernels / np.sum(kernels)


def locally_constant_mean(x, neighbors, successors, weights):
    return np.sum(successors * weights.reshape((-1,1)), axis=0)

def locally_incremental_mean(x, neighbors, successors, weights):
    return x + np.sum((successors - neighbors) * weights.reshape((-1,1)), axis=0)

def locally_linear_mean(x, neighbors, successors, weights):
    W = np.diag(weights)
    X = neighbors
    Y = successors
    X_mean = np.sum(X*weights.reshape((-1,1)), axis=0)
    Y_mean = np.sum(Y*weights.reshape((-1,1)), axis=0)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    cov_X = X_centered.T @ W @ X_centered
    cov_X_inv = np.linalg.pinv(cov_X)
    cov_YX = Y_centered.T @ W @ X_centered
    return Y_mean + cov_YX @ cov_X_inv @ (x - X_mean)


def predictions(catalogue, observations, method, k=100):
    """predict the next state of a list of tourbillons
    observations of shape Mx10, returns array of shape Mx10"""
    tourbillons_suivant=[]
    predecesseurs = catalogue[:,0,:]
    N = predecesseurs.shape[0]
    for tourbillon in observations:
        distances = new_wasserstein(np.stack([tourbillon]*N, axis=0), predecesseurs)
        indices_wt = np.argpartition(distances, k)[:k] # indices of the k nearest neighbors
        neighbors = catalogue[indices_wt,0,:]
        successors = catalogue[indices_wt,1,:]
        distances_neighbors = distances[indices_wt]
        weights = compute_weights(distances_neighbors, l=np.median(distances_neighbors))
        pred = method(tourbillon, neighbors, successors, weights)
        tourbillons_suivant.append(pred)
    return np.array(tourbillons_suivant)



def list_prediction(catalogue,nb_predictions, observations, method, k=50):
    """construit une matrice de taille nombre d'ellipses x nb_predictions x 10
    cette matrice représente les valeurs prédites"""
    mat_prediction = np.empty((observations.shape[0],nb_predictions,observations.shape[1]))
    next_obs = observations
    for j in range(nb_predictions):
        next_obs = predictions(catalogue,next_obs, method, k=k)
        mat_prediction[:,j] = next_obs
    return mat_prediction

def list_true_value(catalogue, means, covMats, nb_predictions,observations,bruit=0.2,center=None,alpha=None,model="follow gradient"):
    """construit une matrice de taille nombre d'ellipses x nb_predictions x 5
    cette matrice représente les valeurs réelles si les tourbillons suivent le modèle prédéfini"""
    mat_true = np.empty((observations.shape[0],nb_predictions,5))
    next_gaussians = observations[:,5:]
    for j in range(nb_predictions):
        next_gaussians = step(next_gaussians, means, covMats, bruit=bruit, center=center, alpha=alpha, model=model)
        mat_true[:,j] = next_gaussians
    return mat_true

def AnDA_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
    return np.sqrt(np.mean((a-b)**2, axis=-1))

def AnDA_Wasserstein(a,b):
    """ Compute the Wassertstein metric between 2 n-dimensional vectors. """
    x, y = a.shape[0], a.shape[1]
    a = a.reshape((-1,5))
    b = b.reshape((-1,5))
    res = wasserstein(a,b)
    res = res.reshape((x,y))
    return res

def load_playground(filename=os.path.join("data", "playground_file.npz")):
    npzfile = np.load(filename)
    list_mean_gauss = npzfile['list_mean_gauss']
    list_covMat_gauss = npzfile['list_covMat_gauss']
    X = npzfile['X']
    Y = npzfile['Y']
    Z = npzfile['Z'].T
    return list_mean_gauss, list_covMat_gauss, X, Y, Z

def load_data(filename ='data\catalogue.txt'):
    data = np.loadtxt(filename)
    data = data.reshape((int(data.shape[0]/2),2,10))
    return data
