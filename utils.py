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
