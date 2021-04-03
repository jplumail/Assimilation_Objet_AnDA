import numpy as np
import utils


class GaussianDistance:
    """Class that defines how a gaussian distance works"""

    def __init__(self, distance):
        """initialize the metric
        distance: a function taking 4 parameters mu1,mu2,covMat1,covMat2 reprsenting gaussians
         of shapes (...,2) for mus and (...,2,2) for covMats
         Computes the distance between the two gaussians (mu1, covMat1) and (mu1, covMat1)
         This function must be broadcastable, ie with arguments of shape :
          mu1.shape, covMat1.shape = (2), (2,2)
          mu2.shape, covMat2.shape = (100,2), (100,2,2)
         it should return an array of shape (100) representing the distances between 1 gaussian and the others
        """
        self.distance = distance
    
    def __call__(self, x, y):
        """Compute the distance between x and y
        x and y of shapes (...,p), x and y must be broadcastable
        p % 5 == 0
        return an array of shape (...) depending on the broadcast
        """
        q = x.shape[-1] // 5
        new_x_shape = list(x.shape)[:-1] + [q,5]
        new_y_shape = list(y.shape)[:-1] + [q,5]
        mu1, covMat1 = utils.gaussian_repr(x.reshape(tuple(new_x_shape)))
        mu2, covMat2 = utils.gaussian_repr(y.reshape(tuple(new_y_shape)))
        return self.distance(mu1, mu2, covMat1, covMat2).sum(axis=-1)


def wasserstein_metric(mu1,mu2,covMat1,covMat2):
    """mus of shape (...,2), covMats of shape (...,2,2)
    The shapes of the first and second set of gaussians must be broadcastable
    returns the wasserstein distance squared"""
    rC2 = utils.sqrtm_22(covMat2)
    mat = covMat1 + covMat2 - (2*utils.sqrtm_22(rC2 @ covMat1 @ rC2))
    wasserstein = np.linalg.norm(mu1-mu2, axis=-1)**2 + np.trace(mat, axis1=-2, axis2=-1)
    return wasserstein

def hellinger_metric(mu1,mu2,covMat1,covMat2):
    """mus of shape (...,2), covMats of shape (...,2,2)
    The shapes of the first and second set of gaussians must be broadcastable
    returns the hellinger distance squared"""
    coef = (utils.det_22(covMat1)*utils.det_22(covMat2))**(1/4) / utils.det_22(0.5*(covMat1+covMat2))
    mu1 = mu1[...,:,np.newaxis]
    mu2 = mu2[...,:,np.newaxis]
    exp_ = utils.broadcasted_prod(
        utils.broadcasted_prod(np.swapaxes(mu1-mu2,-1,-2), utils.inv_22(0.5*(covMat1+covMat2))),
        mu1-mu2
    )[...,0,0]
    hellinger = 1-(coef*np.exp((-1/8)*exp_))
    return hellinger

def KL_div(mu1,mu2,covMat1,covMat2):
    """mus of shape (...,2), covMats of shape (...,2,2)
    The shapes of the first and second set of gaussians must be broadcastable
    returns the KL divergence squared"""
    invCovmat2 = utils.inv_22(covMat2)
    mu1 = mu1[...,:,np.newaxis]
    mu2 = mu2[...,:,np.newaxis]
    prod = utils.broadcasted_prod(
        utils.broadcasted_prod(np.swapaxes(mu1-mu2,-1,-2), invCovmat2),
        mu1-mu2
    )[...,0,0]
    tr = np.trace(invCovmat2 @ covMat1, axis1=-2, axis2=-1)
    kl = 0.5*(tr + prod - 2 + np.log(det_22(covMat2)/det_22(covMat1)))
    return kl


# Distances implemented so far
wasserstein = GaussianDistance(wasserstein_metric)
hellinger = GaussianDistance(hellinger_metric)
kl = GaussianDistance(KL_div)