import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from SlidingWindowVideoTDA.FundamentalFreq import *
from SlidingWindowVideoTDA.TDA import *
from SlidingWindowVideoTDA.VideoTools import *

def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getSlidingWindow(XP, dim, estimateFreq = False, derivWin = -1):
    """
    Return a sliding window video
    :param XP: An N x d matrix of N frames each with d pixels
    :param dim: The dimension of the sliding window
    :param estimateFreq: Whether or not to estimate the fundamental frequency
    or to just use dim as the window size with Tau = 1
    :param derivWin: Whether or not to do a time derivative of each pixel
    :returns: XS: The sliding window video
    """
    X = np.array(XP)
    #Do time derivative
    if derivWin > -1:
        X = getTimeDerivative(X, derivWin)[0]
    pca = PCA(n_components = 1)

    I = np.array([[0, 0]])
    Pers = 0.0

    Tau = 1
    dT = 1
    #Do fundamental frequency estimation
    if estimateFreq:
        xpca = pca.fit_transform(X)
        (maxT, corr) = estimateFundamentalFreq(xpca.flatten(), False)
        #Choose sliding window parameters
        Tau = maxT/float(dim)

    #Get sliding window
    if X.shape[0] <= dim:
        return (Pers, I)
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    XS = XS - np.mean(XS, 1)[:, None]
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    return XS

def getPersistencesBlock(XP, dim, estimateFreq = False, derivWin = -1):
    """
    Return the Sw1Pers score of this block
    """
    XS = getSlidingWindow(XP, dim, estimateFreq, derivWin)
    try:
        PDs2 = doRipsFiltration(XS, 1, coeff=41)
        I = PDs2[1]
        if I.size > 0:
            Pers = np.max(I[:, 1] - I[:, 0])
    except Exception:
        print "EXCEPTION"
    return (Pers, I)

def getD2ChiSqr(XP, dim, estimateFreq = False, derivWin = -1):
    """
    Return the Chi squared distance to the perfect circle distribution
    """
    XS = getSlidingWindow(XP, dim, estimateFreq, derivWin)
    ##TODO
