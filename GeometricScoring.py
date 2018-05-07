import numpy as np
import scipy.io as sio
from scipy import sparse
import time
from ripser import Rips
from VideoTools import *

def getMeanShift(X, theta = np.pi/16):
    N = X.shape[0]
    eps = np.cos(theta)
    XS = X/np.sqrt(np.sum(X**2, 1))[:, None]
    D = XS.dot(XS.T)
    J, I = np.meshgrid(np.arange(N), np.arange(N))
    J = J[D >= eps]
    I = I[D >= eps]
    V = np.ones(I.size)
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    XMean = np.zeros(X.shape)
    for i in range(N):
        idx = D[i, :].nonzero()[1]
        XMean[i, :] = np.mean(X[idx, :], 0)
    return XMean

def getMeanShiftKNN(X, K):
    N = X.shape[0]
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    allidx = np.argsort(D, 1)
    XMean = np.zeros(X.shape)
    for i in range(N):
        idx = allidx[i, 0:K]
        XMean[i, :] = np.mean(X[idx, :], 0)
    return XMean

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

def doRipsFiltrationDMGUDHI(D, maxHomDim, coeff = 2, doPlot = False):
    import gudhi
    rips = gudhi.RipsComplex(distance_matrix=D,max_edge_length=np.inf)
    simplex_tree = rips.create_simplex_tree(max_dimension=maxHomDim+1)
    diag = simplex_tree.persistence(homology_coeff_field=coeff, min_persistence=0)
    if doPlot:
        pplot = gudhi.plot_persistence_diagram(diag)
        pplot.show()
    Is = []
    for i in range(maxHomDim+1):
        Is.append([])
    for (i, (b, d)) in diag:
        Is[i].append([b, d])
    for i in range(len(Is)):
        Is[i] = np.array(Is[i])
    return Is

def getSlidingWindow(XP, dim, derivWin = -1, Tau = 1, dT = 1):
    """
    Return a sliding window video
    :param XP: An N x d matrix of N frames each with d pixels
    :param dim: The dimension of the sliding window
    :param derivWin: Whether or not to do a time derivative of each pixel
    :returns: XS: The sliding window video
    """
    X = np.array(XP)
    #Do time derivative
    if derivWin > -1:
        X = getTimeDerivative(X, derivWin)[0]

    #Get sliding window
    if X.shape[0] <= dim:
        return np.array([[0, 0]])
    XS = getSlidingWindowVideo(X, dim, Tau, dT)

    #Mean-center and normalize sliding window
    #XS = XS - np.mean(XS, 1)[:, None] #Point Centering
    XS = XS - np.mean(XS, 0)[None, :] #Z Normalizing
    XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
    return XS

def getPersistencesBlock(XP, dim, estimateFreq = False, derivWin = -1, cosineDist = False, birthcutoff = 0.7):
    """
    Return the Sw1Pers score of this block
    """
    XS = getSlidingWindow(XP, dim, derivWin)
    #XS = getMeanShift(XS)
    if cosineDist:
        D = XS.dot(XS.T)
        D[D > 1] = 1
        D[D < -1] = -1
        D = np.arccos(D)/np.pi
    else:
        D = getCSM(XS, XS)
    Pers = 0
    I = np.array([[0, 0]])
    try:
        #PDs = rips.fit_transform(D, distance_matrix=True)
        PDs = doRipsFiltrationDMGUDHI(D, 1, coeff=41)
        I = PDs[1]
        if I.size > 0:
            ISub = np.array(I) 
            ISub = ISub[ISub[:, 0] <= birthcutoff, :]
            Pers = np.max(ISub[:, 1] - ISub[:, 0])
    except Exception:
        print("EXCEPTION")
    return {'D':D, 'P':Pers, 'I':I}

def getD2ChiSqr(XP, dim, estimateFreq = False, derivWin = -1):
    """
    Return the Chi squared distance to the perfect circle distribution
    """
    XS = getSlidingWindow(XP, dim, estimateFreq, derivWin)
    ##TODO
