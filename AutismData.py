"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To parse the autism periodicity data at http://cbslab.org/smm-dataset/
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import datetime
from lxml import etree as ET
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("SlidingWindowVideoTDA")
from VideoTools import *
from FundamentalFreq import *
from TDA import *
from sklearn.decomposition import PCA


ACCEL_TYPES = ["Trunk", "Left-wrist", "Right-wrist"]
ACCEL_NUMS = ["01", "08", "11"]

def getTime(s):
    """
    Convert time from YYYY-MM-DD HH:MM:SS.mmmm into unix millisecond time
    """
    t = time.mktime(datetime.datetime.strptime(s[0:-4], "%Y-%m-%d %H:%M:%S").timetuple())
    t = t*1000 + float(s[-3:]) - 5*3600000  #Hack: Somehow the data is ahead by 5 hours
    return t

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

def loadAccelerometerData(filename):
    X = np.loadtxt(filename, delimiter=",")
    return X

def getAccelerometerRange(X, a):
    """
    Return the accelerometer data in the range of an annotation
    :param X: N x 4 array of accelerometer data, (time, ax, ay, az) per row
    :param a: Annotation {'start':float, 'stop':float, 'label':string}
    """
    t1 = a['start']
    t2 = a['stop']
    i1 = np.arange(X.shape[0])[np.argmin(np.abs(X[:, 0] - t1))]
    i2 = np.arange(X.shape[0])[np.argmin(np.abs(X[:, 0] - t2))]

#    print "t1 = %i, i1 = %g, val = %g"%(t1, i1, X[i1, 0])
#    print "t2 = %i, i2 = %g"%(t2, i2)
#    print t1 - X[i1, 0]
#    
#    plt.clf()
#    plt.plot(X[:, 0])
#    plt.hold(True)
#    plt.scatter([X.shape[0]+1, X.shape[0]+2], [t1, t2])
#    plt.show()

    return X[i1:i2+1, :]

def loadAnnotations(filename):
    """
    Load annotations into dictionary format
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    anno = []
    for m in root.getchildren():
        start = -1
        stop = -1
        label = ""
        for c in m.getchildren():
            if c.tag == "START_DT":
                start = getTime(c.text)
            elif c.tag == "STOP_DT":
                stop = getTime(c.text)
            elif c.tag == "LABEL":
                label = c.text
        anno.append({"start":start, "stop":stop, "label":label})
    return anno

def getNormalAnnotations(anno, minTime = 5000, length=3500):
    """
    Return annotations for regions where no sterotypical motions
    are labeled
    """
    nanno = []
    for i in range(len(anno)-1):
        start = anno[i]['stop']
        stop = anno[i+1]['start']
        dT = stop - start
        if dT > minTime:
            #Find a random time interval of length minTime
            diff = dT - length
            start = start + 0.5*diff
            stop = start + length
            nanno.append({"start":start, "stop":stop, "label":"Normal"})
    return nanno

def visualizeLabels(anno, thisa = None, relative = True, doLegend = True):
    """
    Plot the annotations in time, labeling with colors
    :param anno: A list of annotations of the form [{'start':float, 'stop':float, 'label':string}, ...]
    :param thisa: A particular annotation to highlight
    :param relative: Whether or not to plot times in seconds relative to the beginning
    :param doLegend: Whether or not to plot the legend of motions
    """
    labels = {}
    legends = {}
    plt.hold(True)
    colors = ['r', 'g', 'b', 'm', 'c']
    idx = 0
    minTime = np.inf
    for a in anno:
        minTime = min(minTime, a['start'])
    for a in anno:
        t1 = a['start']
        t2 = a['stop']
        l = a['label']
        if not l in labels:
            labels[l] = len(labels)
        if relative:
            t1 = (t1-minTime)/1000.0
            t2 = (t2-minTime)/1000.0
        h = plt.plot([t1, t2], [0, 0], colors[labels[l]], linewidth=4, label = l)[0]
        legends[l] = h
        idx += 1
    if thisa:
        t1 = thisa['start']
        t2 = thisa['stop']
        l = thisa['label']
        if relative:
            t1 = (t1-minTime)/1000.0
            t2 = (t2-minTime)/1000.0
        plt.plot([t1, t1], [-1, 1], colors[labels[l]])
        plt.plot([t2, t2], [-1, 1], colors[labels[l]])
        plt.title(l + ", %g Seconds"%(t2-t1))
    if doLegend:
        plt.legend(handles = [legends[l] for l in legends])


def getMaxPersistenceBlock(XParam, hopSize, winSize, dim, estimateFreq = False, derivWin = -1):
    XP = np.array(XParam)
    #Do time derivative
    if derivWin > -1:
        XP = getTimeDerivative(XP, derivWin)[0]
    NBlocks = 1
    if XP.shape[0] > winSize:
        NBlocks = int(np.ceil((XP.shape[0]-winSize)/float(hopSize)))
    pca = PCA(n_components = 1)
    maxP = 0.0 #Maximum persistence
    maxI = np.array([[0, 0]])
    maxidx = 0
    for i in range(NBlocks):
        X = XP[i*hopSize:i*hopSize+winSize, :]
        
        xpca = pca.fit_transform(X)
        Tau = 1
        dT = 1
        #Do fundamental frequency estimation
        if estimateFreq:
            (maxT, corr) = estimateFundamentalFreq(xpca.flatten(), False)
            #Choose sliding window parameters 
            Tau = maxT/float(dim)
        
        #Get sliding window
        if X.shape[0] <= dim:
            break
        XS = getSlidingWindowVideo(X, dim, Tau, dT)
        
        #Mean-center and normalize sliding window
        XS = XS - np.mean(XS, 1)[:, None]
        XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
        sio.savemat("XS.mat", {"XS":XS})
        
        try:
            PDs2 = doRipsFiltration(XS, 1, coeff=2)
            I1 = PDs2[1]
            if I1.size > 0:
                val = np.max(I1[:, 1] - I1[:, 0])
                if val > maxP:
                    maxP = val
                    maxI = I1
                    maxidx = i*hopSize
        except Exception:
            print "EXCEPTION"
    return (maxP/np.sqrt(3), maxI, maxidx)
    

if __name__ == '__main__':
    foldername = "neudata/data/Study1/URI-001-01-18-08"
    annofilename = "Annotator1Stereotypy.annotation.xml"
    anno = loadAnnotations("%s/%s"%(foldername, annofilename))
    ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
    anno = anno + getNormalAnnotations(anno[1::])[1::]
    #anno = getNormalAnnotations(anno[1::])[1::]
    
    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]

    dim = 30

    plt.figure(figsize=(15, 5*len(ACCEL_TYPES)))
    FigH = 10*len(ACCEL_TYPES)+1
    for i in range(1, len(anno)):
        plt.clf()
        a = anno[i]
        plt.subplot2grid((FigH, 3), (0, 0), rowspan=1, colspan=3)
        visualizeLabels(anno, a, doLegend = False)
        plt.axis('off')
        
        for k in range(len(ACCEL_TYPES)):
            print ACCEL_TYPES[k]
            x = getAccelerometerRange(Xs[k], a)
            
            #Step 1: Plot ordinary SSM
            ax = plt.subplot2grid((FigH, 3), (1+k*10, 0), rowspan=8, colspan=1)
            dT = (a['stop'] - a['start'])/1000.0
            D = getCSM(x[:, 1::], x[:, 1::])
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest', extent = (0, dT, 0, dT))
            plt.title(ACCEL_TYPES[k])
            
            #Step 2: Do delay embedding and plot delay SSM
            plt.subplot2grid((FigH, 3), (1+k*10, 1), rowspan = 8, colspan = 1)
            XS = getSlidingWindowVideo(x[:, 1::], dim, 1, 1)
            #Mean-center and normalize sliding window
            XS = XS - np.mean(XS, 1)[:, None]
            XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
            D = getCSM(XS, XS)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
            
            #Step 3: Plot persistence diagram
            plt.subplot2grid((FigH, 3), (1+k*10, 2), rowspan = 8, colspan = 1)
            (maxP, maxI, maxidx) = getMaxPersistenceBlock(x[:, 1::], 20, 150, dim)
            plotDGM(maxI, color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
            plt.title("maxP = %.3g, maxidx = %i"%(maxP, maxidx))
        
        plt.savefig("%i.png"%i, bbox_inches='tight')

if __name__ == '__main__2':
    XS = sio.loadmat("XS.mat")['XS']
    print XS.shape
    PDs = doRipsFiltration(XS, 1, coeff=2)
    print PDs
    
