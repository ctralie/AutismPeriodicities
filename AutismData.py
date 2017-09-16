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
from SlidingWindowVideoTDA.VideoTools import *
from SlidingWindowVideoTDA.TDA import *
from GeometricScoring import *


ACCEL_TYPES = ["Trunk", "Left-wrist", "Right-wrist"]
ACCEL_NUMS = ["01", "08", "11"]

def getTime(s):
    """
    Convert time from YYYY-MM-DD HH:MM:SS.mmmm into unix millisecond time
    """
    t = time.mktime(datetime.datetime.strptime(s[0:-4], "%Y-%m-%d %H:%M:%S").timetuple())
    t = t*1000 + float(s[-3:]) - 5*3600000  #Hack: Somehow the data is ahead by 5 hours
    return t

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

def getNormalAnnotations(anno, minTime = 4000):
    """
    Return annotations for regions where no sterotypical motions
    are labeled
    """
    nanno = []
    idx = np.argsort([a['start'] for a in anno])
    anno = [anno[i] for i in idx]
    for i in range(len(anno)-1):
        start = anno[i]['stop']
        stop = anno[i+1]['start']
        if stop - start < minTime:
            continue
        nanno.append({"start":start, "stop":stop, "label":"Normal"})
    return nanno

def expandAnnotations(anno, time = 2000, hop = 130):
    """
    Split each annotation into smaller overlapping sub-blocks
    """
    newanno = []
    for a in anno:
        start = a['start']
        stop = a['stop']
        t1 = start
        while t1 + time <= stop:
            newanno.append({"start":t1, "stop":(t1 + time), "label":a["label"]})
            t1 += hop
    return newanno

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
        h = plt.plot([t1, t2], [0, 0], colors[labels[l]], linewidth=1, label = l)[0]
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
        plt.xlim([t1 - 5, t2 + 5])
    if doLegend:
        plt.legend(handles = [legends[l] for l in legends])


if __name__ == '__main__':
    foldername = "neudata/data/Study1/URI-001-01-18-08"
    annofilename = "Annotator1Stereotypy.annotation.xml"
    anno = loadAnnotations("%s/%s"%(foldername, annofilename))
    ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
    anno = anno[1::]
    #anno = anno + getNormalAnnotations(anno)
    nanno = getNormalAnnotations(anno)
    allanno = anno + nanno

    idx = np.argsort([a['start'] for a in allanno])
    allanno = [allanno[i] for i in idx]
    X = np.array([[a['start']-allanno[0]['start'], a['stop']-allanno[0]['start']] for a in allanno])
    sio.savemat("Intervals.mat", {"X":X})

    anno = expandAnnotations(allanno)
    print "There are %i annotations"%len(anno)

    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]

    dim = 30
    derivWin = -1

    plt.figure(figsize=(15, 5*len(ACCEL_TYPES)))
    FigH = 10*len(ACCEL_TYPES)+1
    for i in range(1, len(anno)):
        plt.clf()
        a = anno[i]
        plt.subplot2grid((FigH, 3), (0, 0), rowspan=1, colspan=3)
        visualizeLabels(allanno, a, doLegend = False)
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
            X = np.array(x[:, 1::])
            if derivWin > -1:
                X = getTimeDerivative(X, derivWin)[0]
            XS = getSlidingWindowVideo(X, dim, 1, 1)
            #Mean-center and normalize sliding window
            XS = XS - np.mean(XS, 1)[:, None]
            XS = XS/np.sqrt(np.sum(XS**2, 1))[:, None]
            D = getCSM(XS, XS)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')

            #Step 3: Plot persistence diagram
            plt.subplot2grid((FigH, 3), (1+k*10, 2), rowspan = 8, colspan = 1)
            (P, I) = getPersistencesBlock(x[:, 1::], dim, derivWin = derivWin)
            plotDGM(I, color = np.array([1.0, 0.0, 0.2]), label = 'H1', sz = 50, axcolor = np.array([0.8]*3))
            plt.title("Pers = %.3g"%P)

        plt.savefig("%i.png"%i, bbox_inches='tight')

if __name__ == '__main__2':
    XS = sio.loadmat("XS.mat")['XS']
    print XS.shape
    PDs = doRipsFiltration(XS, 1, coeff=2)
    print PDs
