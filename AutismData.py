"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To parse the autism periodicity data at http://cbslab.org/smm-dataset/
"""
import numpy as np
import scipy.misc
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import cycle
import json
import os
import sys
import glob
import datetime
from lxml import etree as ET
from mpl_toolkits.mplot3d import Axes3D
from VideoTools import *
from ripser import Rips
from GeometricScoring import *


ACCEL_TYPES = ["Trunk", "Left-wrist", "Right-wrist"]
ACCEL_NUMS = ["01", "08", "11"]
#ACCEL_TYPES = ["Right-Wrist", "Left-Wrist", "Torso"]
#ACCEL_NUMS = ["00", "01", "02"]



########################################################################
##                  Functions for parsing annotations                 ##
########################################################################

def getTime(s):
    """
    Convert time from YYYY-MM-DD HH:MM:SS.mmmm into unix millisecond time
    """
    t = time.mktime(datetime.datetime.strptime(s[0:-4], "%Y-%m-%d %H:%M:%S").timetuple())
    t = t*1000 + float(s[-3:]) - 5*3600000  #Hack: Somehow the data is ahead by 5 hours
    return t

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





########################################################################
##           Functions for dealing with accelerometer data            ##
########################################################################

def smoothDataMean(x, gaussSigma = 3):
    """
    Given data in a 2D array, apply a sliding window mean to
        each column
    :param x: An Nxk array of k data streams
    :param gaussSigma: Sigma of sliding window Gaussian
    :return xret: An Mxk array of k smoothed data streams, M < N
    """
    gaussSigma = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussSigma, gaussSigma+1, dtype=np.float64))**2/(2*gaussSigma**2))
    xret = []
    for k in range(x.shape[1]):
        xsmooth = np.convolve(x[:, k], g, 'valid')
        xsmooth = xsmooth.flatten()
        xret.append(xsmooth.tolist())
    return np.array(xret).T

def smoothDataMedian(x, Win = 3):
    """
    Given data in a 2D array, apply a sliding window median to
        each column
    :param x: An Nxk array of k data streams
    :param Win: Width of window
    :return xret: An Mxk array of k smoothed data streams, M <= N
    """
    from scipy.signal import medfilt
    xret = []
    for k in range(x.shape[1]):
        xret.append(medfilt(x[:, k], kernel_size=Win).tolist())
    return np.array(xret).T

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
    return X[i1:i2+1, :]






########################################################################
##           Class for dealing with OpenPose and video data           ##
########################################################################


class Pose(object):
    POSE_SKELETON = [[10, 9], [13, 12], [9, 8], [12, 11], [8, 1], [11, 1], \
                            [2, 1], [5, 1], [4, 3], [3, 2], [7, 6], [6, 5], \
                            [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
    def __init__(self, fileprefix):
        """
        Load open pose keypoints and image
        """
        self.fileprefix = fileprefix
        self.I = scipy.misc.imread("%s.jpg"%fileprefix)
        with open("%s_keypoints.json"%fileprefix) as fin:
            res = json.load(fin)
        people = res['people']
        self.people = []
        for p in people:
            k = p['pose_keypoints']
            x = np.reshape(k, (int(len(k)/3), 3))
            self.people.append({'pose':x})
    
    def render(self):
        plt.imshow(self.I)
        mpl.style.use('default')
        colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        for p, color in zip(self.people, colors):
            #First plot keypoints
            keypts = p['pose']
            toplot = keypts[keypts[:, 2] > 0, :]
            plt.scatter(toplot[:, 0], toplot[:, 1], c=color)
            #Now plot skeleton
            for [i, j] in Pose.POSE_SKELETON:
                if keypts[i, 2] > 0 and keypts[j, 2] > 0:
                    plt.plot(keypts[[i, j], 0], keypts[[i, j], 1], c=color)
        plt.axis('off')
        plt.xlim([0, self.I.shape[1]])
        plt.ylim([self.I.shape[0], 0])
    
def getVideo(studydir, renderpath = ""):
    """
    Given the path to the annotations/XML folder, 
    figure out the path to video frames and load in the video

    Parameters
    ----------
    studydir: string
        Path to the directory of the study for this video
    renderpath: string, default ""
        Path to which to save video of keypoints
    """
    study = studydir.split("/")[-1]
    folder = ""
    for root, dirs, files in os.walk("AutismVideos"):
        if len(folder) > 0:
            break
        for d in dirs:
            if len(folder) > 0:
                break
            if study in d:
                folder = "%s/%s"%(root, d)
                print("Loading video from folder %s"%folder)
    if len(folder) == "":
        print("Error: No video folder found for %s"%studydir)
        return None
    video = []
    allprefixes = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f[-3::] == "jpg":
                fileprefix = "%s/%s"%(root, f[0:-4])
                allprefixes.append(fileprefix)
    allprefixes = sorted(allprefixes)
    for i, fileprefix in enumerate(allprefixes):
        print("Loading video frame %i of %i"%(i+1, len(allprefixes)))
        video.append(Pose(fileprefix))
        if len(renderpath) > 0:
            plt.clf()
            video[-1].render()
            plt.savefig("%s%i.png"%(renderpath, i))
    return video




if __name__ == '__main__':
    NA = len(ACCEL_TYPES)
    studyname = "URI-001-01-18-08"
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

    anno = expandAnnotations(allanno, time = 3000)
    print("There are %i annotations"%len(anno))

    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(NA)]

    dim = 30
    derivWin = -1

    fac = 0.7
    plt.figure(figsize=(fac*15, fac*5*NA))
    FigH = 10*NA+1
    rips = Rips()
    for i in range(1, len(anno)):
        plt.clf()
        a = anno[i]
        dT = (a['stop'] - a['start'])/1000.0
        for k in range(NA):
            print(ACCEL_TYPES[k])
            x = getAccelerometerRange(Xs[k], a)[:, 1::]
            #x = smoothDataMedian(x, 5)
            print("x.shape = ", x.shape)

            plt.subplot(NA, 3, k*3+1)
            plt.plot(np.linspace(0, dT, len(x)), x)
            plt.xlabel("Time (Sec)")
            plt.ylabel("Accelerometer Reading")
            plt.legend(["X", "Y", "Z"])
            #plt.title(a['label'])
            plt.title("Accelerometer %s"%ACCEL_TYPES[k])

            #Step 1: Plot ordinary SSM
            plt.subplot(NA, 3, k*3+2)
            
            D = getCSM(x, x)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest', extent = (0, dT, 0, dT))
            plt.title("SSM %s"%ACCEL_TYPES[k])
            plt.xlabel("Time (Sec)")
            plt.ylabel("Time (Sec)")

            #Step 2: Plot persistence diagram
            plt.subplot(NA, 3, k*3+3)
            res = getPersistencesBlock(x, dim, derivWin = derivWin)
            [I, P, D] = [res['I'], res['P'], res['D']]
            if I.size > 0:
                rips.plot(diagrams=[I], labels=['H1'], size=50, show=False, \
                            xy_range = [0, 2, 0, 2])
            plt.title("Sliding Window \nPersistence Diagram\nMax = %.3g"%P)

            #Step 3: Plot SSM
            """
            plt.subplot(NA, 4, k*4+4)
            plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
            """

        plt.tight_layout()
        plt.savefig("%i.svg"%i, bbox_inches='tight')

if __name__ == '__main__2':
    video = getVideo("URI-001-01-18-08")
    video = video[600:1200]
    for i, f in enumerate(video):
        plt.clf()
        f.render()
        plt.savefig("%i.png"%i)