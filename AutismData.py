"""
Programmer: Chris Tralie (ctralie@alumni.princeton.edu)
Purpose: To parse the autism periodicity data at http://cbslab.org/smm-dataset/
"""
import numpy as np
import scipy.misc
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
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
from scipy.spatial import ConvexHull
import dlib


IMPROVE_WITH_DLIB = False

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
    t = t*1000 + float(s[-3:]) + 1*3600000  #Hack: Somehow the data is ahead by 5 hours
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

def getConvexMask(hull, Y, shape):
    """
    Return a mask that covers the region inside of the specified convex hull
    Parameters
    ----------
    hull: scipy.spatial.convexHull
        A convex hull
    Y: ndarray (N, 2)
        An array of 2D points that convex hull indexes into
    shape: shape of image mask

    Returns
    -------
    mask: ndarray (shape)
        A mask with 1s inside the region and 0s outside
    """
    J, I = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    J = J.flatten()
    I = I.flatten()
    inside = np.ones(J.size, dtype = np.uint8)
    for a, b, c in hull.equations:
        res = a*J + b*I + c
        inside *= (res < 0)
    return np.reshape(inside, (shape[0], shape[1]))

def getCircleMask(hull, Y, shape):
    """
    Return a mask that covers an enclosing circle of the specified convex hull
    ----------
    hull: scipy.spatial.convexHull
        A convex hull
    Y: ndarray (N, 2)
        An array of 2D points that convex hull indexes into
    shape: shape of image mask

    Returns
    -------
    mask: ndarray (shape)
        A mask with 1s inside the region and 0s outside
    """
    J, I = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    J = J.flatten()
    I = I.flatten()
    inside = np.ones(J.size, dtype = np.uint8)
    idxs = set([])
    for i, j in hull.simplices:
        idxs.add(i)
        idxs.add(j)
    X = Y[np.array(list(idxs)), :]
    muX = np.mean(X, 0)
    RSqr = np.max(np.sum((X-muX[None, :])**2, 1))
    inside = (I-muX[1])**2 + (J-muX[0])**2 < RSqr
    inside = np.array(inside, dtype=np.uint8)
    return np.reshape(inside, (shape[0], shape[1]))

def shape_to_np(shape, dtype="float"):
    return np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=dtype)

class Pose(object):
    POSE_SKELETON = [[10, 9], [13, 12], [9, 8], [12, 11], [8, 1], [11, 1], \
                            [2, 1], [5, 1], [4, 3], [3, 2], [7, 6], [6, 5], \
                            [1, 0], [0, 14], [0, 15], [14, 16], [15, 17]]
    POSE_HAND = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],\
                [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],\
                [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    def __init__(self, fileprefix):
        """
        Load open pose keypoints and image, and initialize timestamp
        """
        self.fileprefix = fileprefix
        print(fileprefix)
        self.I = scipy.misc.imread("%s.jpg"%fileprefix)
        with open("%s_keypoints.json"%fileprefix) as fin:
            res = json.load(fin)
        people = res['people']
        self.people = []
        for p in people:
            pobj = {}
            for s in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d'):
                k = p[s]
                x = np.reshape(k, (int(len(k)/3), 3))
                pobj[s] = x
            self.people.append(pobj)
        #YYYY-MM-DD HH:MM:SS.mmmm
        s = fileprefix.split("/")[-1]
        fields = tuple(s.split("-"))
        self.timestamp = getTime("%s-%s-%s %s:%s:%s:%s"%fields)
    
    def render(self, showLandmarks = True, blurFace = True, blurlast = []):
        I = np.array(self.I)
        xblurs = []

        mpl.style.use('default')
        colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        for p, color in zip(self.people, colors):
            if showLandmarks:
                #First plot keypoints
                for keypts, skeleton, sz in zip( (p['pose_keypoints_2d'], p['hand_left_keypoints_2d'], \
                                                                    p['hand_right_keypoints_2d']),\
                                            (Pose.POSE_SKELETON, Pose.POSE_HAND, Pose.POSE_HAND),\
                                            (20, 3, 3) ):
                    toplot = keypts[keypts[:, 2] > 0, :]
                    plt.scatter(toplot[:, 0], toplot[:, 1], sz, c=color)
                    #Now plot skeleton
                    for [i, j] in skeleton:
                        if keypts[i, 2] > 0 and keypts[j, 2] > 0:
                            plt.plot(keypts[[i, j], 0], keypts[[i, j], 1], c=color)
            if blurFace:
                x1 = p['face_keypoints_2d'][:, 0:2]
                x2 = p['pose_keypoints_2d'][[0, 14, 15, 16, 17], 0:2]
                x = np.concatenate((x1, x2), 0)
                x = x[np.sum(x, 1) > 0, :]
                xblurs.append(x)
                if x.size > 0:
                    try:
                        hull = ConvexHull(x)
                        mask = getCircleMask(hull, x, I.shape)
                        I *= (1-mask[:, :, None])
                    except:
                        print("Convex hull error")
        if blurFace and IMPROVE_WITH_DLIB:
            predictor_path = "shape_predictor_68_face_landmarks.dat"
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(predictor_path)
            dets = detector(self.I, 1)
            xs = [shape_to_np(predictor(self.I, d)) for d in dets]
            xblurs += xs
            for x in xs + blurlast:
                if x.size > 0:
                    try:
                        hull = ConvexHull(x)
                        mask = getCircleMask(hull, x, I.shape)
                        I *= (1-mask[:, :, None])
                    except:
                        print("Convex hull error")
        plt.axis('off')
        plt.imshow(I)
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])
        return xblurs


def updateAssociations(video, npreceding=30):
    """
    Given a sequence of pose objects, make sure the first skeleton is 
    more similar with the first skeleton in some preceding number of frames
    than other skeletons are to the first skeleton (swap if not the case)
    Parameters
    ----------
    videos: array (Pose)
        An array of sequential pose objects in a video
    npreceding: int
        The number of preceding frames to consider
    
    """
    N = min(npreceding, len(video)-1)
    minDists = np.inf
    minIdx = 0
    for i in range(len(video[-1].people)):
        distsi = 0.0
        x = video[-1].people[i]['pose_keypoints_2d']
        for j in range(N):
            y = video[-2-j].people[0]['pose_keypoints_2d']
            idx = (np.sum(x, 1) > 0)*(np.sum(y, 1) > 0)
            dists = np.sqrt(np.sum((x[idx, :] - y[idx, :])**2, 1))
            distsi += np.mean(dists)
        if distsi < minDists:
            minDists = distsi
            minIdx = i
    if minIdx > 0:
        p = video[-1].people[0]
        video[-1].people[0] = video[-1].people[minIdx]
        video[-1].people[minIdx] = p






def getVideo(studydir, save_skeletons = False, blurFace = False, framerange = (0, np.inf)):
    """
    Given the path to the annotations/XML folder, 
    figure out the path to video frames and load in the video

    Parameters
    ----------
    studydir: string
        Path to the directory of the study for this video
    save_skeletons: boolean, default False
        Whether to save video of keypoints/skeletons
    blurFace: boolean, default False
        If saving keypoints/skeletons, whether to blur the face
    framerange: tuple, default (0, inf)
        Range of frames to load in the video.  If not specified, all
        frames are loaded
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
    skeletondir = ""
    blurlast = []
    blurlastlast = []
    if save_skeletons:
        skeletondir = "%s/Skeletons"%folder
        if not os.path.exists(skeletondir):
            os.mkdir(skeletondir)
    video = []
    allprefixes = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f[-3::] == "jpg":
                fileprefix = "%s/%s"%(root, f[0:-4])
                allprefixes.append(fileprefix)
    allprefixes = sorted(allprefixes)
    for i, fileprefix in enumerate(allprefixes):
        if i < framerange[0] or i > framerange[1]:
            continue
        print("Loading video frame %i of %i"%(i+1, len(allprefixes)))
        video.append(Pose(fileprefix))
        updateAssociations(video)
        if save_skeletons:
            plt.clf()
            if blurFace:
                blurlastlast = blurlast
                blurlast = video[-1].render(blurFace=True, blurlast=blurlast+blurlastlast)
            else:
                video[-1].render(blurFace=False)
            plt.savefig("%s/%s.png"%(skeletondir, fileprefix.split("/")[-1]))
    return video

def getNearestVideoFrame(video, timestamp):
    ts = np.array([f.timestamp for f in video])
    idx = np.argmin(np.abs(ts-timestamp))
    return video[idx]


if __name__ == '__main__2':
    NA = len(ACCEL_TYPES)
    studyname = "URI-001-01-18-08"
    foldername = "neudata/data/Study1/URI-001-01-18-08"
    annofilename = "Annotator1Stereotypy.annotation.xml"
    anno = loadAnnotations("%s/%s"%(foldername, annofilename))
    ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
    anno = anno[1::]
    #anno = anno + getNormalAnnotations(anno)
    #nanno = getNormalAnnotations(anno)
    #allanno = anno[2::] + nanno
    allanno = anno[3::]

    idx = np.argsort([a['start'] for a in allanno])
    allanno = [allanno[i] for i in idx]
    X = np.array([[a['start']-allanno[0]['start'], a['stop']-allanno[0]['start']] for a in allanno])
    sio.savemat("Intervals.mat", {"X":X})

    anno = expandAnnotations(allanno, time = 3000)
    print("There are %i annotations"%len(anno))

    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(NA)]

    dim = 30
    derivWin = -1

    fac = 0.5
    plt.figure(figsize=(fac*15, fac*5*NA))
    FigH = 10*NA+1
    rips = Rips()
    for i in range(1, len(anno)):
        plt.clf()
        a = anno[i]
        print(a)
        dT = (a['stop'] - a['start'])/1000.0
        for k in range(NA):
            print(ACCEL_TYPES[k])
            x = getAccelerometerRange(Xs[k], a)[:, 1::]
            #x = smoothDataMedian(x, 5)

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
            res = getPersistencesBlock(x, dim, derivWin = derivWin, birthcutoff=np.inf)
            [I, P, DTDA] = [res['I'], res['P'], res['D']]
            if I.size > 0:
                rips.plot(diagrams=[I], labels=['H1'], size=50, show=False, \
                            xy_range = [0, 2, 0, 2])
            if k == 0:
                plt.title("Sliding Window \nPersistence Diagram\nMax = %.3g"%P)
            else:
                plt.title("Max = %.3g"%P)

            if i == 8 and k == 0:
                sio.savemat("D.mat", {"D":D, "DTDA":DTDA, "dT":dT})

            #Step 3: Plot SSM
            """
            plt.subplot(NA, 4, k*4+4)
            plt.imshow(DTDA, cmap = 'afmhot', interpolation = 'nearest')
            """

        plt.tight_layout()
        plt.savefig("%i.svg"%i, bbox_inches='tight')

if __name__ == '__main__':
    # 640
    video = getVideo("URI-001-01-18-08", save_skeletons = True, blurFace = False, framerange = (600, 1500))
