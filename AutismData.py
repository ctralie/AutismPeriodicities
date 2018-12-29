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
import calendar
from lxml import etree as ET
from mpl_toolkits.mplot3d import Axes3D
from VideoTools import *
from ripser import Rips, plot_dgms
from GeometricScoring import *
from scipy.spatial import ConvexHull

IMPROVE_WITH_DLIB = False
if IMPROVE_WITH_DLIB:
    import dlib

ACCEL_TYPES = ["Trunk", "Left-wrist", "Right-wrist"]
ACCEL_NUMS = ["01", "08", "11"]
#ACCEL_TYPES = ["Right-Wrist", "Left-Wrist", "Torso"]
#ACCEL_NUMS = ["00", "01", "02"]



########################################################################
##                  Functions for parsing annotations                 ##
########################################################################

def getTime(s):
    """
    Convert time from YYYY-MM-DD HH:MM:SS.mmm into unix millisecond time
    """
    # Extract milliseconds first
    spre, smilli = s.split(".")
    t = calendar.timegm(datetime.datetime.strptime(spre, "%Y-%m-%d %H:%M:%S").timetuple())
    t = t*1000 + float(smilli)
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
        #YYYY-MM-DD HH:MM:SS.mmmm
        s = fileprefix.split("/")[-1]
        fields = tuple(s.split("-"))
        self.timestamp = getTime("%s-%s-%s %s:%s:%s.%s"%fields)
        self.I = scipy.misc.imread("%s.jpg"%fileprefix)
        self.people = []
        with open("%s_keypoints.json"%fileprefix) as fin:
            try:
                res = json.load(fin)
            except:
                return
        people = res['people']
        for p in people:
            pobj = {}
            for s in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'face_keypoints_2d'):
                k = p[s]
                x = np.reshape(k, (int(len(k)/3), 3))
                pobj[s] = x
            self.people.append(pobj)
    
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



class PoseVideo(object):
    def __init__(self, studydir, save_skeletons = False, delete_frames = True, blurFace = False, framerange = (0, np.inf)):
        """
        Given the path to the annotations/XML folder, 
        figure out the path to video frames and load in the video

        Parameters
        ----------
        studydir: string
            Path to the directory of the study for this video
        save_skeletons: boolean, default False
            Whether to save video of keypoints/skeletons
        delete_frames: boolean, default True
            If saving skeleton, whether to delete the individual frames and
            make a video
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
        self.frames = []
        allprefixes = []
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f[-3::] == "jpg":
                    fileprefix = "%s/%s"%(root, f[0:-4])
                    allprefixes.append(fileprefix)
        allprefixes = sorted(allprefixes)
        N = 0
        for i, fileprefix in enumerate(allprefixes):
            if i < framerange[0] or i > framerange[1]:
                continue
            print("Loading video frame %i of %i"%(i+1, len(allprefixes)))
            self.frames.append(Pose(fileprefix))
            self.updateAssociations()
            if save_skeletons:
                plt.clf()
                if blurFace:
                    blurlastlast = blurlast
                    blurlast = self.frames[-1].render(blurFace=True, blurlast=blurlast+blurlastlast)
                else:
                    self.frames[-1].render(blurFace=False)
                #plt.savefig("%s/%s.png"%(skeletondir, fileprefix.split("/")[-1]))
                plt.savefig("%s/%i.png"%(skeletondir, i-framerange[0]))
            N += 1
        print("Loaded %i frames total"%N)
        if save_skeletons and delete_frames:
            # Save video
            subprocess.call(["ffmpeg", "-r", "5", "-i", "%s/%s.png"%(skeletondir, "%d"), "-r", "5", "-b", "5000k", "%s/skeleton.avi"%skeletondir])
            for i in range(N):
                os.remove("%s/%i.png"%(skeletondir, i))

    def updateAssociations(self, npreceding=30):
        """
        Given a sequence of pose objects, make sure the first skeleton is 
        more similar with the first skeleton in some preceding number of frames
        than other skeletons are to the first skeleton (swap if not the case)
        Parameters
        ----------
        npreceding: int
            The number of preceding frames to consider
        
        """
        N = min(npreceding, len(self.frames)-1)
        minDists = np.inf
        minIdx = 0
        for i in range(len(self.frames[-1].people)):
            distsi = 0.0
            if len(self.frames[-1].people) < 2:
                continue
            x = self.frames[-1].people[i]['pose_keypoints_2d']
            for j in range(N):
                if len(self.frames[-2-j].people) == 0:
                    continue
                y = self.frames[-2-j].people[0]['pose_keypoints_2d']
                idx = (np.sum(x, 1) > 0)*(np.sum(y, 1) > 0)
                dists = np.sqrt(np.sum((x[idx, :] - y[idx, :])**2, 1))
                distsi += np.mean(dists)
            if distsi < minDists:
                minDists = distsi
                minIdx = i
        if minIdx > 0:
            p = self.frames[-1].people[0]
            self.frames[-1].people[0] = self.frames[-1].people[minIdx]
            self.frames[-1].people[minIdx] = p

    def getKeypointStack(self, kstr):
        """
        For this video with N frames and K keypoints of a certain type, 
        return an Nx2K array with all keypoint coordinates stacked up
        for the first skeleton
        Parameters
        ----------
        kstr: string
            String of the type of keypoints
        
        Returns
        -------
        X: ndarray(N, 2K)
            An array of all keypoint coordinates
        M: ndarray(N, 2K)
            A binary mask indicating where them coordinates were detected
        """
        N = len(self.frames)
        K = self.frames[-1].people[0][kstr].shape[0]
        X = np.zeros((N, 2*K))
        M = np.zeros_like(X)
        for i in range(N):
            x = self.frames[i].people[0][kstr]
            xy = x[:, 0:2]
            mask = np.zeros_like(xy)
            mask[x[:, 2] > 0, :] = 1
            X[i, :] = xy.flatten()
            M[i, :] = mask.flatten()
        return X, M

    def interpolateMissingKeypoints(self):
        """
        Linearly interpolate missing keypoints on the first skeleton
        """
        N = len(self.frames)
        idxs = np.arange(N)
        for kstr in ['pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            K = self.frames[-1].people[0][kstr].shape[0]
            X, M = self.getKeypointStack(kstr)
            # Make the NaN regions some average constant value for plotting contrast
            X[M == 0] = np.mean(X) 
            for k in range(X.shape[1]):
                idxs1 = idxs[M[:, k] == 0]
                idxs2 = idxs[M[:, k] == 1]
                if len(idxs1) > 0 and len(idxs2) > 0:
                    res = np.interp(idxs1, idxs2, X[idxs2, k])
                    X[M[:, k] == 0, k] = res
            for i in range(N):
                x = np.reshape(X[i, :], (K, 2))
                self.frames[i].people[0][kstr][:, 0:2] = x



    def getNearestFrame(self, timestamp):
        """
        Return the video frame that's closest in time to a given timestamp
        Parameters
        ----------
        video: array(Pose)
            The video sequence
        timestamp: float
            The time
        
        Returns
        -------
        frame: Pose
            The nearest video frame in time
        """
        ts = np.array([f.timestamp for f in self.frames])
        idx = np.argmin(np.abs(ts-timestamp))
        return self.frames[idx]


########################################################################
##                Code for testing the loaded data                    ##
########################################################################



def testAccelerometerTDA():
    """
    Look at an example of using sliding window + TDA on 3-axis accelerometer data
    """
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

def makeAllSkeletonVideos():
    folders = glob.glob("AutismVideos/Study1/*/*")
    for i, f in enumerate(folders):
        print("Doing %i of %i"%(i+1, len(folders)))
        PoseVideo(f, save_skeletons=True)

def testVideoSkeletonTDA():
    """
    Look at an example of using sliding window + TDA on keypoints from OpenPose
    """
    winlen = 10
    Tau = 0.1
    dT = 0.1
    dim = int(winlen/Tau)
    keypt_types = ['pose_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']

    video = PoseVideo("URI-001-01-18-08", save_skeletons=False, delete_frames=False, framerange = (1000, 1200))
    video.interpolateMissingKeypoints()
    colors = cycle(['C0', 'C1', 'C2'])
    resol = 5
    plt.figure(figsize=(resol*5, resol*3))
    for i, v in enumerate(video.frames[0:-winlen]):
        p = v.people[0]
        plt.clf()
        I = np.array(v.I)

        ## Step 1: Plot detected keypoints
        plt.subplot2grid((3, 5), (0, 0), rowspan=2, colspan=2)
        plt.imshow(I)
        for k, (kstr, color) in enumerate(zip(keypt_types, colors)):
            keypts = p[kstr]
            # Plot detected keypoints as dots
            toplot = keypts[keypts[:, 2] > 0, :]
            plt.scatter(toplot[:, 0], toplot[:, 1], 4, color)
            # Plot interpolated keypoints as xs
            toplot = keypts[keypts[:, 2] == 0, :]
            plt.scatter(toplot[:, 0], toplot[:, 1], 8, color, marker="x")
        plt.axis('off')
        plt.xlim([0, I.shape[1]])
        plt.ylim([I.shape[0], 0])

        # Step 2: Plot sliding windows, SSMs, and H1
        for k, kstr in enumerate(keypt_types):
            title = kstr.split("_keypoints")[0]
            X, M = video.getKeypointStack(kstr)
            X = X[1::, :] - X[0:-1, :]
            Y = getSlidingWindowVideo(X[i:i+winlen*2], dim, Tau, dT)
            Y -= np.mean(Y, 0)[None, :]
            YNorm = np.sqrt(np.sum(Y**2, 1))
            YNorm[YNorm == 0] = 1
            Y /= YNorm[:, None]
            D = getCSM(Y, Y)
            dgm = ripser(D, maxdim=1, distance_matrix=True, coeff=41)['dgms'][1]

            plt.subplot(3, 5, k+3)
            plt.imshow(X, aspect='auto', cmap='magma_r', interpolation='none')
            plt.plot([0, X.shape[1]], [i, i], c='g')
            plt.plot([0, X.shape[1]], [i+winlen, i+winlen], c='g')
            plt.title("%s Coords"%title)

            plt.subplot(3, 5, 5+k+3)
            plt.imshow(D, interpolation='none')
            plt.title("%s SSM"%title)

            plt.subplot(3, 5, 10+k+3)
            mp = 0.0
            if dgm.size > 0:
                plot_dgms(diagrams=[dgm], labels=['H1'], size=50, xy_range = [0, 2, 0, 2], show=False)
                mp = np.max(dgm[:, 1] - dgm[:, 0])
            plt.xlim([0, 2])
            plt.ylim([0, 2])
            plt.title("%s Max Pers = %.3g"%(title, mp))

        plt.savefig("%i.png"%i, bbox_inches='tight')


if __name__ == '__main__':
    #testAccelerometerTDA()
    testVideoSkeletonTDA()
    #makeAllSkeletonVideos()