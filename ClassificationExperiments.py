from AutismData import *
from GeometricScoring import *
import glob
import matplotlib.pyplot as plt
from RQA import *
from ripser import Rips
import pandas as pd
import time

# Dictionary to convert behavioral labels into numbered indices for classification
LABELS_DICT = {s:i for i, s in enumerate(['Flap-Rock', 'Rock', 'Flap', 'Normal'])}


def getAllFeaturesStudy(studiesdir, csvname, seed=100):
    """
    Compute all of the features for all time intervals for all subjects in a study
    """
    np.random.seed(seed)
    keypt_types = ['pose_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']
    blocktime = 2500

    # Accelerometer sliding window parameters
    dim = 30
    derivWin = -1

    # Video sliding window parameters
    winlen = 5 #Assuming video is 4-5 fps on average, this is somewhere around 1 second
    fac = 10 # Interpolation factor
    
    # RQA Parameters for both accelerometer and video
    dmin = 5
    vmin = 5
    Kappa = 0.2

    folders = glob.glob(studiesdir+"*")
    studies = [f.split("/")[-1] for f in folders]

    AllFeatures = pd.DataFrame()

    for j, foldername in enumerate(folders):
        print("Doing %s"%foldername)
        annofilename = "Annotator1Stereotypy.annotation.xml"
        anno = loadAnnotations("%s/%s"%(foldername, annofilename))
        nanno = getNormalAnnotations(anno[1::])
        anno = expandAnnotations(anno[1::], time=blocktime)
        nanno = expandAnnotations(nanno, time=blocktime)
        #Keep the annotations balanced by subsampling the negative regions
        if len(nanno) > len(anno) / 3:
            print("Subsampling %i negative annotations"%len(nanno))
            nanno = [nanno[k] for k in np.random.permutation(len(nanno))[0:int(len(anno)/3)]]
        print("There are %i annotations and %i negative annotations"%(len(anno), len(nanno)))
        anno = anno + nanno
        ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
        XsAccel = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]
        
        video = PoseVideo(foldername.split("/")[-1], save_skeletons=False)
        XsVideo = []
        tsvideo = []
        for k, kstr in enumerate(keypt_types):
            X, M, ts = video.getKeypointStack(kstr)
            XNew, tsnew = upsampleFeatureStack(X, M, ts, fac)
            XNew = getAccelerationFeatureStack(XNew, 1)
            XsVideo.append(XNew)
            tsvideo.append(tsnew)
        
        for i, a in enumerate(anno[1::]):
            print("Doing Annotation %i of %i"%(i, len(anno)))
            if not a['label'] in LABELS_DICT:
                continue
            features = {'label':[a['label']], 'subject':[j], 'study':[studies[j]]}

            ## Step 1: Get statistics for all accelerometers
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(XsAccel[k], a)[:, 1::]
                #x = smoothDataMedian(x, 3)
                res = getPersistencesBlock(x, dim, derivWin = derivWin)
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                rqas = getRQAStats(B, dmin, vmin)
                features["Accel_%s_TDA"%ACCEL_TYPES[k]] = [res['P']]
                for rqstr in rqas.keys():
                    features["Accel_%s_RQA_%s"%(ACCEL_TYPES[k], rqstr)] = [rqas[rqstr]]
            
            ## Step 2: Get statistics for video keypoints
            for k, (kstr, tsk, XVk) in enumerate(zip(keypt_types, tsvideo, XsVideo)):
                # Extract keypoints within time interval of this annotation
                tidxs = np.arange(tsk.size)
                i1 = tidxs[np.argmin(np.abs(a['start']-tsk))]
                i2 = tidxs[np.argmin(np.abs(a['stop']-tsk))]
                X = XVk[i1:i2, :]

                # Compute max persistence
                maxpers = 0.0
                if X.shape[0] > winlen*fac:
                    if k == 0:
                        print("%i frames"%X.shape[0])
                    # If the video isn't skipping too much
                    Y = getSlidingWindowVideoInteger(X, winlen*fac)
                    Y -= np.mean(Y, 0)[None, :]
                    YNorm = np.sqrt(np.sum(Y**2, 1))
                    YNorm[YNorm == 0] = 1
                    Y /= YNorm[:, None]
                    D = getCSM(Y, Y)
                    dgm = ripser(D, maxdim=1, distance_matrix=True, coeff=41)['dgms'][1]
                    if dgm.size > 0:
                        maxpers = np.max(dgm[:, 1]-dgm[:, 0])
                else:
                    print("X.shape[0] = %i is too small for window length %i"%(X.shape[0], winlen*fac))
                features["Video_%s_TDA"%kstr] = maxpers

                # Compute RQA features
                B = CSMToBinaryMutual(getCSM(X, X), Kappa)
                for rqstr in rqas.keys():
                    features["Video_%s_RQA_%s"%(kstr, rqstr)] = [rqas[rqstr]]

            ## Step 3: Append features to dictionary
            AllFeatures = AllFeatures.append(pd.DataFrame.from_dict(features))
        fout = open(csvname, "w")
        AllFeatures.to_csv(fout)
        fout.close()

            
def doClassificationStudy(csvname):
    """
    Perform a bunch of cross-validated classification experiments with
    the features and labels
    """
    # Normalize data per column with standard deviations from test set (or look up other standard way to do this)
    """
    https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html#sphx-glr-auto-examples-preprocessing-plot-scaling-importance-py
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    
    
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
    
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
    """
    pass

if __name__ == '__main__':
    getAllFeaturesStudy("neudata/data/Study1/", "study1_all.csv")
