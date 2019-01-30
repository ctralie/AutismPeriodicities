from AutismData import *
from GeometricScoring import *
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from RQA import *
from ripser import Rips
import pandas as pd
import time
import seaborn as sns
import itertools
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import tree

# Dictionary to convert behavioral labels into numbered indices for classification
LABELS = ['Flap-Rock', 'Rock', 'Flap', 'Normal']
KEYPT_TYPES = ['pose_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']

def getAllFeaturesStudy(studiesdir, csvname, seed=100):
    """
    Compute all of the features for all time intervals for all subjects in a study
    """
    np.random.seed(seed)
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
        for k, kstr in enumerate(KEYPT_TYPES):
            X, M, ts = video.getKeypointStack(kstr)
            XNew, tsnew = upsampleFeatureStack(X, M, ts, fac)
            XNew = getAccelerationFeatureStack(XNew, 1)
            XsVideo.append(XNew)
            tsvideo.append(tsnew)
        
        for i, a in enumerate(anno[1::]):
            print("Doing Annotation %i of %i"%(i, len(anno)))
            if not a['label'] in LABELS:
                continue
            features = {'label':[a['label']], 'subject':[j], 'study':[studies[j]]}

            ## Step 1: Get statistics for all accelerometers
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(XsAccel[k], a)[:, 1::]
                #x = smoothDataMedian(x, 3)
                res = getPersistencesBlock(x, dim, mean_center=False, sphere_normalize=True, birthcutoff=np.inf)
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                rqas = getRQAStats(B, dmin, vmin)
                features["Accel_%s_TDA"%ACCEL_TYPES[k]] = [res['P']]
                for rqstr in rqas.keys():
                    features["Accel_%s_RQA_%s"%(ACCEL_TYPES[k], rqstr)] = [rqas[rqstr]]
            
            ## Step 2: Get statistics for video keypoints
            for k, (kstr, tsk, XVk) in enumerate(zip(KEYPT_TYPES, tsvideo, XsVideo)):
                # Extract keypoints within time interval of this annotation
                tidxs = np.arange(tsk.size)
                i1 = tidxs[np.argmin(np.abs(a['start']-tsk))]
                i2 = tidxs[np.argmin(np.abs(a['stop']-tsk))]
                x = XVk[i1:i2, :]
                # Compute max persistence
                if k == 0:
                    print("%i frames"%x.shape[0])
                # If the video isn't skipping too much
                res = getPersistencesBlock(x, winlen*fac, mean_center=True, sphere_normalize=True, birthcutoff=np.inf)
                features["Video_%s_TDA"%kstr] = [res['P']]

                # Compute RQA features
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                for rqstr in rqas.keys():
                    features["Video_%s_RQA_%s"%(kstr, rqstr)] = [rqas[rqstr]]

            ## Step 3: Append features to dictionary
            AllFeatures = AllFeatures.append(pd.DataFrame.from_dict(features))
        fout = open(csvname, "w")
        AllFeatures.to_csv(fout)
        fout.close()

def doAccelKeypointsCorrelations(csvname):
    """
    Plot the correlation coefficient between TDA measures from the accelerometer and video
    """
    accels = ["Accel_%s_TDA"%s for s in ["Trunk", "Left-wrist", "Right-wrist"]]
    videos = ["Video_%s_TDA"%s for s in ['pose_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']]
    data = pd.read_csv(csvname)
    X1 = data[accels].values
    X2 = data[videos].values
    videos = ["pose", "hand_left", "hand_right"]
    accels = ["trunk", "left-wrist", "right-wrist"]
    D = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            D[i, j] = np.corrcoef(X1[:, i], X2[:, j])[0, 1]
    plt.matshow(D, cmap = 'Purples')
    ax = plt.gca()
    for i in range(3):
        for j in range(3):
            ax.text(j, i, "%.3g"%D[i, j], va='center', ha='center')
    plt.xticks(np.arange(3), videos)
    plt.yticks(np.arange(3), accels)
    plt.show()

def scatterTDAScores(csvname):
    accels = ["Accel_%s_TDA"%s for s in ["Trunk", "Left-wrist", "Right-wrist"]]
    videos = ["Video_%s_TDA"%s for s in ['pose_keypoints_2d','hand_left_keypoints_2d','hand_right_keypoints_2d']]
    data = pd.read_csv(csvname)
    X1 = data[accels].values
    X2 = data[videos].values
    labels_str = data['label'].values
    label2index = {}
    index2label = {}
    for label in labels_str:
        if not label in label2index:
            label2index[label] = len(label2index)
            index2label[len(index2label)] = label
    labels = np.array([label2index[label] for label in labels_str])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    idx = np.random.permutation(labels.size)[0:1000]
    X = X2[idx, :]
    labels = labels[idx]
    for i in range(len(index2label)):
        ax.scatter(X[labels==i, 0], X[labels==i, 1], X[labels==i, 2])
    plt.legend([index2label[i] for i in range(len(index2label))])
    ax.set_xlabel("Pose")
    ax.set_ylabel("Left Hand")
    ax.set_zlabel("Right Hand")
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '%.2g' if normalize else '%d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt%cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def doClassificationStudy(csvname, periodic = False, seed = 0, n_splits=4):
    """
    Perform a bunch of cross-validated classification experiments with
    the features and labels
    Parameters
    ----------
    cvsname: string
        Path to the comma separated value file with all of the features
        and labels in it
    periodic: boolean
        If true, do a classification between periodic and not periodic only
        If false, do a classification between all 4 classes
    seed: int
        Seed for k fold splits
    n_splits: int
        k in the k-fold validation
    """
    data = pd.read_csv(csvname)
    ystrs = data['label'].values
    label2idx = {s:i for i, s in enumerate(LABELS)}
    yall = []
    yperiodic = []
    for ystr in ystrs:
        yall.append(label2idx[ystr])
        if ystr == 'Normal':
            yperiodic.append(0)
        else:
            yperiodic.append(1)
    yall = np.array(yall)
    yperiodic = np.array(yperiodic)

    ## Step 1: Print out some statistics on the dataset
    bysubject = data[['subject', 'label']].values
    countsbysubj = {}
    countsbyclass = {}
    for i in range(bysubject.shape[0]):
        subj, label = bysubject[i, :]
        if not subj in countsbysubj:
            countsbysubj[subj] = {}
        if not label in countsbysubj[subj]:
            countsbysubj[subj][label] = 0
        if not label in countsbyclass:
            countsbyclass[label] = 0
        countsbysubj[subj][label] += 1
        countsbyclass[label] += 1
    print(countsbyclass)


    ## Step 2: Perform classification experiments
    res = 4
    for (clfmethod, clfstr) in [(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), "LogisticRegression"), \
                                (KNeighborsClassifier(n_neighbors=5), 'KNN5'), \
                                #(svm.SVC(gamma='scale'), 'SVM_RBF'), \
                                #(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), 'RandomForest'), \
                                (tree.DecisionTreeClassifier(), "DecisionTree")]:
        print(clfstr)
        plt.figure(figsize=(res*4, res*3))
        for i, accelvideo in enumerate(["Accel", "Video", ""]):
            for j, rqatda in enumerate(["RQA", "TDA", "Lmax", ""]):
                keys = [d for d in data.keys() if (accelvideo in d and rqatda in d)]
                keys = [k for k in keys if ("Accel" in k or "Video" in k)]
                avname = accelvideo
                rqaname = rqatda
                if len(accelvideo) == 0:
                    avname = "Both"
                if len(rqatda) == 0:
                    rqaname = "Both"
                testname = "%s_%s"%(avname, rqaname)
                X = np.array(data[keys].values, dtype=float)
                kf = KFold(n_splits=n_splits, shuffle = True, random_state = seed)
                Total = 0
                CorrectAll = 0
                CorrectPeriodic = 0
                ConfAll = np.zeros((len(LABELS), len(LABELS)))
                ConfPeriodic = np.zeros((2, 2))
                for train_index, test_index in kf.split(X):
                    clf = make_pipeline(StandardScaler(), clfmethod)
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    Total += len(test_index)
                    if periodic:
                        yperiodic_train, yperiodic_test = yperiodic[train_index], yperiodic[test_index]
                        clf.fit(X_train, yperiodic_train)
                        y_pred = clf.predict(X_test)
                        CorrectPeriodic += np.sum(yperiodic_test == y_pred)
                        ConfPeriodic += confusion_matrix(yperiodic_test, y_pred)
                    else:
                        yall_train, yall_test = yall[train_index], yall[test_index]
                        clf.fit(X_train, yall_train)
                        y_pred = clf.predict(X_test)
                        CorrectAll += np.sum(yall_test == y_pred)
                        ConfAll += confusion_matrix(yall_test, y_pred)
                plt.subplot(3, 4, i*4+j+1)
                percentage = 0.0
                if periodic:
                    percentage = 100.0*CorrectPeriodic/Total
                    plot_confusion_matrix(ConfPeriodic, ['Normal', 'SMM'])
                else:
                    percentage = 100.0*CorrectAll/Total
                    plot_confusion_matrix(ConfAll, LABELS)
                outstr = "%s: %.3g %s"%(testname, percentage, "%")
                print(outstr)
                plt.title(outstr)
        plt.tight_layout()
        if periodic:
            plt.savefig("Periodic_%s_%s.svg"%(csvname, clfstr), bbox_inches='tight')
        else:
            plt.savefig("%s_%s.svg"%(csvname, clfstr), bbox_inches='tight')



def evenlySubsampleByClass(y, samples_per_class):
    classes = np.unique(y)
    idxs = np.array([])
    for c in classes:
        idxc = np.arange(len(y))[y == c]
        idxs = np.concatenate((idxs, np.random.permutation(idxc)[0:samples_per_class]))
    idxs = np.array(idxs, dtype=int)
    return idxs

def doClassificationStudyBySubject(csvname, periodic = False, seed = 0, samples_per_class=200):
    """
    Perform a bunch of cross-validated classification experiments with
    the features and labels, leaving one subject out and balancing the rest
    Parameters
    ----------
    cvsname: string
        Path to the comma separated value file with all of the features
        and labels in it
    periodic: boolean
        If true, do a classification between periodic and not periodic only
        If false, do a classification between all 4 classes
    seed: int
        Seed for evenmly subasmpmling
    samples_per_class: int
        The number of samples for each class for traning.  Total training samples is then
        4x this number
    """
    data = pd.read_csv(csvname)
    ystrs = data['label'].values
    label2idx = {s:i for i, s in enumerate(LABELS)}
    print(label2idx)
    yall = []
    yperiodic = []
    for ystr in ystrs:
        yall.append(label2idx[ystr])
        if ystr == 'Normal':
            yperiodic.append(0)
        else:
            yperiodic.append(1)
    yall = np.array(yall)
    yperiodic = np.array(yperiodic)
    subject = (data[['subject']].values).flatten()
    NSubjects = len(np.unique(subject))


    ## Step 2: Perform classification experiments
    res = 4
    for (clfmethod, clfstr) in [(LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'), "LogisticRegression"), \
                                (KNeighborsClassifier(n_neighbors=5), 'KNN5'), \
                                #(svm.SVC(gamma='scale'), 'SVM_RBF'), \
                                #(RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0), 'RandomForest'), \
                                (tree.DecisionTreeClassifier(), "DecisionTree")]:
        print(clfstr)
        plt.figure(figsize=(res*4, res*3))
        for subj in range(NSubjects):
            plt.clf()
            test_index = np.arange(yall.size)[yall == subj]
            idxother = np.arange(yall.size)[(yall==subj)==0]
            idxresample = evenlySubsampleByClass(yall[idxother], samples_per_class)
            train_index = idxother[idxresample]
            counts = np.zeros(4)
            print(np.unique(yall[train_index]))
            for c in np.unique(yall[train_index]):
                counts[c] = np.sum(yall[train_index] == c)
            print(counts)
            return
            for i, accelvideo in enumerate(["Accel", "Video", ""]):
                for j, rqatda in enumerate(["RQA", "TDA", "Lmax", ""]):
                    keys = [d for d in data.keys() if (accelvideo in d and rqatda in d)]
                    keys = [k for k in keys if ("Accel" in k or "Video" in k)]
                    avname = accelvideo
                    rqaname = rqatda
                    if len(accelvideo) == 0:
                        avname = "Both"
                    if len(rqatda) == 0:
                        rqaname = "Both"
                    testname = "%s_%s"%(avname, rqaname)
                    X = np.array(data[keys].values, dtype=float)
                    ConfAll = np.zeros((len(LABELS), len(LABELS)))
                    ConfPeriodic = np.zeros((2, 2))

                    clf = make_pipeline(StandardScaler(), clfmethod)
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    Total = X_test.shape[0]
                    if periodic:
                        yperiodic_train, yperiodic_test = yperiodic[train_index], yperiodic[test_index]
                        clf.fit(X_train, yperiodic_train)
                        y_pred = clf.predict(X_test)
                        CorrectPeriodic = np.sum(yperiodic_test == y_pred)
                        ConfPeriodic = confusion_matrix(yperiodic_test, y_pred)
                    else:
                        yall_train, yall_test = yall[train_index], yall[test_index]
                        clf.fit(X_train, yall_train)
                        y_pred = clf.predict(X_test)
                        CorrectAll = np.sum(yall_test == y_pred)
                        ConfAll = confusion_matrix(yall_test, y_pred)
                    plt.subplot(3, 4, i*4+j+1)
                    percentage = 0.0
                    if periodic:
                        percentage = 100.0*CorrectPeriodic/Total
                        plot_confusion_matrix(ConfPeriodic, ['Normal', 'SMM'])
                    else:
                        percentage = 100.0*CorrectAll/Total
                        plot_confusion_matrix(ConfAll, LABELS)
                    outstr = "%s: %.3g %s"%(testname, percentage, "%")
                    print(outstr)
                    plt.title(outstr)
            plt.tight_layout()
            if periodic:
                plt.savefig("%i_Periodic_%s_%s.svg"%(subj, csvname, clfstr), bbox_inches='tight')
            else:
                plt.savefig("%i_%s_%s.svg"%(subj, csvname, clfstr), bbox_inches='tight')



if __name__ == '__main__':
    #getAllFeaturesStudy("neudata/data/Study1/", "study1_all_centroid.csv")
    for periodic in [False, True]:
        doClassificationStudyBySubject("study1_all.csv", periodic=periodic, seed=10, samples_per_class=200)

    #scatterTDAScores("study1_all.csv")
    #doAccelKeypointsCorrelations("study1_all_centroid.csv")