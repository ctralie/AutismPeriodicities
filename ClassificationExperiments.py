from AutismData import *
from GeometricScoring import *
import glob
import matplotlib.pyplot as plt
from RQA import *
from ripser import Rips
import pandas as pd

# Dictionary to convert behavioral labels into numbered indices for classification
LABELS_DICT = {s:i for i, s in enumerate(['Flap-Rock', 'Rock', 'Flap', 'Normal'])}


def getAllFeaturesStudy(studiesdir, csvname, seed=100):
    """
    Compute all of the features for all time intervals for all subjects in a study
    """
    np.random.seed(seed)
    # Sliding window parameters
    dim = 30
    derivWin = -1

    # RQA Parameters
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
        anno = expandAnnotations(anno[1::])
        nanno = expandAnnotations(nanno)
        #Keep the annotations balanced by subsampling the negative regions
        if len(nanno) > len(anno) / 3:
            print("Subsampling %i negative annotations"%len(nanno))
            nanno = [nanno[k] for k in np.random.permutation(len(nanno))[0:int(len(anno)/3)]]
        print("There are %i annotations and %i negative annotations"%(len(anno), len(nanno)))
        anno = anno + nanno
        ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
        XsAccel = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]
        
        for i, a in enumerate(anno[1::]):
            print("Doing Annotation %i of %i"%(i, len(anno)))
            if not a['label'] in LABELS_DICT:
                continue
            features = {'label':[a['label']], 'subject':[j], 'study':[studies[j]]}

            # Step 1: Get statistics for all accelerometers
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(XsAccel[k], a)[:, 1::]
                #x = smoothDataMedian(x, 3)
                res = getPersistencesBlock(x, dim, derivWin = derivWin)
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                rqas = getRQAStats(B, dmin, vmin)
                features["%s_TDA"%ACCEL_TYPES[k]] = [res['P']]
                for rqstr in rqas.keys():
                    features["%s_RQA_%s"%(ACCEL_TYPES[k], rqstr)] = [rqas[rqstr]]
            
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
    getAllFeaturesStudy("neudata/data/Study1/", "study1.csv")
