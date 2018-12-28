from AutismData import *
from GeometricScoring import *
import glob
import matplotlib.pyplot as plt
from RQA import *
from ripser import Rips
import pandas as pd

def doClassificationTests(studiesDir = "neudata/data/Study1/", seed=100):
    np.random.seed(seed)

    # Sliding window parameters
    dim = 30
    derivWin = -1

    # RQA Parameters
    dmin = 5
    vmin = 5
    Kappa = 0.2

    # Labels dictionary
    labelsDict = {s:i for i, s in enumerate(['Flap-Rock', 'Rock', 'Flap', 'Normal'])}

    folders = glob.glob(studiesDir+"*")
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
        plt.clf()
        visualizeLabels(anno, doLegend = True)
        plt.savefig("%s/annotations.svg"%foldername, bbox_inches='tight')
        Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]
        
        
        for i, a in enumerate(anno[1::]):
            print("Doing Annotation %i of %i"%(i, len(anno)))
            if not a['label'] in labelsDict:
                continue
            features = {'label':[a['label']], 'subject':[j], 'study':[studies[j]]}

            # Step 1: Get statistics for all accelerometers
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(Xs[k], a)[:, 1::]
                #x = smoothDataMedian(x, 3)
                res = getPersistencesBlock(x, dim, derivWin = derivWin)
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                rqas = getRQAStats(B, dmin, vmin)
                features["%s_TDA"%ACCEL_TYPES[k]] = [res['P']]
                for rqstr in rqas.keys():
                    features["%s_RQA_%s"%(ACCEL_TYPES[k], rqstr)] = [rqas[rqstr]]
            
            AllFeatures = AllFeatures.append(pd.DataFrame.from_dict(features))
            fout = open("AllFeatures.csv", "w")
            AllFeatures.to_csv(fout)
            fout.close()
        print(AllFeatures)

            


if __name__ == '__main__':
    doClassificationTests()