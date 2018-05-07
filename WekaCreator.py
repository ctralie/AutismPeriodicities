from AutismData import *
from GeometricScoring import *
import glob
import matplotlib.pyplot as plt
from RQA import *
from ripser import Rips

def writeWekaHeader(fout, studies):
    rqa = getRQAStats(np.random.randn(10, 10) > 0, 5, 5).keys()
    fout.write("@RELATION Persistences\n")
    for i in range(len(ACCEL_TYPES)):
        for r in rqa:
            fout.write("@ATTRIBUTE %s%s real\n"%(ACCEL_TYPES[i], r))
        fout.write("@ATTRIBUTE %sPers real\n"%ACCEL_TYPES[i])
    labels = ['Flap-Rock', 'Rock', 'Flap', 'Normal']
    fout.write("@ATTRIBUTE class {")
    labels = [l for l in labels]
    for i in range(len(labels)):
        fout.write(labels[i])
        if i < len(labels)-1:
            fout.write(",")
    fout.write("}\n")
    fout.write("@ATTRIBUTE periodic {Periodic, NonPeriodic}\n")
    if len(studies) > 0:
        fout.write("@ATTRIBUTE testname {")
        for i in range(len(studies)):
            fout.write(studies[i])
            if i < len(studies)-1:
                fout.write(",")
        fout.write("}\n")
    fout.write("@DATA\n")

if __name__ == '__main__':
    studiesDir = "neudata/data/Study1/"
    dim = 30
    Kappa = 0.2
    dmin = 5
    vmin = 5
    derivWin = -1
    folders = glob.glob(studiesDir+"*")
    studies = [f.split("/")[-1] for f in folders]
    fout = open("Persistences.arff", "w")
    writeWekaHeader(fout, studies)
    np.random.seed(100)
    for j in range(len(folders)):
        foldername = folders[j]
        thisfout = open("%s/Persistences.arff"%foldername, "w")
        writeWekaHeader(thisfout, [])
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
        for i in range(1, len(anno)):
            print("Doing Annotation %i of %i"%(i, len(anno)))
            a = anno[i]
            if not a['label'] in ['Flap-Rock', 'Rock', 'Flap', 'Normal']:
                continue
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(Xs[k], a)[:, 1::]
                #x = smoothDataMedian(x, 3)
                res = getPersistencesBlock(x, dim, derivWin = derivWin)
                B = CSMToBinaryMutual(getCSM(x, x), Kappa)
                rqas = getRQAStats(B, dmin, vmin)
                Pers = res['P']
                for rqstr in rqas.keys():
                    fout.write("%g,"%rqas[rqstr])
                    thisfout.write("%g,"%rqas[rqstr])
                fout.write("%g,"%Pers)
                thisfout.write("%g,"%Pers)
            fout.write("%s,"%a['label'])
            thisfout.write("%s,"%a['label'])
            if a['label'] == 'Normal':
                fout.write("NonPeriodic,")
                thisfout.write("NonPeriodic,")
            else:
                fout.write("Periodic,")
                thisfout.write("Periodic")
            fout.write(studies[j])
            fout.write("\n")
            fout.flush()
            thisfout.write("\n")
            thisfout.flush()
        thisfout.close()
    fout.close()
