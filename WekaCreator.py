from AutismData import *
import glob
import matplotlib.pyplot as plt

def writeWekaHeader(fout, studies):
    fout.write("@RELATION Persistences\n")
    for i in range(len(ACCEL_TYPES)):
        fout.write("@ATTRIBUTE %s real\n"%ACCEL_TYPES[i])
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
        print "Doing ", foldername
        annofilename = "Annotator1Stereotypy.annotation.xml"
        anno = loadAnnotations("%s/%s"%(foldername, annofilename))
        nanno = getNormalAnnotations(anno[1::])
        anno = expandAnnotations(anno[1::])
        nanno = expandAnnotations(nanno)
        #Keep the annotations balanced by subsapling the negative regions
        if len(nanno) > len(anno) / 3:
            print "Subsampling %i negative annotations"%len(nanno)
            nanno = [nanno[k] for k in np.random.permutation(len(nanno))[0:len(anno)/3]]
        print "There are %i annotations and %i negative annotations"%(len(anno), len(nanno))
        anno = anno + nanno
        ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
        plt.clf()
        visualizeLabels(anno, doLegend = True)
        plt.savefig("%s/annotations.svg"%foldername, bbox_inches='tight')
        Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]
        for i in range(1, len(anno)):
            a = anno[i]
            if not a['label'] in ['Flap-Rock', 'Rock', 'Flap', 'Normal']:
                continue
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(Xs[k], a)
                (Pers, I) = getPersistencesBlock(x[:, 1::], dim, derivWin = derivWin)
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
