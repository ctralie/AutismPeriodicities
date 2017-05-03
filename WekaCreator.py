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
    folders = glob.glob(studiesDir+"*")
    studies = [f.split("/")[-1] for f in folders]
    fout = open("Persistences.arff", "w")
    writeWekaHeader(fout, studies)
    for j in range(len(folders)):
        foldername = folders[j]
        print "Doing ", foldername
        annofilename = "Annotator1Stereotypy.annotation.xml"
        anno = loadAnnotations("%s/%s"%(foldername, annofilename))
        ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
        normal = getNormalAnnotations(anno[1::])[1::]
        print "%i SMM, %i Normal"%(len(anno), len(normal))
        anno = anno + normal
        plt.clf()
        visualizeLabels(anno, doLegend = True)
        plt.savefig("%s/annotations.svg"%foldername, bbox_inches='tight')
        Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(len(ACCEL_TYPES))]
        for i in range(1, len(anno)):
            a = anno[i]
            for k in range(len(ACCEL_TYPES)):
                x = getAccelerometerRange(Xs[k], a)
                (maxP, maxI, maxidx) = getMaxPersistenceBlock(x[:, 1::], 20, 150, dim)
                fout.write("%g,"%maxP)
            fout.write("%s,"%a['label'])
            if a['label'] == 'Normal':
                fout.write("NonPeriodic,")
            else:
                fout.write("Periodic,")
            fout.write(studies[j])
            fout.write("\n")
            fout.flush()
    fout.close()
                
