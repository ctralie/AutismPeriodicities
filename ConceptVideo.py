import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from AutismData import *
from ripser import Rips

def makeConceptVideo(annoidx = 0, showPDs = False):
    rips = Rips(coeff=41)
    NA = len(ACCEL_TYPES)
    #studyname = "URI-001-01-18-08"
    studyname = "URI-001-01-25-08"
    foldername = "neudata/data/Study1/%s"%studyname
    annofilename = "Annotator1Stereotypy.annotation.xml"
    annos = loadAnnotations("%s/%s"%(foldername, annofilename))
    ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
    annos = annos[1::]
    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(NA)]
    video = getVideo(studyname)
    a = annos[annoidx]
    clipLen = a['stop'] - a['start']
    print("Clip is %.3g seconds long:"%(clipLen/1000.0))
    
    padding = 4000 #Amount of padding around annotation in milliseconds
    hop = 50
    dim = 30
    a['start'] -= padding
    a['stop'] += padding
    start = a['start']
    annos = expandAnnotations([a], hop=hop)
    if showPDs:
        plt.figure(figsize=(16, 7))
    else:
        plt.figure(figsize=(12, 7))
    gridsize = (3, 3)
    if showPDs:
        gridsize = (3, 4)
    for i, a in enumerate(annos):
        plt.clf()
        scores = np.zeros(NA)
        dT = (a['stop'] - a['start'])
        frame = getNearestVideoFrame(video, a['stop'])
        plt.subplot2grid(gridsize, (0, 0), rowspan=3, colspan=1)
        frame.render(showLandmarks=False)
        if i*hop > padding and i*hop-padding < clipLen:
            plt.title("Annotated %s Action"%a["label"])
        for k in range(NA):
            x = getAccelerometerRange(Xs[k], a)[:, 1::]
            plt.subplot2grid(gridsize, (k, 1))
            plt.plot((a['start']-start+np.linspace(0, dT, len(x)))/1000.0, x)
            if k == 0:
                plt.legend(["X", "Y", "Z"], loc=(-0.4, 0.4))
            if k < 2:
                ax = plt.gca()
                ax.set_xticks([])
            else:
                plt.xlabel("Time (Sec)")
            plt.title("%s Accelerometer"%ACCEL_TYPES[k])
            res = getPersistencesBlock(x, dim)
            [I, P] = [res['I'], res['P']]
            if showPDs:
                plt.subplot2grid(gridsize, (k, 2))
                if I.size > 0:
                    rips.plot(diagrams=[I], labels=['H1'], size=50, show=False)#, \
                                #xy_range = [0, 2, 0, 2])
                plt.title("%s Persistence Dgm"%ACCEL_TYPES[k])
            scores[k] = P/np.sqrt(3)
        if showPDs:
            plt.subplot2grid(gridsize, (0, 3), rowspan=3, colspan=1)
        else:
            plt.subplot2grid(gridsize, (0, 2), rowspan=3, colspan=1)
        plt.barh(-np.arange(NA), scores)
        ax = plt.gca()
        ax.set_yticks([])
        plt.xlim([0, 1])
        plt.xlabel("Score")
        plt.title("Periodicity Scores")
        plt.savefig("anno%i_%i.png"%(annoidx, i), bbox_inches='tight')

if __name__ == '__main__':
    #makeConceptVideo(4, showPDs = True)
    for i in range(20):
        for showPDs in [True, False]:
            makeConceptVideo(i, showPDs = showPDs)
            extra = ""
            if showPDs:
                extra = "withPDs"
            subprocess.call(["avconv", "-r", "20", "-i", "anno%i_%sd.png"%(i, "%"), "-r", "20", "-b", "10000k", "%i%s.avi"%(i, extra)])
            for f in glob.glob("*.png"):
                os.remove(f)
