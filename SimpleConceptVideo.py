import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io as sio
from AutismData import *

def makeConceptVideo(annoidx = 0):
    NA = len(ACCEL_TYPES)
    #studyname = "URI-001-01-18-08"
    studyname = "URI-001-01-25-08"
    foldername = "neudata/data/Study1/%s"%studyname
    annofilename = "Annotator1Stereotypy.annotation.xml"
    annos = loadAnnotations("%s/%s"%(foldername, annofilename))
    ftemplate = foldername + "/MITes_%s_RawCorrectedData_%s.RAW_DATA.csv"
    annos = annos[1::]
    # Load in accelerometer data
    Xs = [loadAccelerometerData(ftemplate%(ACCEL_NUMS[i], ACCEL_TYPES[i])) for i in range(NA)]
    # Load in video data
    video = PoseVideo(studyname)
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
    plt.figure(figsize=(18, 6))
    gridsize = (3, 3)
    for i, a in enumerate(annos):
        plt.clf()
        scores = np.zeros(NA)
        dT = (a['stop'] - a['start'])
        frame = video.getNearestFrame(a['stop'])
        plt.subplot2grid(gridsize, (0, 0), rowspan=3, colspan=1)
        frame.render(showLandmarks=False, blurFace=False)
        if i*hop > padding and i*hop-padding < clipLen:
            plt.title("Annotated %s Action"%a["label"])
        for k in range(NA):
            x = getAccelerometerRange(Xs[k], a)[:, 1::]
            plt.subplot2grid(gridsize, (k, 1))
            plt.plot((a['start']-start+np.linspace(-dT/50, dT*1.02, len(x)))/1000.0, x)
            if k == 0:
                plt.legend(["X", "Y", "Z"], loc=(-0.4, 0.4))
            if k < 2:
                ax = plt.gca()
                ax.set_xticks([])
            else:
                plt.xlabel("Time (Sec)")
            plt.title("%s Accelerometer"%ACCEL_TYPES[k])
        plt.savefig("anno%i_%i.png"%(annoidx, i), bbox_inches='tight')

if __name__ == '__main__':
    makeConceptVideo(0)