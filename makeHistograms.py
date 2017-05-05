import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    fin = open("PeriodicityScores.arff")
    Classes = {}
    Periodic = {}
    started = False
    for l in fin.readlines():
        if l.rstrip() == "@DATA":
            started = True
            continue
        if not started:
            continue
        f = l.split(",")
        [trunk, lwrist, rwrist] = [float(f[0]), float(f[1]), float(f[2])]
        c = f[3]
        p = f[4]
        if not c in Classes:
            Classes[c] = []
        Classes[c].append([trunk, lwrist, rwrist])
        if not p in Periodic:
            Periodic[p] = []    
        Periodic[p].append([trunk, lwrist, rwrist])
    fin.close()
    for c in Classes:
        Classes[c] = np.array(Classes[c])
    for p in Periodic:
        Periodic[p] = np.array(Periodic[p])
    Names = ['Trunk', 'Left Wrist', 'Right Wrist']
    for i in range(len(Names)):
        plt.clf()
        num = 1
        for c in Classes:
            plt.subplot(2, 2, num)
            x = Classes[c][:, i]
            hist, bins = np.histogram(x)
            center = (bins[:-1] + bins[1:]) / 2
            width = 0.7*(bins[1] - bins[0])
            plt.bar(center, hist, width=width)
            num += 1
            plt.title("%s - %s"%(c, Names[i]))
        plt.savefig("%i.svg"%i, bbox_inches = 'tight')
