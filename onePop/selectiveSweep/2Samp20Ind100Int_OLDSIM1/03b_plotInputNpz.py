import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
from initializeVar import *
sys.path.insert(1, '/pine/scr/e/m/emae/timeSeriesSweeps')
import runCmdAsJob

def makeHeatmap(data, plotTitle, axTitles, plotFileName, mean=True):
    plt.figure()
    fig, axes = plt.subplots(1, 2)

    if mean:
        data[0] = getMeanMatrix(data[0])
        data[1] = getMeanMatrix(data[1])
        #data[2] = getMeanMatrix(data[2])
    else:
        raise NotImplementedError
    print("hard")
    print(data[0])
    print("soft")
    print(data[1])
    #print("neut")
    #print(data[2])

    minMin = np.amin(data)
    maxMax = np.amax(data)

    for i in range(len(data)):
        heatmap = axes[i].pcolor(data[i], cmap=plt.cm.Blues, norm=matplotlib.colors.LogNorm(vmin=minMin, vmax=maxMax))
        cbar = plt.colorbar(heatmap, cmap=plt.cm.Blues, ax=axes[i])
        axes[i].set_title(axTitles[i], fontsize=14)
        #cbar.set_label(cbarTitle, rotation=270, labelpad=20, fontsize=10.5)
    
        #axes[i].set_yticks(np.arange(data[i].shape[1]) + 0.5, minor=False)
        #axes[i].set_yticklabels(np.arange(data[i].shape[1]))
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].set_xlabel("Timepoint")
        axes[i].set_ylabel("Frequency")

    fig.set_size_inches(5, 3)
    plt.suptitle(plotTitle, fontsize=20, y=1.08)
    plt.tight_layout()
    plt.savefig(plotFileName, bbox_inches="tight")

def readTrainXFromNpz(inFileName):
    u = np.load(inFileName)
    trainX, testX, valX = u['trainX'], u['testX'], u['valX']
    print(trainX.shape)
    if "haps" in inFileName:
        trainX = trainX[:,:20]
    print(trainX.shape)
    trainy, testy, valy = u['trainy'], u['testy'], u['valy']
    one = trainy == 1
    zero = trainy == 0
    return [trainX[one], trainX[zero]], ["sweep", "neut"]

def getMeanMatrix(data):
    if len(data.shape) == 3:
        nMats, nRows, nCols = data.shape
    else:
        nMats, nRows = data.shape
        nCols = 1
        data = data.reshape(nMats, nRows, nCols)
    return(np.mean(data, axis=0))

#prefixLs = ['hard_v_neut_ttv_ali', 'hard_v_neut_ttv_haps', 'hard_v_neut_ttv_sfs']
prefixLs = ['hard_soft_neut_ttv_sfs', 'hard_soft_neut_ttv_haps']

for simType in ["", "1Samp"]:
    plotDir = baseDir + "/npzPlots" + simType
    os.system("mkdir -p {}".format(plotDir))

    for prefix in prefixLs:
        inFileName = "{}/npzs{}/{}.npz".format(baseDir, simType, prefix)
        plotFileName = "{}/npzPlots{}/{}.mean.pdf".format(baseDir, simType, prefix)
        data, titles = readTrainXFromNpz(inFileName)
        print(inFileName)
        print(data[0].shape, data[1].shape)
        #print(data[0].shape, data[1].shape, data[2].shape)
        makeHeatmap(data, prefix, titles, plotFileName, mean=True)
