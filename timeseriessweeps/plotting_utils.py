import os
import sys

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt

mpl.use("Agg")

import numpy as np

from timeseriessweeps.initializeVar import *
from timeseriessweeps.utils import run_batch_job

sys.path.insert(1, '/proj/dschridelab/timeSeriesSweeps')

"""TODO
- Functionalize all graph creation processes
- Get correct args working 
"""

def makeHist(baseDir):

    baseSimDir = baseDir + '/simLogs'
    histDir = baseDir + '/histograms'
    outDir = baseDir + '/FracFixed_OLDSIM'

    os.system("mkdir -p {}".format(histDir))
    os.system("mkdir -p {}".format(outDir))

    FreqsToPlot = {}
    GensToPlot = {}

    for simType in ["hard", "soft"]:
        simDir = baseSimDir + "/" + simType

        finalFreqs = []
        finalGens = []
        
        #outDir = baseOutDir + '/simType'
        #os.system("mkdir -p {}".format(outDir))

        for infile in os.listdir(simDir):

            with open(simDir+'/'+infile) as f:
                lines = f.readlines()

                freq = {}
                gen = {}
                counter = 0
                for line in lines:
                    if 'starting' in line:
                        counter += 1
                        freq[counter] = []
                        gen[counter] = []
                    elif 'SEGREGATING' in line:
                        freq[counter].append(line.strip('SEGREGATING at \n'))
                    else:
                        gen[counter].append(line.strip('Sampling at generation \n'))


                for i in freq:
                    f = freq[i]
                    g = gen[i]
                    f.reverse()
                    finalFreqs.append(float(f[0]))
                    place = len(f)-1
                    finalGens.append(int(g[place]))
                    
        FreqsToPlot[simType] = finalFreqs
        GensToPlot[simType] = finalGens


    FracFixed = {}
    GensToPlotFixed = {}

    for i in FreqsToPlot:
        TotalNum = len(FreqsToPlot[i])
        FixedNum = 0
        GenForSweepType = GensToPlot[i]
        GensOfFix = []
        for num, j in enumerate(FreqsToPlot[i]):
            if j >= 0.98:
                FixedNum += 1
                GensOfFix.append(GenForSweepType[num])
            else:
                continue
        FracFixed[i] = FixedNum/TotalNum
        GensToPlotFixed[i] = GensOfFix
        

    plotFileNameFreq = histDir + '/FreqHist.png'
    plotFileNameGen = histDir + '/GenHist.png'

    fig1, axs1 = plt.subplots(1,2, sharex = True, sharey = True)
    fig1.suptitle('Final Frequencies for Hard and Soft Sweeps')
    axs1[0].hist(FreqsToPlot['hard'])
    axs1[0].set_title('Hard Sweep')
    axs1[1].hist(FreqsToPlot['soft'])
    axs1[1].set_title('Soft Sweep')
    fig1.savefig(plotFileNameFreq)

    fig2, axs2 = plt.subplots(1,2, sharex = True, sharey = True)
    fig2.suptitle('Time of Fixation for Hard and Soft Sweeps (Generation)')
    axs2[0].hist(GensToPlotFixed['hard'])
    axs2[0].set_title('Hard Sweep')
    axs2[1].hist(GensToPlotFixed['soft'])
    axs2[1].set_title('Soft Sweep')
    fig2.savefig(plotFileNameGen)

    fileName = "{}/fractionThatReachedFixation.txt".format(outDir)
    file = open(fileName, 'w')
    FracFixed = str(FracFixed)
    file.write(FracFixed)
    file.close()

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
        heatmap = axes[i].pcolor(data[i], 
                                 cmap=plt.cm.Blues, 
                                 norm=matplotlib.colors.LogNorm(vmin=minMin, vmax=maxMax))
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

def getMeanMatrix(data):
    if len(data.shape) == 3:
        nMats, nRows, nCols = data.shape
    else:
        nMats, nRows = data.shape
        nCols = 1
        data = data.reshape(nMats, nRows, nCols)
    return(np.mean(data, axis=0))

#prefixLs = ['hard_v_neut_ttv_ali', 'hard_v_neut_ttv_haps', 'hard_v_neut_ttv_sfs']
