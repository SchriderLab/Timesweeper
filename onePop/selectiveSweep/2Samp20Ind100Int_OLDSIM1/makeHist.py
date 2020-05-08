import numpy as np
import os
import matplotlib.pyplot as plt
from initializeVar import *


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
