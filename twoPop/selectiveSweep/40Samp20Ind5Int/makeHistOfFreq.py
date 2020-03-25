import numpy as np
import os
import matplotlib.pyplot as plt
from initializeVar import *


baseSimDir = baseDir + '/simLogs'

histDir = baseDir + '/histogramOfFreqs'

os.system("mkdir -p {}".format(histDir))


FreqsToPlot = {}

for simType in ["hard", "soft"]:
    simDir = baseSimDir + "/" + simType

    finalFreqs = []
    
    #outDir = baseOutDir + '/simType'
    #os.system("mkdir -p {}".format(outDir))

    for infile in os.listdir(simDir):

        with open(simDir+'/'+infile) as f:
            lines = f.readlines()

            doc = {}    
            counter = 0
            for line in lines:
                if 'starting' in line:
                    counter += 1
                    doc[counter] = []
                else:
                    doc[counter].append(line.strip('SEGREGATING at \n'))


            for i in doc:
                l = doc[i]
                l.reverse()
                finalFreqs.append(float(l[0]))
                
    FreqsToPlot[simType] = finalFreqs

plotFileName = histDir + '/FreqHist.png'

fig, axs = plt.subplots(1,2, sharex = True, sharey = True)
fig.suptitle('Final Frequencies for Hard and Soft Sweeps')
axs[0].hist(FreqsToPlot['hard'])
axs[0].set_title('Hard Sweep')
axs[1].hist(FreqsToPlot['soft'])
axs[1].set_title('Soft Sweep')
fig.savefig(plotFileName)
