import numpy as np
import os
import matplotlib.pyplot as plt
from initializeVar import *


baseSimDir = baseDir + '/simLogs'

outDir = baseDir + '/FracFixed_OLDSIM'

os.system("mkdir -p {}".format(outDir))


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

FracFixed = {}

for i in FreqsToPlot:
    TotalNum = len(FreqsToPlot[i])
    FixedNum = 0
    for j in FreqsToPlot[i]:
        if j >= 0.98:
            FixedNum += 1
        else:
            continue
    FracFixed[i] = FixedNum/TotalNum
            

fileName = "{}/fractionThatReachedFixation.txt".format(outDir)
file = open(fileName, 'w')
FracFixed = str(FracFixed)
file.write(FracFixed)
file.close()
