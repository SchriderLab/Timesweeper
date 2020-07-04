import numpy as np
import os
import matplotlib.pyplot as plt
from initializeVar import *


baseSimDir = baseDir + '/simLogs'

histDir = baseDir + '/histograms'

outDir = baseDir + '/FracFixed'

os.system("mkdir -p {}".format(histDir))
os.system("mkdir -p {}".format(outDir))


FreqsToPlot = {}
GensToPlot = {}
SimsThatFix = {}
SimEndTimesToPlot = {}
StartGensToPlot = {}
StartFreqsToPlot = {}

for simType in ["hard", "soft"]:
    simDir = baseSimDir + "/" + simType

    MasterCounter = 0

    finalFreqs = []
    finalGens = []
    fixNum = []
    begGens = []
    endTimes = []
    begFreqs = []

    for infile in os.listdir(simDir):

        with open(simDir+'/'+infile) as f:
            lines = f.readlines()

            freq = {}
            gen = {}
            genB = {}
            end = {}

            for line in lines:
                if 'starting rep' in line:
                    MasterCounter += 1
                    freq[MasterCounter] = []
                    gen[MasterCounter] = []
                    genB[MasterCounter] = []
                    end[MasterCounter] = []
                elif 'NO LONGER SEGREGATING at generation' in line:
                    genNumAndEndT = line.strip('NO LONGER SEGREGATING at generation ; mut was FIXED \n')
                    genNumAndEndT = genNumAndEndT.split('simEndTime is')
                    genNum = int(genNumAndEndT[0])
                    simEndT = int(genNumAndEndT[1])
                    gen[MasterCounter].append(genNum)
                    end[MasterCounter].append(simEndT)
                    fixNum.append(MasterCounter)
                elif 'Sampling at generation' in line:
                    genB[MasterCounter].append(line.strip('Sampling at generation \n'))
                else:
                    freq[MasterCounter].append(line.strip('SEGREGATING at \n'))


            for i in freq:
                f = freq[i]
                g = gen[i]
                gB = genB[i]
                e = end[i]
                f.reverse()
                if len(f) > 0:
                    finalFreqs.append(float(f[0]))
                    begFreqs.append(float(f[-1]))
                elif len(f) == 0:
                    finalFreqs.append(1)
                    begFreqs.append(1)
                if len(g) > 0:
                    finalGens.append(int(g[0]))
                if len(gB) > 0:
                    begGens.append(int(gB[0]))
                if len(e) > 0:
                    endTimes.append(int(e[0]))
                #place = len(f)-1
                #finalGens.append(int(g[place]))
                
    fixNum = np.unique(fixNum)
    FreqsToPlot[simType] = finalFreqs
    GensToPlot[simType] = finalGens
    StartGensToPlot[simType] = begGens
    SimEndTimesToPlot[simType] = endTimes
    SimsThatFix[simType] = fixNum
    StartFreqsToPlot[simType] = begFreqs

FinalGensToPlot = {}
FinalSimEndTimesToPlot = {}
FinalSimsThatFix = {}

for simType in ["hard", "soft"]:
    FinalGensToPlot[simType] = []
    FinalSimEndTimesToPlot[simType] = []
    FinalSimsThatFix[simType] = []
    
    for counter, i in enumerate(GensToPlot[simType]):
        if i >= ((SimEndTimesToPlot[simType][counter]) - 50):
            FinalGensToPlot[simType].append(i)
            FinalSimEndTimesToPlot[simType].append(SimEndTimesToPlot[simType][counter])
            FinalSimsThatFix[simType].append(SimsThatFix[simType][counter])


FinalFreqsToPlot = {}
FinalStartGensToPlot = {}
FinalStartFreqsToPlot = {}

for simType in ["hard", "soft"]:
    FinalFreqsToPlot[simType] = []
    FinalStartGensToPlot[simType] = []
    FinalStartFreqsToPlot[simType] = []
    for i in FinalSimsThatFix[simType]:
        FinalFreqsToPlot[simType].append(FreqsToPlot[simType][i-1])
        FinalStartGensToPlot[simType].append(StartGensToPlot[simType][i-1])
        FinalStartFreqsToPlot[simType].append(StartFreqsToPlot[simType][i-1])

FracFixed = {}

for i in FreqsToPlot:
    TotalNum = len(FreqsToPlot[i])
    FixedNum = len(FinalGensToPlot[i])
    FracFixed[i] = FixedNum/TotalNum
    
#Old method for finding fraction of fixations and the associated generations
'''
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
'''    

plotFileNameFreq = histDir + '/FreqHist.png'
plotFileNameGen = histDir + '/GenHist.png'
plotFileNameStartGen = histDir + '/GenStartHist.png'
plotFileNameSimEnd = histDir + '/SimEndTimes.png'
plotFileNameStartFreqs = histDir + '/StartFreqs.png'

fig1, axs1 = plt.subplots(1,2, sharex = True, sharey = True)
fig1.suptitle('Final Frequencies for Hard and Soft Sweeps')
axs1[0].hist(FreqsToPlot['hard'])
axs1[0].set_title('Hard Sweep')
axs1[1].hist(FreqsToPlot['soft'])
axs1[1].set_title('Soft Sweep')
fig1.savefig(plotFileNameFreq)

fig2, axs2 = plt.subplots(2,1, sharex = True, sharey = True)
fig2.suptitle('Time of Fixation for Hard and Soft Sweeps (Generation)')
axs2[0].hist(FinalGensToPlot['hard'])
axs2[0].set_title('Hard Sweep')
axs2[1].hist(FinalGensToPlot['soft'])
axs2[1].set_title('Soft Sweep')
plt.xticks(rotation=30, ha='right')
fig2.savefig(plotFileNameGen)

fig3, axs3 = plt.subplots(2,1, sharex = True, sharey = True)
fig3.suptitle('Start Times for Sims That Fix (Generation)')
axs3[0].hist(FinalStartGensToPlot['hard'])
axs3[0].set_title('Hard Sweep')
axs3[1].hist(FinalStartGensToPlot['soft'])
axs3[1].set_title('Soft Sweep')
plt.xticks(rotation=30, ha='right')
fig3.savefig(plotFileNameStartGen)

fig4, axs4 = plt.subplots(2,1, sharex = True, sharey = True)
fig4.suptitle('Sim End Times (Generation)')
axs4[0].hist(FinalSimEndTimesToPlot['hard'])
axs4[0].set_title('Hard Sweep')
axs4[1].hist(FinalSimEndTimesToPlot['soft'])
axs4[1].set_title('Soft Sweep')
plt.xticks(rotation=30, ha='right')
fig4.savefig(plotFileNameSimEnd)

fig5, axs5 = plt.subplots()
axs5.hist(FinalStartFreqsToPlot['soft'])
axs5.set_title('Initial Frequencies for Soft Sweeps')
axs5.set_xlabel("Frequency")
axs5.set_ylabel("Counts")
fig5.savefig(plotFileNameStartFreqs)

fileName = "{}/fractionThatReachedFixation.txt".format(outDir)
file = open(fileName, 'w')
FracFixed = str(FracFixed)
file.write(FracFixed)
file.close()
