import os
import sys
import numpy as np
from initializeVar import *

sys.path.insert(1, '/pine/scr/e/m/emae/timeSeriesSweeps')

import runCmdAsJob


baseSimDir=baseDir+"/sims"
baseSimLogDir=baseDir+"/simLogs"

for timeSeries in [True,False]:
    if timeSeries:
        suffix = ""
    else:
        suffix = "1Samp"
    combinedSimDir=baseDir+"/combinedSims"+suffix
    logDir=baseDir+"/combinedSimLogs"+suffix
    os.system("mkdir -p {} {}".format(combinedSimDir, logDir))
    
    for simType in ["hard", "soft", "neut"]:
        simDir = baseSimDir + "/" + simType + suffix
        simLogs = baseSimLogDir + "/" + simType + suffix

        MasterCounter = 0

        fixNum = []
        finalGens = []
        endTimes = []
        
        for infile in os.listdir(simLogs):
            with open(simLogs+'/'+infile) as f:
                lines = f.readlines()
                gen = {}
                end = {}
                for line in lines:
                    if 'starting rep' in line:
                        MasterCounter += 1
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
                        continue
                    else:
                        continue

                for i in gen:
                    g = gen[i]
                    e = end[i]

                    if len(g) > 0:
                        finalGens.append(int(g[0]))
                    if len(e) > 0:
                        endTimes.append(int(e[0]))
                    
        fixNum = np.unique(fixNum)

        finalFixNum = []
        
        for counter, i in enumerate(finalGens):
            if i >= ((endTimes[counter]) - 50):
                finalFixNum.append(fixNum[counter])
        

        if len(finalFixNum) == 0:
            finalFixNum = [0]

        finalFixNum = ','.join(map(str, finalFixNum))
        
        cmd = "python /pine/scr/e/m/emae/timeSeriesSweeps/combineMSFileDir.py {} {} no_shuffle | gzip > {}/{}.msOut.gz".format(simDir, finalFixNum, combinedSimDir, simType)
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "combine", "combine.txt", "12:00:00", "general", "32G", "{}/{}.log".format(logDir, simType))
