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
        for infile in os.listdir(simLogs):
            with open(simLogs+'/'+infile) as f:
                lines = f.readlines()
                for line in lines:
                    if 'starting rep' in line:
                        MasterCounter += 1
                    elif 'NO LONGER SEGREGATING at generation' in line:
                        fixNum.append(MasterCounter)
                    elif 'Sampling at generation' in line:
                        continue
                    else:
                        continue
        fixNum = np.unique(fixNum)

        if len(fixNum) == 0:
            fixNum.append(0)

        fixNum = ','.join(map(str, fixNum))
        
        cmd = "python /pine/scr/e/m/emae/timeSeriesSweeps/combineMSFileDir.py {} {} no_shuffle | gzip > {}/{}.msOut.gz".format(simDir, fixNum, combinedSimDir, simType)
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "combine", "combine.txt", "12:00:00", "general", "32G", "{}/{}.log".format(logDir, simType))
