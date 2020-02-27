import os
import sys
from initializeVar import *

sys.path.insert(1, '/pine/scr/e/m/emae/timeSeriesSweeps')

import runCmdAsJob


baseOutDir=baseDir+"/sims"
baseDumpDir=baseDir+"/simDumps"
baseLogDir=baseDir+"/simLogs"
for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
    os.system("mkdir -p {}/{} {}/{} {}/{}".format(baseOutDir, name, baseLogDir, name, baseDumpDir, name))
physLen=100000


numBatches = 100
repsPerBatch=100
for timeSeries in [True,False]:
    for i in range(numBatches):
        if timeSeries:
            suffix = ""
        else:
            suffix = "1Samp"
        for simType in ["hard", "soft", "neut"]:
            outDir = baseOutDir + "/" + simType + suffix
            dumpDir = baseDumpDir + "/" + simType + suffix
            logDir = baseLogDir + "/" + simType + suffix
            outFileName = "{}/{}_{}.msOut.gz".format(outDir, simType, i)
            dumpFileName = "{}/{}_{}.trees.dump".format(dumpDir, simType, i)
            cmd = "python ../../runAndParseSlim.py ../sweep.slim {} {} {} {} {} {} {} {} {} {} {} | gzip > {}".format(sampleSizePerStepTS, numSamplesTS, samplingIntervalTS, sampleSizePerStep1Samp, numSamples1Samp, samplingInterval1Samp, repsPerBatch, physLen, timeSeries, simType, dumpFileName, outFileName)
            runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, simType+suffix, "{}{}.txt".format(simType, suffix), "12:00:00", "general", "2G", "{}/{}_{}.log".format(logDir, simType, i))
