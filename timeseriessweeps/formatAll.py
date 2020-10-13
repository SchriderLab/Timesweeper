import os
import sys
from initializeVar import *

sys.path.insert(1, '/proj/dschridelab/timeSeriesSweeps')

import runCmdAsJob

maxSnps=200

stepToInputFormat = {'a':'ali', 'b':'sfs', 'c':'haps', 'd':'jsfs'}
sampleSizesPerTS = {'a':[sampleSizePerStepTS, sampleSizePerStep1Samp], 'b':[sampleSizePerStepTS, sampleSizePerStep1Samp], 'c':[sampleSizePerStepTS, sampleSizePerStep1Samp], 'd': [sampleSizePerStepTS, sampleSizePerStep1Samp]}
#stepToInputFormat = {'a':'ali'}

suffices = ["", "1Samp"]
for i in range(len(suffices)):
    suffix = suffices[i]
    inDir = baseDir + "/combinedSims" + suffix
    outDir = baseDir + "/npzs" + suffix
    logDir = baseDir + "/npzLogs" + suffix
    os.system("mkdir -p {} {}".format(outDir, logDir))

    for step in stepToInputFormat:
        cmd = "python ../../02{}_formatNpz_{}.py {} {} {} {}/hard_soft_neut_ttv_{}.npz".format(step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[step][i], outDir, stepToInputFormat[step])
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "format", "format.txt", "12:00:00", "general", "64GB", logDir+"/hard_soft_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))
