import os
import runCmdAsJob

baseDir="/pine/scr/d/s/dschride/data/popGenCnn/timeSeriesSweeps/twoPop/selectiveSweep"
baseOutDir=baseDir+"/sims"
baseDumpDir=baseDir+"/simDumps"
baseLogDir=baseDir+"/simLogs"
for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
    os.system("mkdir -p {}/{} {}/{} {}/{}".format(baseOutDir, name, baseLogDir, name, baseDumpDir, name))
physLen=100000

numBatches = 100
repsPerBatch = 100
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
            cmd = "python ../../runAndParseSlim.py sweep_twoPop.slim {} {} {} {} {} | gzip > {}".format(repsPerBatch, physLen, timeSeries, simType, dumpFileName, outFileName)
            runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, simType+suffix, "{}{}.txt".format(simType, suffix), "12:00:00", "general", "2G", "{}/{}_{}.log".format(logDir, simType, i))
