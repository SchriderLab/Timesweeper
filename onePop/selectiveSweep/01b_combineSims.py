import os
import runCmdAsJob

baseDir="/pine/scr/e/m/emae/data/popGenCnn/timeSeriesSweeps/onePop/selectiveSweep"
baseSimDir=baseDir+"/sims"

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
        cmd = "python ~/klBase/msUtils/combineMSFileDir.py {} no_shuffle | gzip > {}/{}.msOut.gz".format(simDir, combinedSimDir, simType)
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "combine", "combine.txt", "12:00:00", "general", "32G", "{}/{}.log".format(logDir, simType))
