import os
import sys

import timeseriessweeps.plotting_utils as pu
import timeseriessweeps.utils as ut
from timeseriessweeps.dataprepper import (AliPrepper, HapsPrepper, JSFSPrepper,
                                          SFSPrepper)
from timeseriessweeps.initializeVar import *

sys.path.insert(1, '/pine/scr/e/m/emae/timeSeriesSweeps')

def main():
    """
    Submits jobs for each type of simulation and parameter set.
    TODO Update docs

    """

    # Testing vars #############################################################
    #TODO set these as argparse args OR as file naming in slimfiles
    # Time Series
    sampleSizePerStepTS = 20  # individuals in sample for each time interval in time series
    numSamplesTS = 2  # number of time points sampled in time series
    samplingIntervalTS = 100  # spacing between time points

    # 1 Sample at 1 One Time Point
    # size of population sampled at one time point so that it is same size as time series data
    sampleSizePerStep1Samp = 40
    numSamples1Samp = 1  # number of time points sampled
    samplingInterval1Samp = 200  # spacing between time points
    # End Testing vars #########################################################

    #TODO Also these in argparse
    baseDir = '/proj/dschridelab/timeSeriesSweeps'
    maxSnps = 200


    """Example of what was happening before

    stepToInputFormat = {'a':'ali', 'b':'sfs', 'c':'haps'}
    sampleSizesPerTS = {'a':["20 20", "200 200"], 'b':[
        "20 20", "200 200"], 'c':["20 20", "200 200"]}
    # stepToInputFormat = {'a':'ali'}

    suffices = ["", "1Samp"]
    for i in range(len(suffices)):
        suffixes[i] = suffices[i]
        inDir = baseDir + "/combinedSims" + suffixes[i]
        outDir = baseDir + "/npzs" + suffixes[i]
        logDir = baseDir + "/npzLogs" + suffixes[i]
        os.system("mkdir -p {} {}".format(outDir, logDir))

        for step in stepToInputFormat:
            cmd = "python ../02{}_formatNpz_{}.py {} {} {} {}/hard_v_neut_ttv_{}.npz".format(
                step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[step][i], outDir, stepToInputFormat[step])
            runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(
                cmd, "format", "format.txt", "12:00:00", "general", "64GB", logDir+"/hard_v_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))
    """

    sampleSizesPerTS = ["20 20", "200 200"]

    suffixes = ["", "1Samp"]
    for i in range(len(suffixes[i])):
        inDir = baseDir + "/combinedSims" + suffixes[i]
        outDir = baseDir + "/npzs" + suffixes[i]
        logDir = baseDir + "/npzLogs" + suffixes[i]

        for itdir in [outDir, logDir]:
            if not os.path.exists(os.path.join(baseDir, npzLogs)):
                os.mkdir(os.path.join(baseDir, itdir))

        # Init prepper objects, these handle all data preprocessing
        ali_prepper =   AliPrepper(inDir, maxSnps, 'hard_v_neut_ttv_ali.npz')
        sfs_prepper =   SFSPrepper(inDir, maxSnps, 'hard_v_neut_ttv_sfs.npz')
        haps_prepper = HapsPrepper(inDir, maxSnps, 'hard_v_neut_ttv_hap.npz')
        jsfs_prepper = JSFSPrepper(inDir, maxSnps, 'hard_v_neut_ttv_jsfs.npz')

        for i in sampleSizesPerTS:
            hard_soft_neut_ttv_{}.npz .format(step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[i], outDir, stepToInputFormat[step])

        ut.runjob(cmd, 
               "format", 
               "format.txt",
               "12:00:00", 
               "general", 
               "64GB",
               logDir+"/hard_soft_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))

    prefixLs = ['hard_soft_neut_ttv_sfs', 'hard_soft_neut_ttv_haps']

    plotDir = baseDir + "/npzPlots" + simType
    os.mkdir(plotDir)

    for prefix in prefixLs:
        inFileName = "{}/npzs{}/{}.npz".format(baseDir, simType, prefix)
        plotFileName = "{}/npzPlots{}/{}.mean.pdf".format(baseDir, simType, prefix)
        data, titles = readTrainXFromNpz(inFileName)
        print(inFileName)
        print(data[0].shape, data[1].shape)
        makeHeatmap(data, prefix, titles, plotFileName, mean=True)


#launchsims

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
            cmd = "python ../../../runAndParseSlim.py sweep.slim {} {} {} {} {} {} {} {} {} {} {} | gzip > {}".format(sampleSizePerStepTS, numSamplesTS, samplingIntervalTS, sampleSizePerStep1Samp, numSamples1Samp, samplingInterval1Samp, repsPerBatch, physLen, timeSeries, simType, dumpFileName, outFileName)
            runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, simType+suffix, "{}{}.txt".format(simType, suffix), "12:00:00", "general", "2G", "{}/{}_{}.log".format(logDir, simType, i))

# Formatall


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

# traincnn


prefixLs = ['hard_soft_neut_ttv_ali', 'hard_soft_neut_ttv_haps', 'hard_soft_neut_ttv_sfs', 'hard_soft_neut_ttv_jsfs']

simTypeToScript = {"":"../../keras_CNN_loadNrun.py", "1Samp":"../../keras_DNN_loadNrun.py"}
for simType in ["", "1Samp"]:
    outDir="{}/classifiers{}".format(baseDir, simType)
    os.system("mkdir -p {}".format(outDir))

    for prefix in prefixLs:
        cmd = "python {0} -i {1}/npzs{2}/{3}.npz -c {4}/{3}.mod".format(simTypeToScript[simType], baseDir, simType, prefix, outDir)
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(cmd, "trainTS", "trainTS.txt", "12:00:00", "general", "32GB", outDir+"/{}.log".format(prefix))













if __name__ == "__main__":
    main()
