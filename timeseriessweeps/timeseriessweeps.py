import os
import sys

from dataprepper import (AliPrepper, DataPrepper, HapsPrepper, JSFSPrepper,
                         SFSPrepper)
import utils as ut

sys.path.insert(1, '/proj/dschridelab/timeSeriesSweeps')


def main():
    """Submits jobs for each type of simulation and parameter set.

    """

    # Testing vars #############################################################
    #TODO set these as argparse args
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
    baseDir = /proj/dschridelab/timeSeriesSweeps
    maxSnps = 200


"""Example of what was happening before

stepToInputFormat = {'a':'ali', 'b':'sfs', 'c':'haps'}
sampleSizesPerTS = {'a':["20 20", "200 200"], 'b':[
    "20 20", "200 200"], 'c':["20 20", "200 200"]}
# stepToInputFormat = {'a':'ali'}

suffices = ["", "1Samp"]
for i in range(len(suffices)):
    suffix = suffices[i]
    inDir = baseDir + "/combinedSims" + suffix
    outDir = baseDir + "/npzs" + suffix
    logDir = baseDir + "/npzLogs" + suffix
    os.system("mkdir -p {} {}".format(outDir, logDir))

    for step in stepToInputFormat:
        cmd = "python ../02{}_formatNpz_{}.py {} {} {} {}/hard_v_neut_ttv_{}.npz".format(
            step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[step][i], outDir, stepToInputFormat[step])
        runCmdAsJob.runCmdAsJobWithoutWaitingWithLog(
            cmd, "format", "format.txt", "12:00:00", "general", "64GB", logDir+"/hard_v_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))
"""

    sampleSizesPerTS = ["20 20", "200 200"]

    suffixes = ["", "1Samp"]
    for i in range(len(suffixes)):
        inDir = baseDir + "/combinedSims" + suffix
        outDir = baseDir + "/npzs" + suffix
        logDir = baseDir + "/npzLogs" + suffix

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


if __name__ == "__main__":
    main()
