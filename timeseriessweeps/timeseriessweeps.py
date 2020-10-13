from dataprepper import AliPrepper, DataPrepper, HapsPrepper, SFSPrepper
from runCmdAsJob import runCmdAsJobWithoutWaitingWithLog as runjob
import sys
import os
sys.path.insert(1, '/proj/dschridelab/timeSeriesSweeps')


def main():
    """Submits jobs for each type of simulation and parameter set.

    """

    # Testing vars
    # Time Series
    sampleSizePerStepTS = 20  # individuals in sample for each time interval in time series
    numSamplesTS = 2  # number of time points sampled in time series
    samplingIntervalTS = 100  # spacing between time points

    # 1 Sample at 1 One Time Point
    # size of population sampled at one time point so that it is same size as time series data
    sampleSizePerStep1Samp = 40
    numSamples1Samp = 1  # number of time points sampled
    samplingInterval1Samp = 200  # spacing between time points
    # End Testing vars

    baseDir = /proj/dschridelab/timeSeriesSweeps
    maxSnps = 200

    # Should condense this to just accept the args from main runner
    stepToInputFormat = {'a': 'ali',
                         'b': 'sfs',
                         'c': 'haps',
                         'd': 'jsfs'}

    sampleSizesPerTS = {'a': [sampleSizePerStepTS, sampleSizePerStep1Samp],
                        'b': [sampleSizePerStepTS, sampleSizePerStep1Samp],
                        'c': [sampleSizePerStepTS, sampleSizePerStep1Samp],
                        'd': [sampleSizePerStepTS, sampleSizePerStep1Samp]}

    for suffix in ["", "1Samp"]:  # Is this necessary?
        inDir = baseDir + "/combinedSims" + suffix
        outDir = baseDir + "/npzs" + suffix
        logDir = baseDir + "/npzLogs" + suffix

        for itdir in [outDir, logDir]:
            if not os.path.exists(os.path.join(baseDir, npzLogs)):
                os.mkdir(os.path.join(baseDir, itdir))

        # Init prepper objects
        ali_prepper = AliPrepper(inDir, maxSnps, sampleSizePerTimeStep, out)
        sfs_prepper = SFSPrepper(inDir, maxSnps, sampleSizePerTimeStep, out)
        haps_prepper = HapsPrepper(inDir, maxSnps, sampleSizePerTimeStep, out)
       
        hard_soft_neut_ttv_{}.npz .format(step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[step][i], outDir, stepToInputFormat[step])

        runjob(cmd, "format", "format.txt",
               "12:00:00", "general", "64GB",
               logDir+"/hard_soft_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))


if __name__ == "__main__":
    main()
