import os
import sys
import argparse

from timeseriessweeps import plotting_utils as pu
import utils as ut
from .initializeVar import *

# Testing vars #############################################################
#TODO set these as argparse args OR as file naming in slimfiles? JSON?
# Time Series
sampleSizePerStepTS = 20  # individuals in sample for each time interval in time series
numSamplesTS = 2  # number of time points sampled in time series
samplingIntervalTS = 100  # spacing between time points

# 1 Sample at 1 One Time Point
# size of population sampled at one time point so that it is same size as time series data
sampleSizePerStep1Samp = 40
numSamples1Samp = 1  # number of time points sampled
samplingInterval1Samp = 200  # spacing between time points

#TODO baseDir should be specified as arg call, rest should be functionalized off
baseDir = '/proj/dschridelab/timeSeriesSweeps'
maxSnps = 200

baseSimDir=baseDir+"/sims"
baseDumpDir=baseDir+"/simDumps"
baseLogDir=baseDir+"/simLogs"

# End Testing vars #########################################################


def launch_sims():
    for name in ["hard", "neut", "hard1Samp", "neut1Samp"]:
        os.system("mkdir -p {}/{} {}/{} {}/{}".format(baseSimDir, name, 
                                                      baseLogDir, name, 
                                                      baseDumpDir, name))
    physLen=100000

    numBatches = 100
    repsPerBatch=100
    for timeSeries in [True,False]:
        for i in range(numBatches):
            if timeSeries:
                suffix = ""
            else:
                suffix = "1Samp"
            for simType in ["hard","neut"]:
                outDir = baseSimDir + "/" + simType + suffix
                dumpDir = baseDumpDir + "/" + simType + suffix
                logDir = baseLogDir + "/" + simType + suffix
                outFileName = "{}/{}_{}.msOut.gz".format(outDir, simType, i)
                dumpFileName = "{}/{}_{}.trees.dump".format(dumpDir, simType, i)
                cmd = "python scripts/runAndParseSlim.py adaptiveIntrogressionTS.slim {} {} {} {} {} | gzip > {}".format(repsPerBatch, physLen, timeSeries, simType, dumpFileName, outFileName)
                ut.run_batch_job(cmd, simType+suffix, "{}{}.txt".format(simType, suffix), "12:00:00", "general", "2G", "{}/{}_{}.log".format(logDir, simType, i))

def combine_sims():
    for timeSeries in [True,False]:
        if timeSeries:
            suffix = ""
        else:
            suffix = "1Samp"
        combinedSimDir=baseDir+"/combinedSims"+suffix
        logDir=baseDir+"/combinedSimLogs"+suffix
        os.system("mkdir -p {} {}".format(combinedSimDir, logDir))
        for simType in ["hard","neut"]:
            simDir = baseSimDir + "/" + simType + suffix
            cmd = "python scripts/combineMSFileDir.py {} no_shuffle | gzip > {}/{}.msOut.gz".format(simDir, combinedSimDir, simType)
            ut.run_batch_job(cmd, "combine", "combine.txt", "12:00:00", "general", "32G", "{}/{}.log".format(logDir, simType))

def format_all():
    stepToInputFormat = {'a':'ali', 'b':'sfs', 'c':'haps'}
    sampleSizesPerTS = {'a':[20, 200], 'b':[20, 200], 'c':[20, 200]}
    #stepToInputFormat = {'a':'ali'}

    suffices = ["", "1Samp"]
    for i in range(len(suffices)):
        suffix = suffices[i]
        inDir = baseDir + "/combinedSims" + suffix
        outDir = baseDir + "/npzs" + suffix
        logDir = baseDir + "/npzLogs" + suffix
        os.system("mkdir -p {} {}".format(outDir, logDir))

        for step in stepToInputFormat:
            cmd = "python formatters/formatNpz_{}.py {} {} {} {}/hard_v_neut_ttv_{}.npz".format(step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS[step][i], outDir, stepToInputFormat[step])
            ut.run_batch_job(cmd, "format", "format.txt", "12:00:00", "general", "64GB", logDir+"/hard_v_neut_ttv_{}.npz.log".format(stepToInputFormat[step]))

def train_nets():
    #This is out for now, need to make new model
    prefixLs = ['hard_v_neut_ttv_ali', 'hard_v_neut_ttv_haps', 'hard_v_neut_ttv_sfs']
    simTypeToScript = {"":"../keras_CNN_loadNrun.py", "1Samp":"../keras_DNN_loadNrun.py"}
    for simType in ["", "1Samp"]:
        outDir="{}/classifiers{}".format(baseDir, simType)
        os.system("mkdir -p {}".format(outDir))

        for prefix in prefixLs:
            cmd = "python {0} -i {1}/npzs{2}/{3}.npz -c {4}/{3}.mod".format(simTypeToScript[simType], baseDir, simType, prefix, outDir)
            ut.run_batch_job(cmd, "trainTS", "trainTS.txt", "12:00:00", "general", "32GB", outDir+"/{}.log".format(prefix))

def plot_input_npz():
    #prefixLs = ['hard_v_neut_ttv_ali', 'hard_v_neut_ttv_haps', 'hard_v_neut_ttv_sfs']
    prefixLs = ['hard_v_neut_ttv_sfs', 'hard_v_neut_ttv_haps']
    for simType in ["", "1Samp"]:
        plotDir = baseDir + "/npzPlots" + simType
        os.system("mkdir -p {}".format(plotDir))

        for prefix in prefixLs:
            inFileName = "{}/npzs{}/{}.npz".format(baseDir, simType, prefix)
            plotFileName = "{}/npzPlots{}/{}.mean.pdf".format(baseDir, simType, prefix)
            data, titles = pu.readTrainXFromNpz(inFileName)
            print(inFileName)
            print(data[0].shape, data[1].shape)
            pu.makeHeatmap(data, prefix, titles, plotFileName, mean=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='A set of functions that run slurm \
        jobs to create and parse SLiM simulations for sweep detection.')

    parser.add_argument('-f', '--function', 
                        metavar='SCRIPT_FUNCTION',
                        help='Use one of the available \
                            functions by specifying its name.',
                        required=True, 
                        dest='run_func', 
                        type=str,
                        choices=['launch_sims',
                                 'combine_sims',
                                 'format_all',
                                 'train_nets',
                                 'plot_input_npz']
                        )

    args = parser.parse_args()

    return args

def main():
    ua = parse_arguments()

    # Gotta be a better way to do this
    if ua.run_func == 'launch_sims':
        launch_sims()
    elif ua.run_func == 'combine_sims':
        combine_sims()
    elif ua.run_func == 'format_all':
        format_all()
    elif ua.run_func == 'train_nets':
        train_nets()
    elif ua.run_func == 'plot_input_npz':
        plot_input_npz()
        
if __name__=='__main__':
    main()