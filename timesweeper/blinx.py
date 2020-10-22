import argparse
import glob
import os
import random

import numpy as np

import plotting_utils as pu
import utils as ut

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

#TODO store these as a json made by user
maxSnps = 200

slimFile = 'test'
baseDir = '/proj/dschridelab/timeSeriesSweeps' 

slimDir = baseDir + '/' + slimFile

baseSimDir=slimDir+"/sims"
baseDumpDir=slimDir+"/simDumps"
baseLogDir=slimDir+"/simLogs"

# End Testing vars #########################################################


def launch_sims():
    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]: #Why no soft?
        os.system("mkdir -p {}/{} {}/{} {}/{} {}/{}".format(baseSimDir, name, 
                                                            baseLogDir, name, 
                                                            baseDumpDir, name,
                                                            slimDir, name))

    physLen=100000
    numBatches = 1000
    repsPerBatch=100
    for timeSeries in [True,False]:
        for i in [1]:#range(numBatches):
            if timeSeries:
                suffix = ""
            else:
                suffix = "1Samp"
            for simType in ["hard", "soft", "neut"]:
                dumpDir = baseDumpDir + "/" + simType + suffix
                logDir = baseLogDir + "/" + simType + suffix
                outFileName = "{}/{}/{}_{}.msOut".format(baseSimDir, simType, simType, i)
                dumpFileName = "{}/{}_{}.trees.dump".format(dumpDir, simType, i)
                #Replace /test/ with slimdfile directory
                cmd = "python {}/timesweeper/scripts/runAndParseSlim.py {}/test/{}.slim {} {} {} {} {} {} {} {} {} {} {} > {}".format(baseDir, 
                                                                                                                                      baseDir,
                                                                                                                                      slimFile,
                                                                                                                                      sampleSizePerStepTS, 
                                                                                                                                      numSamplesTS, 
                                                                                                                                      samplingIntervalTS, 
                                                                                                                                      sampleSizePerStep1Samp, 
                                                                                                                                      numSamples1Samp, 
                                                                                                                                      samplingInterval1Samp, 
                                                                                                                                      repsPerBatch, 
                                                                                                                                      physLen, 
                                                                                                                                      timeSeries, 
                                                                                                                                      simType, 
                                                                                                                                      dumpFileName, 
                                                                                                                                      outFileName)

                ut.run_batch_job(cmd, simType+suffix, "{}/{}{}.txt".format(slimDir, simType, suffix), "10:00", "general", "1G", "{}/{}_{}.log".format(logDir, simType, i))

def clean_sims():
    for dirtyfile in glob.glob('./**/*.msOut', recursive=True):
        ut.clean_msOut(dirtyfile)

def create_shic_feats():
    stepToInputFormat = {'a':'ali', 'b':'sfs', 'c':'haps'}
    #stepToInputFormat = {'a':'ali'}

    suffices = ["", "1Samp"]
    for i in range(len(suffices)):
        suffix = suffices[i]
        inDir = slimDir + "/sims" + suffix
        outDir = slimDir + "/fvecs" + suffix
        os.system("mkdir -p {}".format(outDir))

        for step in stepToInputFormat:
            ut.clean_msOut(simDir)
            cmd = "python {}/diploSHIC/diploSHIC.py fvecSim haploid {} {}".format(baseDir, step, stepToInputFormat[step], inDir, maxSnps, sampleSizesPerTS, outDir, stepToInputFormat[step])
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
   
def main():
    ua = ut.parse_arguments()

    #TODO Gotta be a better way to do this
    if ua.run_func == 'launch_sims':
        launch_sims()
    elif ua.run_func == 'clean_sims':
        clean_sims()
    elif ua.run_func == 'train_nets':
        train_nets()

        
if __name__=='__main__':
    main()
