import argparse
import glob
import os
import random

import numpy as np

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
    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
        os.system("mkdir -p {}/{} {}/{} {}/{} {}/{}".format(baseSimDir, name, 
                                                            baseLogDir, name, 
                                                            baseDumpDir, name))
    os.mkdir(os.path.join(slimDir, 'jobfiles'))

    physLen=100000
    numBatches = 1000
    repsPerBatch=100
    for timeSeries in [True,False]:
        for i in range(numBatches):
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
                cmd = "python {}/timesweeper/runAndParseSlim.py {}/test/{}.slim {} {} {} {} {} {} {} {} {} {} {} > {}".format(baseDir, 
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

                ut.run_batch_job(cmd, simType+suffix, "{}/jobfiles/{}{}.txt".format(slimDir, simType, suffix), "10:00", "general", "1G", "{}/{}_{}.log".format(logDir, simType, i))

def clean_sims():
    """Finds and iterates through all raw msOut files recursively, \
        cleans them by stripping out unwanted lines.
    """
    for dirtyfile in glob.glob('./**/*.msOut', recursive=True):
        if 'cleaned' in dirtyfile:
            continue
        else:
            ut.clean_msOut(dirtyfile)

def create_shic_feats():
    """Finds all cleaned MS-format files recursively and runs diploSHIC fvecSim on them.
    Writes files to fvec subdirectory of sweep type.
    #TODO Make an arg pass to this to specify which folder you want
    """
    for cleanfile in glob.glob('./**/cleaned*.msOut', recursive=True):
        filepath = os.path.split(cleanfile)[0]
        filename = os.path.split(cleanfile)[1].split('.')[0]
    
        if not os.path.exists(os.path.join(filepath, 'fvecs')):
            os.mkdir(os.path.join(filepath, 'fvecs'))
            os.mkdir(os.path.join(filepath, 'fvecs/logs'))

        cmd = "python {}/diploSHIC/diploSHIC.py fvecSim haploid {} {}".format(baseDir, cleanfile, os.path.join(filepath, 'fvecs', filename + ".fvec"))
        ut.run_batch_job(cmd, "shic", "{}/jobfiles/shic.txt".format(slimDir), "1:00:00", "general", "1GB", "{}/{}_shic_fvec.log".format(os.path.join(filepath, 'fvecs', 'logs'), filename))

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
    elif ua.run_func == 'create_feat_vecs':
        create_shic_feats()
    elif ua.run_func == 'train_nets':
        train_nets()

        
if __name__=='__main__':
    main()
