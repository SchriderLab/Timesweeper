import argparse
import os
import sys

import numpy as np


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('-mode', 
                        metavar='TRAIN_FILT_PREDICT',
                        help='Use one of the available modes, training a new\
                            model, filtering from other callsets using\
                            pre-generated npy files, or predicting from a BAM.',
                        required=True, 
                        dest='run_mode', 
                        type=str)

    args = parser.parse_args()

    return args


def run_batch_job(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile):
    with open(launchFile,"w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" %(jobName))
        f.write("#SBATCH --time=%s\n" %(wallTime))
        f.write("#SBATCH --partition=%s\n" %(qName))
        f.write("#SBATCH --output=%s\n" %(logFile))
        f.write("#SBATCH --mem=%s\n" %(mbMem))
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write("\n%s\n" %(cmd))
    os.system("sbatch %s" %(launchFile))
    

def readTrainXFromNpz(inFileName):
    u = np.load(inFileName)
    trainX, testX, valX = u['trainX'], u['testX'], u['valX']
    print(trainX.shape)
    if "haps" in inFileName:
        trainX = trainX[:,:20]
    print(trainX.shape)
    trainy, testy, valy = u['trainy'], u['testy'], u['valy']
    one = trainy == 1
    zero = trainy == 0
    return [trainX[one], trainX[zero]], ["sweep", "neut"]
