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

def clean_msOut(msFile):
    """Reads in MS-style output from Slim, removes all extraneous information \
        so that the MS output is the only thing left. Writes to "cleaned_" slimfile.

    Args:
        msFile (str): Filepath of slim output file.
    """
    with open(msFile, 'r') as rawfile:
        rawMS = [i.strip() for i in rawfile.readlines()]

    #Filter out lines that have integers after them
    cleanMS = []
    listMS = [i for i in rawMS if i]
    for i in range(len(listMS)):
        #print(listMS[i].split())
        #Filter out lines where integer line immediately follows
        if ((listMS[i] == '// Initial random seed:') 
            or (listMS[i] == '// Starting run at generation <start>:')
            or (listMS[i-1] == '// Initial random seed:') 
            or (listMS[i-1] == '// Starting run at generation <start>:')):
            #Remove both the header for seeds and the value
            #For anything that is a 2-line entry with number as second
            continue
        #Filter out commented lines that aren't ms related
        #Get rid of lines like '// RunInitializeCallbacks():'
        elif ((listMS[i].split()[0] == '//') and (len(listMS[i].split()) > 1)):        
            continue
        #Capture SHIC-required header
        elif listMS[i].split()[0] == 'SLiM/build/slim':
            shic_header = listMS[i]
        else:
            cleanMS.append(listMS[i])

    #Filter out everything else that isn't ms related
    cleanMS = [i for i in cleanMS if ';' not in i]
    cleanMS = [i for i in cleanMS if '#' not in i]
    cleanMS.pop(0) #Remove slimfile name

    cleanMS.insert(0, shic_header)
        
    with open('cleaned_' + msFile, 'w') as outFile:
        outFile.write('\n'.join(cleanMS))
        
