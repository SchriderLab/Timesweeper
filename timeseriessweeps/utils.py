import argparse
import sys
import os

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