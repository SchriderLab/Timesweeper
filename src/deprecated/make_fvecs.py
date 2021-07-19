from glob import glob
import os
import sys
import subprocess

"""
Quick script that runs a for loop across all files in a given directory and \
    calls the diploSHIC fvecSim function to generate feature vectors.

Used to parallelize the process across directories without submitting too many \
    SLURM jobs.

Format of input should be 
python make_fvecs.py <msOutDir> <baseDir>
"""

for cleanfile in glob(os.path.join(sys.argv[1], "muts/*/*/*.ms")):
    print(cleanfile)
    if not os.path.exists(cleanfile.split(".")[0] + ".fvec"):
        cmd = "python /overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/diploSHIC/diploSHIC.py fvecSim haploid {} {} ".format(
            cleanfile, cleanfile.split(".")[0] + ".fvec"
        )
        print(cmd)
        subprocess.run(cmd, shell=True)