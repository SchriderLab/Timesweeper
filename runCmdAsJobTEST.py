import sys, os

def runCmdAsJobWithoutWaitingWithLog(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile):
    with open(launchFile,"w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" %(jobName))
        f.write("#SBATCH --time=%s\n" %(wallTime))
        f.write("#SBATCH --partition=%s\n" %(qName))
        f.write("#SBATCH --output=%s\n" %(logFile))
        f.write("#SBATCH --mem=%s\n" %(mbMem))
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write("\n%s\n" %("unset OMP_NUM_THREADS"))
        f.write("\n%s\n" %("SIMG_PATH=/nas/longleaf/apps/tensorflow_nogpu_py3/1.9.0/simg"))
        f.write("\n%s\n" %("SIMG_NAME=tensorflow1.9.0-py3-nogpu-ubuntu18.04.simg"))
        f.write("\n%s\n" %("singularity exec --nv -B /pine -B /proj $SIMG_PATH/$SIMG_NAME " cmd))
    os.system("sbatch %s" %(launchFile))

def main():
    cmd, jobName, launchFile, wallTime, qName, mbMem, logFile = sys.argv[1:]
    runCmdAsJobWithoutWaitingWithLog(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile)

if __name__ == "__main__":
    main()
