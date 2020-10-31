import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A set of functions that run slurm \
                                                  jobs to create and parse SLiM \
                                                  simulations for sweep detection."
    )

    parser.add_argument(
        "-f",
        "--function",
        metavar="SCRIPT_FUNCTION",
        help="Use one of the available \
                            functions by specifying its name.",
        required=True,
        dest="run_func",
        type=str,
        choices=["launch_sims", "clean_sims", "create_feat_vecs", "train_nets"],
    )

    parser.add_argument(
        "-s",
        "--slim-paramfile",
        metavar="SLIM_SIMULATION_FILE",
        help="Filename of slimfile in /slimfiles/ dir.\
                              New directory will be created with this as prefix \
                              and will contain all the relevant files for this \
                              set of parameters.",
        dest="slim_name",
        type=str,
        required=False,
        default="test.slim",
    )

    args = parser.parse_args()

    return args


def run_batch_job(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile):
    with open(launchFile, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % (jobName))
        f.write("#SBATCH --time=%s\n" % (wallTime))
        f.write("#SBATCH --partition=%s\n" % (qName))
        f.write("#SBATCH --output=%s\n" % (logFile))
        f.write("#SBATCH --mem=%s\n" % (mbMem))
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write("\n%s\n" % (cmd))
    os.system("sbatch %s" % (launchFile))