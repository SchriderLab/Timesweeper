import argparse
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

import numpy as np

from timesweeper import read_config

logging.basicConfig()
logger = logging.getLogger("simple_simulate")


def make_d_block(sweep, outFile, dumpfile, verbose=False):
    """
    This is meant to be a very customizeable block of text for adding custom args to SLiM as constants.
    Can add other functions to this module and call them here e.g. pulling selection coeff from a dist.
    This block MUST INCLUDE the 'sweep' and 'outFile' params, and at the very least the outFile must be used as output for outputVCFSample.
    Please note that when feeding strings as a constant you must escape them since this is a shell process.
    """

    num_sample_points = 20
    inds_per_tp = 5  # Diploid inds
    physLen = 500000

    d_block = f"""
    -d sweep=\"{sweep}\" \
    -d outFile=\"{outFile}\" \
    -d dumpFile=\"{dumpfile}\" \
    -d samplingInterval={200/num_sample_points} \
    -d numSamples={num_sample_points} \
    -d sampleSizePerStep={inds_per_tp} \
    -d physLen={physLen} \
    -d seed={np.random.randint(0, 1e16)} \
    """
    if verbose:
        logger.info(f"Using the following constants with SLiM: {d_block}")

    return d_block


def run_slim(slimfile, slim_path, d_block):
    cmd = f"{slim_path} {d_block} {slimfile}"

    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        logger.error(e.output)

    sys.stdout.flush()
    sys.stderr.flush()


def get_ua():
    uap = argparse.ArgumentParser(
        description="Simulates selection for training Timesweeper using a pre-made SLiM script."
    )
    uap.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count(),
        dest="threads",
        help="Number of processes to parallelize across.",
    )
    uap.add_argument(
        "--rep-range",
        required=False,
        dest="rep_range",
        nargs=2,
        help="<start, stop>. If used, only range(start, stop) will be simulated for reps. \
            This is to allow for easy SLURM parallel simulations.",
    )
    subparsers = uap.add_subparsers(dest="config_format")
    subparsers.required = True
    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "-w",
        "--work-dir",
        dest="work_dir",
        type=str,
        help="Directory used as work dir for simulate modules. Should contain simulated vcfs processed using process_vcf.py.",
        required=False,
        default=os.getcwd(),
    )
    cli_parser.add_argument(
        "-i",
        "--slim-file",
        required=True,
        type=str,
        help="SLiM Script to simulate with. Must output to a single VCF file. ",
        dest="slim_file",
    )
    cli_parser.add_argument(
        "--slim-path",
        required=False,
        type=str,
        default="./SLiM/build/slim",
        dest="slim_path",
        help="Path to SLiM executable.",
    )
    cli_parser.add_argument(
        "--reps",
        required=False,
        type=int,
        help="Number of replicate simulations to run. If using rep_range can just fill with random int.",
        dest="reps",
    )
    return uap.parse_args()


def main():
    """
    For simulating non-stdpopsim SLiMfiles.
    Currently only works with 1 pop models where m2 is the sweep mutation.
    Otherwise everything else is identical to stdpopsim version, just less complex.

    Generalized block of '-d' arguments to give to SLiM at the command line allow for 
    flexible script writing within the context of this wrapper. If you write your SLiM script
    to require args set at runtime, this should be easily modifiable to do what you need and 
    get consistent results to plug into the rest of the workflow.

    The two things you *will* need to specify in your '-d' args to SLiM (and somewhere in the slim script) are:
    - sweep [str] One of "neut", "hard", or "soft". If you're testing only a neut/hard model, 
        make the soft a dummy switch for the neutral scenario.
    - outFile [path] You will need to define this as a population outputVCFSample input, with replace=T and append=T. 
        This does *not* need to be specified by you in the custom -d block, it will be standardized to work with the rest of the pipeline using work_dir.
        example line for slim script: `p1.outputVCFSample(sampleSizePerStep, replace=F, append=T, filePath=outFile);`
        
    Please note that since this is supposed to be modifiable I am leaving it as a cli-argument module only.
    This means that you will have to replicate any args this may share with the YAML you use for the rest of the workflow, if that's how you choose to run it.
    This also means, however, that you 
    """
    ua = get_ua()
    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        work_dir, slim_file, slim_path, reps, rep_range = (
            yaml_data["work dir"],
            yaml_data["slimfile"],
            yaml_data["slim path"],
            yaml_data["reps"],
            ua.rep_range,
        )
    elif ua.config_format == "cli":
        work_dir, slim_file, slim_path, reps, rep_range = (
            ua.work_dir,
            ua.slim_file,
            ua.slim_path,
            ua.reps,
            ua.rep_range,
        )

    work_dir = work_dir
    vcf_dir = f"{work_dir}/vcfs"
    dumpfile_dir = f"{work_dir}/dumpfiles"

    sweeps = ["neut", "hard", "soft"]

    for i in [vcf_dir, dumpfile_dir]:
        for sweep in sweeps:
            os.makedirs(f"{i}/{sweep}", exist_ok=True)

    mp_args = []
    # Inject info into SLiM script and then simulate, store params for reproducibility
    if rep_range:  # Take priority
        replist = range(int(rep_range[0]), int(rep_range[1]) + 1)
    else:
        replist = range(reps)

    for rep in replist:
        for sweep in sweeps:
            outFile = f"{vcf_dir}/{sweep}/{rep}.multivcf"
            dumpFile = f"{dumpfile_dir}/{sweep}/{rep}.dump"

            # Grab those constants to feed to SLiM
            if rep == 0:
                d_block = make_d_block(sweep, outFile, dumpFile, True)
            else:
                d_block = make_d_block(sweep, outFile, dumpFile, False)

            mp_args.append((slim_file, slim_path, d_block))

    pool = mp.Pool(processes=ua.threads)
    pool.starmap(run_slim, mp_args, chunksize=5)

    # Log the constant params just in case, just use last one
    with open(f"{work_dir}/slim_params.txt", "w") as paramsfile:
        cleaned_block = "\n".join([i.strip() for i in d_block.split() if "-d" not in i])
        paramsfile.writelines(cleaned_block)

    # Cleanup
    for rep in replist:
        for sweep in sweeps:
            dumpFile = f"{dumpfile_dir}/{sweep}/{rep}.dump"
            os.remove(dumpFile)

    logger.info(
        f"Simulations finished, parameters saved to {work_dir}/slim_params.csv."
    )


if __name__ == "__main__":
    main()
