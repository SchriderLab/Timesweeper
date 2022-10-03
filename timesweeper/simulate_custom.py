import argparse
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

logging.basicConfig()
logger = logging.getLogger("simple_simulate")


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


def make_d_block(
    sweep,
    outFileVCF,
    outFileMS,
    dumpfile,
    num_sample_points,
    inds_per_tp,
    physLen,
    verbose=False,
):
    """
    This is meant to be a very customizeable block of text for adding custom args to SLiM as constants.
    Can add other functions to this module and call them here e.g. pulling selection coeff from a dist.
    This block MUST INCLUDE the 'sweep' and 'outFile' params, and at the very least the outFile must be used as output for outputVCFSample.
    Please note that when feeding strings as a constant you must escape them since this is a shell process.
    """
    d_block = f"""
    -d sweep=\"{sweep}\" \
    -d outFileVCF=\"{outFileVCF}\" \
    -d outFileMS=\"{outFileMS}\" \
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
    uap.add_argument(
        "-y",
        "--yaml",
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all required options defined.",
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
    - sweep [str] One of "neut", "sdn", or "ssv". If you're testing only a neut/sdn model, 
        make the ssv a dummy switch for the neutral scenario.
    - outFile [path] You will need to define this as a population outputVCFSample input, with replace=F and append=T. 
        This does *not* need to be specified by you in the custom -d block, it will be standardized to work with the rest of the pipeline using work_dir.
        example line for slim script: `p1.outputVCFSample(sampleSizePerStep, replace=F, append=T, filePath=outFile);`
        
    """
    ua = get_ua()
    yaml_data = read_config(ua.yaml_file)
    (
        work_dir,
        slim_file,
        slim_path,
        reps,
        rep_range,
        num_sample_points,
        inds_per_tp,
        physLen,
    ) = (
        yaml_data["work dir"],
        yaml_data["slimfile"],
        yaml_data["slim path"],
        yaml_data["reps"],
        yaml_data["num_sample_points"],
        yaml_data["inds_per_tp"],
        yaml_data["physLen"],
        ua.rep_range,
    )

    vcf_dir = f"{work_dir}/vcfs"
    ms_dir = f"{work_dir}/mss"
    dumpfile_dir = f"{work_dir}/dumpfiles"

    sweeps = ["neut", "sdn", "ssv"]

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
            outFileVCF = f"{vcf_dir}/{sweep}/{rep}.multivcf"
            outFileMS = f"{ms_dir}/{sweep}/{rep}.multiMsOut"
            dumpFile = f"{dumpfile_dir}/{sweep}/{rep}.dump"

            # Grab those constants to feed to SLiM
            if rep == 0:
                d_block = make_d_block(
                    sweep,
                    outFileVCF,
                    outFileMS,
                    dumpFile,
                    num_sample_points,
                    inds_per_tp,
                    physLen,
                    True,
                )
            else:
                d_block = make_d_block(
                    sweep,
                    outFileVCF,
                    outFileMS,
                    dumpFile,
                    num_sample_points,
                    inds_per_tp,
                    physLen,
                    False,
                )

            mp_args.append((slim_file, slim_path, d_block))

    pool = mp.Pool(processes=ua.threads)
    pool.starmap(run_slim, mp_args, chunksize=5)

    # Cleanup
    # shutil.rmtree(dumpfile_dir)

    # Log the constant params just in case, just use last one
    with open(f"{work_dir}/slim_params.txt", "w") as paramsfile:
        cleaned_block = "\n".join([i.strip() for i in d_block.split() if "-d" not in i])
        paramsfile.writelines(cleaned_block)

    logger.info(
        f"Simulations finished, parameters saved to {work_dir}/slim_params.csv."
    )


if __name__ == "__main__":
    main()
