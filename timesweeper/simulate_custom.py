import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pprint import pprint
import numpy as np
from .utils.gen_utils import read_config, get_logger

logger = get_logger("sim_custom")


def randomize_selCoeff_uni(lower_bound=0.000025, upper_bound=0.25):
    """Draws selection coefficient from log uniform dist to vary selection strength."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )
    log_low = np.math.log10(lower_bound)
    log_upper = np.math.log10(upper_bound)
    rand_log = rng.uniform(log_low, log_upper, 1)

    return 10 ** rand_log[0]


def make_d_block(sweep, outFile, dumpfile, verbose=False):
    """
    This is meant to be a very customizeable block of text for adding custom args to SLiM as constants.
    You can add other functions to this module and call them here e.g. pulling selection coeff from a distribution.

    **This block MUST INCLUDE the 'scenario' and 'outFile' params, and at the very least the outFile must be used as output for outputVCFSample with `replacement=F`.**

    Please note that when feeding strings as a constant you must escape them since this is a shell process.
    """
    selCoeff = randomize_selCoeff()
    num_sample_points = 20
    inds_per_tp = 10  # Diploid inds
    physLen = 500000

    d_block = f"""
    -d sweep=\"{sweep}\" \
    -d outFile=\"{outFile}\" \
    -d dumpFile=\"{dumpfile}\" \
    -d samplingInterval={200/num_sample_points} \
    -d numSamples={num_sample_points} \
    -d sampleSizePerStep={inds_per_tp} \
    -d selCoeff={selCoeff} \
    -d physLen={physLen} \
    -d seed={np.random.randint(0, 1e16)} \
    """
    if verbose:
        logger.info(f"Using the following constants with SLiM: {pprint(d_block)}")

    return d_block


def clean_d_block(d_block):
    return (
        str(d_block[0])
        + "\t"
        + "\t".join([i.strip() for i in d_block[1].split() if "-d" not in i])
    )


def log_d_blocks(d_block_list, work_dir):
    header = "rep\t" + "\t".join(
        [
            i.strip().split("=")[0].strip()
            for i in d_block_list[0][1].split()
            if "-d" not in i
        ]
    )

    with open(f"{work_dir}/slim_params.txt", "w") as paramsfile:
        paramsfile.write(header + "\n")
        for d in d_block_list:
            paramsfile.writelines(clean_d_block(d) + "\n")


def run_slim(slimfile, slim_path, d_block):
    cmd = f"{slim_path} {d_block} {slimfile}"

    try:
        subprocess.check_output(cmd.split())
    except subprocess.CalledProcessError as e:
        logger.error(e.output)

    sys.stdout.flush()
    sys.stderr.flush()


def main(ua):
    """
    For simulating non-stdpopsim SLiMfiles.
    - Currently only works with 1 pop models where m2 is the sweep mutation.
    - Otherwise everything else is identical to stdpopsim version, just less complex.
    - Generalized block of '-d' arguments to give to SLiM at the command line allow for
        flexible script writing within the context of this wrapper. If you write your SLiM script
        to require args set at runtime, this should be easily modifiable to do what you need and
        get consistent results to plug into the rest of the workflow.
    - The two things you *will* need to specify in your '-d' args to SLiM (and somewhere in the slim script) are:
        1. scenario [str] Some descriptor of what type of model you're simulating. De facto example
            is "neutral", "hard_sweep", or "soft_sweep".
        2. outFile [path] You will need to define this as a population outputVCFSample input, with replace=F and append=T.
            This does *not* need to be specified by you in the custom -d block, it will be standardized to work with the rest of the pipeline using work_dir.
            example line for slim script: `p1.outputVCFSample(sampleSizePerStep, replace=F, append=T, filePath=outFile);`

    """
    yaml_data = read_config(ua.yaml_file)
    scenarios, work_dir, slim_file, slim_path, reps, rep_range = (
        yaml_data["scenarios"],
        yaml_data["work dir"],
        yaml_data["slimfile"],
        yaml_data["slim path"],
        yaml_data["reps"],
        ua.rep_range,
    )

    work_dir = work_dir
    vcf_dir = f"{work_dir}/vcfs"
    dumpfile_dir = f"{work_dir}/dumpfiles"

    for i in [vcf_dir, dumpfile_dir]:
        for s in scenarios:
            os.makedirs(f"{i}/{s}", exist_ok=True)

    mp_args = []
    # Inject info into SLiM script and then simulate, store params for reproducibility
    if rep_range:  # Take priority
        replist = range(int(rep_range[0]), int(rep_range[1]) + 1)
    else:
        replist = range(reps)

    d_blocks = []
    for rep in replist:
        for s in scenarios:
            outFile = f"{vcf_dir}/{s}/{rep}.multivcf"
            dumpFile = f"{dumpfile_dir}/{s}/{rep}.dump"

            # Grab those constants to feed to SLiM
            if rep == 0:
                d_block = make_d_block(s, outFile, dumpFile, True)
            else:
                d_block = make_d_block(s, outFile, dumpFile, False)

            mp_args.append((slim_file, slim_path, d_block))
            d_blocks.append((rep, d_block))

    pool = mp.Pool(processes=ua.threads)
    pool.starmap(run_slim, mp_args, chunksize=5)

    log_d_blocks(d_blocks, work_dir)

    logger.info(
        f"Simulations finished, parameters saved to {work_dir}/slim_params.csv."
    )
