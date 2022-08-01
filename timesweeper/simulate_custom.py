import logging
import multiprocessing as mp
import os
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


def randomize_selCoeff(lower_bound=0.02, upper_bound=0.2):
    """Draws selection coefficient from log normal dist to vary selection strength."""
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
    Can add other functions to this module and call them here e.g. pulling selection coeff from a dist.
    This block MUST INCLUDE the 'sweep' and 'outFile' params, and at the very least the outFile must be used as output for outputVCFSample.
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
        d_block[0]
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
    yaml_data = read_config(ua.yaml_file)
    work_dir, slim_file, slim_path, reps, rep_range = (
        yaml_data["work dir"],
        yaml_data["slimfile"],
        yaml_data["slim path"],
        yaml_data["reps"],
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

    d_blocks = []
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
            d_blocks.append((rep, d_block))

    pool = mp.Pool(processes=ua.threads)
    pool.starmap(run_slim, mp_args, chunksize=5)

    log_d_blocks(d_blocks, work_dir)

    logger.info(
        f"Simulations finished, parameters saved to {work_dir}/slim_params.csv."
    )
