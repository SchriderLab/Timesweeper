import logging
import multiprocessing as mp
import os
import subprocess
import argparse
from glob import glob

import numpy as np
import yaml

logging.basicConfig()
logger = logging.getLogger("sim_custom")


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


# Simulation
def randomize_selCoeff_uni(lower_bound=0.00025, upper_bound=0.25):
    """Draws selection coefficient from log uniform dist to vary selection strength."""
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )

    return rng.uniform(lower_bound, upper_bound, 1)[0]


def randomize_sampGens(num_timepoints, dev=50, span=200):
    rng = np.random.default_rng(
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    )
    start = round(rng.uniform(-dev, dev, 1)[0])
    if num_timepoints == 1:
        sampGens = [start + span]
    else:
        sampGens = [
            round(i) for i in np.linspace(start, start + span + 1, num_timepoints)
        ]

    return sampGens


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
    selCoeff = randomize_selCoeff_uni()
    if num_sample_points == 1:
        randomize_sampGens(num_sample_points)
    sampGens = [str(i) for i in randomize_sampGens(num_sample_points)]

    d_block = f"""\
    -d "sweep='{sweep}'" \
    -d "outFileVCF='{outFileVCF}'" \
    -d "outFileMS='{outFileMS}'" \
    -d "dumpFile='{dumpfile}'" \
    -d selCoeff={selCoeff} \
    -d sampGens='c({','.join(sampGens)})' \
    -d numSamples={num_sample_points} \
    -d sampleSizePerStep={inds_per_tp} \
    -d physLen={physLen} \
    -d seed={np.random.randint(0, 1e16)} \
    """
    if verbose:
        logger.info(f"Using the following constants with SLiM: {d_block}")

    return d_block


def simulate(slim_path, d_block, slimfile, logfile):
    cmd = " ".join(["time", slim_path, d_block, slimfile, ">>", logfile]).replace(
        "    ", ""
    )
    with open(logfile, "w") as ofile:
        ofile.write(cmd)

    try:
        subprocess.run(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        logger.error(e.output)


# VCF Processing
def read_multivcf(input_vcf):
    """Reads in file and returns as list of strings."""
    with open(input_vcf, "r") as input_file:
        raw_lines = [i.strip() for i in input_file.readlines()]

    return raw_lines


def split_multivcf(vcf_lines, header):
    """Splits the lines of multi-vcf file into list of vcf entries by <header> using itertools."""
    header_idxs = [i for i in range(len(vcf_lines)) if vcf_lines[i] == header]

    split_vcfs = []
    for idx in range(len(header_idxs[:-1])):
        split_vcfs.append(vcf_lines[header_idxs[idx] : header_idxs[idx + 1]])

    split_vcfs.append(vcf_lines[header_idxs[-1] :])

    return split_vcfs


def write_vcfs(vcf_lines, vcf_dir):
    """Writes list of vcf entries to numerically-sorted vcf files."""
    for idx, lines in enumerate(vcf_lines):
        with open(os.path.join(vcf_dir, f"{idx}.vcf"), "w") as outfile:
            outfile.writelines("\n".join(lines))


def index_vcf(vcf):
    """
    Indexes and sorts vcf file.
    Commands are run separately such that processes complete before the next one starts.
    """
    bgzip_cmd = f"bgzip -c {vcf} > {vcf}.gz"
    tabix_cmd = f"tabix -f -p vcf {vcf}.gz"
    bcftools_cmd = f"bcftools sort -Ov {vcf}.gz | bgzip -f > {vcf}.sorted.gz"
    tabix_2_cmd = f"tabix -f -p vcf {vcf}.sorted.gz"
    subprocess.run(
        bgzip_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        tabix_cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        bcftools_cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )
    subprocess.run(
        tabix_2_cmd.split(), stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
    )


def merge_vcfs(vcf_dir):
    num_files = len(glob(f"{vcf_dir}/*.vcf.sorted.gz"))
    if num_files == 1:
        cmd = f"""zcat {f"{vcf_dir}/0.vcf.sorted.gz"} > {vcf_dir}/merged.vcf"""

    else:
        cmd = f"""bcftools merge -Ov -0 \
                --force-samples --info-rules 'MT:join,S:join' \
                {" ".join([f"{vcf_dir}/{i}.vcf.sorted.gz" for i in range(num_files)])} > \
                {vcf_dir}/merged.vcf \
                """

    subprocess.run(cmd, shell=True)


def get_num_inds(vcf_file):
    num_ind = subprocess.check_output(
        """awk '{if ($1 == "#CHROM"){print NF-9; exit}}' """ + vcf_file, shell=True,
    )
    return int(num_ind)


def cleanup_intermed(vcf_dir):
    for ifile in glob(f"{vcf_dir}/*"):
        if "merged" not in ifile and "final" not in ifile:
            pass
            os.remove(ifile)


def make_vcf_dir(input_vcf):
    """Creates directory named after vcf basename."""
    dirname = os.path.basename(input_vcf).split(".")[0]
    dirpath = os.path.dirname(input_vcf)
    vcf_dir = os.path.join(dirpath, dirname)
    if os.path.exists(vcf_dir):
        for ifile in glob(f"{vcf_dir}/*"):
            os.remove(ifile)

    os.makedirs(vcf_dir, exist_ok=True)

    final_vcf = f"{input_vcf}.final"
    if os.path.exists(final_vcf):
        os.rename(final_vcf, f"{vcf_dir}/{final_vcf.split('/')[-1]}")

    return vcf_dir


def process_vcfs(input_vcf, num_tps):
    try:
        # Split into multiples after SLiM just concats to same file
        raw_lines = read_multivcf(input_vcf)
        split_lines = split_multivcf(raw_lines, "##fileformat=VCFv4.2")
        if len(split_lines) > 0:
            split_lines = split_lines[len(split_lines) - num_tps :]

            # Creates subdir for each rep
            vcf_dir = make_vcf_dir(input_vcf)
            write_vcfs(split_lines, vcf_dir)

            # Now index and merge
            [index_vcf(vcf) for vcf in glob(f"{vcf_dir}/*.vcf")]
            merge_vcfs(vcf_dir)

            cleanup_intermed(vcf_dir)

    except Exception as e:
        print(f"[ERROR] Couldn't process {e}")
        pass
    
    
def simulate_prep(
    vcf_file, num_sample_points, slimfile, slim_path, d_block, logfile, dumpFile
):
    simulate(slim_path, d_block, slimfile, logfile)
    os.remove(dumpFile)

    process_vcfs(vcf_file, num_sample_points)
    os.remove(vcf_file)


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
    - sweep [str] One of "neut", "sdn", or "ssv". If you're testing only a neut/sdn model,
        make the ssv a dummy switch for the neutral scenario.
    - outFile [path] You will need to define this as a population outputVCFSample input, with replace=F and append=T.
        This does *not* need to be specified by you in the custom -d block, it will be standardized to work with the rest of the pipeline using work_dir.
        example line for slim script: `p1.outputVCFSample(sampleSizePerStep, replace=F, append=T, filePath=outFile);`

    """
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
        ua.rep_range,
        yaml_data["num_sample_points"],
        yaml_data["inds_per_tp"],
        yaml_data["physLen"],
    )

    vcf_dir = f"{work_dir}/vcfs"
    ms_dir = f"{work_dir}/mss"
    dumpfile_dir = f"{work_dir}/dumpfiles"
    logfile_dir = f"{work_dir}/logs"

    sweeps = ["neut", "sdn", "ssv"]

    for i in [vcf_dir, dumpfile_dir, logfile_dir]:
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
            logFile = f"{logfile_dir}/{sweep}/{rep}.log"

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

            mp_args.append(
                (
                    outFileVCF,
                    num_sample_points,
                    slim_file,
                    slim_path,
                    d_block,
                    logFile,
                    dumpFile,
                )
            )

    pool = mp.Pool(processes=ua.threads)
    pool.starmap(simulate_prep, mp_args, chunksize=1)

if __name__=="__main__":
    sim_c_parser = argparse.ArgumentParser()
    sim_c_parser.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count(),
        dest="threads",
        help="Number of processes to parallelize across. Defaults to all.",
    )
    sim_c_parser.add_argument(
        "--rep-range",
        required=False,
        dest="rep_range",
        nargs=2,
        help="<start, stop>. If used, only range(start, stop) will be simulated for reps. \
            This is to allow for easy SLURM parallel simulations.",
    )
    sim_c_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML_CONFIG",
        required=True,
        dest="yaml_file",
        help="YAML config file with all required options defined.",
    )
    args = sim_c_parser.parse_args()
    main(args)