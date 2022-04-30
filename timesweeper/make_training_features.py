import argparse as ap
import logging
import multiprocessing as mp
import os
import pickle
import sys
from glob import glob
from itertools import cycle

import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

import find_sweeps as fs
from utils import snp_utils as su

logging.basicConfig()
logger = logging.getLogger("make_training_feats")
logger.setLevel("INFO")


def add_missingness(data, m_rate, nan_val=-1):
    rng = default_rng(np.random.seed(int.from_bytes(os.urandom(4), byteorder="little")))
    missing = rng.binomial(1, m_rate, data.shape)
    data[missing == 1] = nan_val

    return data


def get_aft_central_window(snps, genos, samp_sizes, win_size, sweep, missingness):
    """
    Iterates through windows of MAF time-series matrix and gets the central window.
    Not the most efficient way to do it, but it replicates the logic of timesweeper and I like the intuitiveness of that.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP.
        genos (allel.GenotypeArray): Genotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        sweep (str): One of ["neut", "hard", "soft"]
        missingness (float): Parameter of binomial distribution to pull missingness from.

    Returns:
        np.arr: The central-most window, either based on mutation type or closest to half size of chrom.
    """
    ts_aft = fs.prep_ts_aft(genos, samp_sizes)

    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in centers:
        if sweep in ["hard", "soft"]:
            if snps[center][2] == 2:  # Check for mut type of 2
                win_idxs = fs.get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window
        else:
            if center == centers[int(len(centers) / 2)]:
                win_idxs = fs.get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window

    missing_center_aft = add_missingness(center_aft, m_rate=missingness)

    return missing_center_aft


def check_freq_increase(ts_afs, min_increase=0.25):
    """Quick test to make sure a given example is increasing properly, can be used to filter training."""
    center = int(ts_afs.shape[0] / 2)
    if ts_afs[-1, center] - ts_afs[0, center] >= min_increase:
        return True
    else:
        return False


def parse_ua(u_args=None):
    uap = ap.ArgumentParser(
        description="Creates training data from simulated merged vcfs after process_vcfs.py has been run."
    )
    uap.add_argument(
        "--threads",
        required=False,
        type=int,
        default=mp.cpu_count() - 1,
        dest="threads",
        help="Number of processes to parallelize across.",
    )
    uap.add_argument(
        "-o",
        "--outfile",
        required=False,
        type=str,
        default="training_data.pkl",
        dest="outfile",
        help="Pickle file to dump dictionaries with training data to. Should probably end with .pkl.",
    )
    uap.add_argument(
        "-m",
        "--missingness",
        metavar="MISSINGNESS",
        dest="missingness",
        type=float,
        required=False,
        default=0.0,
        help="Missingness rate in range of [0,1], used as the parameter of a binomial distribution for randomly removing known values.",
    )
    uap.add_argument(
        "-f",
        "--freq-increase-threshold",
        metavar="FREQ_INC_THRESHOLD",
        dest="freq_inc_thr",
        type=float,
        required=False,
        help="If given, only include sim replicates where the sweep site has a minimum increase of <freq_inc_thr> from the first timepoint to the last.",
    )
    uap.add_argument(
        "--verbose",
        action="store_true",
        dest="verbose",
        help="Whether to print error messages, usually from VCF loading errors.",
    )
    uap.add_argument(
        "--no-progress",
        action="store_true",
        dest="no_progress",
        help="Turn off progress bar.",
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
        "-s",
        "--sample-sizes",
        dest="samp_sizes",
        help="Number of individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling poinfs.",
        required=True,
        nargs="+",
        type=int,
    )

    return uap.parse_args(u_args)


def worker(
    in_vcf, samp_sizes, win_size, missingness, freq_inc_thr, verbose=False,
):
    benchmark = True
    try:
        id = fs.get_rep_id(in_vcf)
        sweep = fs.get_sweep(in_vcf)
        vcf = su.read_vcf(in_vcf, benchmark)

        genos, snps = su.vcf_to_genos(vcf, benchmark)
        central_aft = get_aft_central_window(
            snps, genos, samp_sizes, win_size, sweep, missingness
        )

        if sweep != "neut":
            if freq_inc_thr:
                if check_freq_increase(central_aft, freq_inc_thr):
                    return id, sweep, central_aft
            else:
                return id, sweep, central_aft

        else:
            return id, sweep, central_aft

    except Exception as e:
        if verbose:
            logger.warning(f"Could not process {in_vcf}")
            logger.warning(f"Exception: {e}")
            sys.stdout.flush()
            sys.stderr.flush()
        return None


def main(ua):
    if ua.config_format == "yaml":
        yaml_data = fs.read_config(ua.yaml_file)
        work_dir, samp_sizes, threads = (
            yaml_data["work dir"],
            yaml_data["sample sizes"],
            ua.threads,
        )

    win_size = 51  # Must be consistent with training data
    filelist = glob(f"{work_dir}/vcfs/*/*/merged.vcf", recursive=True)
    work_args = zip(
        filelist,
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([ua.missingness]),
        cycle([ua.freq_inc_thr]),
        cycle([ua.verbose]),
    )

    pool = mp.Pool(threads)
    if ua.no_progress:
        work_res = pool.starmap(worker, work_args, chunksize=4,)
    else:
        work_res = pool.starmap(
            worker,
            tqdm(work_args, total=len(filelist), desc="Condensing training data",),
            chunksize=4,
        )
    # Save this way so that if a single piece of data needs to be inspected/plotted it's always identifiable
    pickle_dict = {}
    for res in work_res:
        if res:
            rep, sweep, aft = res
            if sweep not in pickle_dict.keys():
                pickle_dict[sweep] = {}

            pickle_dict[sweep][rep] = {}
            pickle_dict[sweep][rep]["aft"] = aft

    pickle.dump(pickle_dict, open(ua.outfile, "wb"))


if __name__ == "__main__":
    ua = parse_ua()
    main(ua)
