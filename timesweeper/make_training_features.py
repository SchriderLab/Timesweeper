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
        elif sweep == "neut":
            if center == centers[int(len(centers) / 2)]:
                win_idxs = fs.get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window

    missing_center_aft = add_missingness(center_aft, m_rate=missingness)

    return missing_center_aft


def parse_ua(u_args=None):

    return uap.parse_args(u_args)


def worker(in_vcf, samp_sizes, win_size, missingness, benchmark=True):
    try:
        id = fs.get_rep_id(in_vcf)
        sweep = fs.get_sweep(in_vcf)
        vcf = su.read_vcf(in_vcf, benchmark)

        genos, snps = su.vcf_to_genos(vcf, benchmark)
        central_aft = get_aft_central_window(
            snps, genos, samp_sizes, win_size, sweep, missingness
        )

        return id, sweep, central_aft

    except Exception as e:
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
    elif ua.config_format == "cli":
        work_dir, samp_sizes, threads = (
            ua.work_dir,
            ua.samp_sizes,
            ua.threads,
        )

    win_size = 51  # Must be consistent with training data

    work_args = zip(
        glob(f"{work_dir}/vcfs/*/*/merged.vcf", recursive=True),
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([ua.missingness]),
    )

    pool = mp.Pool(threads)
    work_res = pool.starmap(worker, work_args, chunksize=10)

    # Save this way so that if a single piece of data needs to be inspected/plotted it's always identifiable
    pickle_dict = {}
    for res in work_res:
        if res:
            rep, sweep, aft = res
            if sweep not in pickle_dict.keys():
                pickle_dict[sweep] = {}

            pickle_dict[sweep][rep] = {}
            pickle_dict[sweep][rep]["aft"] = aft

    # with open(, "w") as pklfile:
    pickle.dump(pickle_dict, open(f"{work_dir}/training_data.pkl", "wb"))


if __name__ == "__main__":
    ua = parse_ua()
    main(ua)
