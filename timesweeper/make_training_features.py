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

from .find_sweeps_vcf import prep_ts_aft, get_window_idxs
from .utils import snp_utils as su
from .utils.gen_utils import get_rep_id, get_sweep, read_config

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
    ts_aft = prep_ts_aft(genos, samp_sizes)

    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in centers:
        if sweep in ["hard", "soft"]:
            if snps[center][2] == 2:  # Check for mut type of 2
                win_idxs = get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window
        else:
            if center == centers[int(len(centers) / 2)]:
                win_idxs = get_window_idxs(center, win_size)
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


def worker(
    in_vcf, samp_sizes, win_size, missingness, freq_inc_thr, verbose=False,
):
    benchmark = True
    try:
        id = get_rep_id(in_vcf)
        sweep = get_sweep(in_vcf)
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
        yaml_data = read_config(ua.yaml_file)
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
