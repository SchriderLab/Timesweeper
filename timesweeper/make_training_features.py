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
from .utils.hap_utils import haps_to_strlist, getTSHapFreqs

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


def get_hft_central_window(snps, haps, samp_sizes, win_size, sweep):
    """
    Iterates through windows of MAF time-series matrix and gets the central window.
    Does not have as many utility functions as AFT such as missingness and variable sorting methods.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP.
        haps (np.arr): Haplotypes of all samples.
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
    Returns:
        np.arr: The central-most window, either based on mutation type or closest to half size of chrom.
    """
    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in centers:
        if sweep in ["hard", "soft"]:
            if snps[center][2] == 2:
                win_idxs = get_window_idxs(center, win_size)
                window = np.swapaxes(haps[win_idxs, :], 0, 1)
                str_window = haps_to_strlist(window)
                central_hfs = getTSHapFreqs(str_window, samp_sizes)
        elif sweep == "neut":
            if center == centers[int(len(centers) / 2)]:
                win_idxs = get_window_idxs(center, win_size)
                window = np.swapaxes(haps[win_idxs, :], 0, 1)
                str_window = haps_to_strlist(window)
                central_hfs = getTSHapFreqs(str_window, samp_sizes)

    return central_hfs


def check_freq_increase(ts_afs, min_increase=0.25):
    """Quick test to make sure a given example is increasing properly, can be used to filter training."""
    center = int(ts_afs.shape[0] / 2)
    if ts_afs[-1, center] - ts_afs[0, center] >= min_increase:
        return True
    else:
        return False


def worker(
    in_vcf,
    samp_sizes,
    win_size,
    missingness,
    freq_inc_thr,
    hft=False,
    ploidy=1,
    verbose=False,
):
    benchmark = True
    try:
        id = get_rep_id(in_vcf)
        sweep = get_sweep(in_vcf)
        vcf = su.read_vcf(in_vcf, benchmark)

        genos, snps = su.vcf_to_genos(vcf, benchmark)

        if hft:
            central_hft = get_hft_central_window(
                snps,
                genos,
                [ploidy * i for i in samp_sizes],
                win_size,
                sweep,
            )

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
            return id, sweep, central_aft, central_hft

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
        work_dir, samp_sizes, ploidy, threads = (
            yaml_data["work dir"],
            yaml_data["sample sizes"],
            yaml_data["ploidy"],
            ua.threads,
        )
    elif ua.config_format == "cli":
        work_dir, samp_sizes, ploidy, threads = (
            ua.work_dir,
            ua.samp_sizes,
            ua.ploidy,
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
        cycle([ua.hft]),
        cycle([int(ploidy)]),
        cycle([ua.verbose]),
    )
    print(len(list(work_args)))

    pool = mp.Pool(threads)
    if ua.no_progress:
        work_res = pool.starmap(
            worker,
            work_args,
            chunksize=4,
        )

    else:
        work_res = pool.starmap(
            worker,
            tqdm(work_args, desc="Condensing training data", total=len(filelist)), 
            chunksize=4,
        )

    # Save this way so that if a single piece of data needs to be inspected/plotted it's always identifiable
    pickle_dict = {}
    for res in work_res:
        if res:
            rep, sweep, afs, hfs = res
            if sweep not in pickle_dict.keys():
                pickle_dict[sweep] = {}

            pickle_dict[sweep][rep] = {}
            pickle_dict[sweep][rep]["afs"] = afs
            if ua.hft:
                pickle_dict[sweep][rep]["hfs"] = hfs

    pickle.dump(pickle_dict, open(ua.outfile, "wb"))
