import logging
import multiprocessing as mp
import os
import pickle
import sys
from glob import glob
from itertools import cycle
import random
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from .find_sweeps_vcf import prep_ts_aft, get_window_idxs
from .utils import snp_utils as su
from .utils.gen_utils import get_rep_id, read_config, get_scenario_from_filename
from .utils.hap_utils import haps_to_strlist, getTSHapFreqs

logging.basicConfig()
logger = logging.getLogger("make_training_feats")
logger.setLevel("INFO")

import warnings

warnings.filterwarnings("error")


def add_missingness(data, m_rate, nan_val=-1):
    rng = default_rng(np.random.seed(int.from_bytes(os.urandom(4), byteorder="little")))
    missing = rng.binomial(1, m_rate, data.shape)
    data[missing == 1] = nan_val

    return data


def get_aft_central_window(snps, genos, samp_sizes, win_size, missingness, mut_types):
    """
    Check for whether a non-control mutation type is present. If not, return the central-most mutation.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP.
        genos (allel.GenotypeArray): Genotypes of all samples.
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        scenario (str): Entry from the scenarios config option.
        missingness (float): Parameter of binomial distribution to pull missingness from.
        mut_types (list[int]): List of mutation types that are not considered the "control" case.
    Returns:
        np.arr: The central-most window, either based on mutation type or closest to half size of chrom.
        float: Selection coefficient.
    """
    ts_aft = prep_ts_aft(genos, samp_sizes)

    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)

    center_idx = int(len(centers) / 2)
    for center in centers:
        if snps[center][2] in mut_types:
            center_idx = center
            break
        else:
            pass

    win_idxs = get_window_idxs(center_idx, win_size)
    window = ts_aft[:, win_idxs]
    center_aft = window
    sel_coeff = snps[center_idx][3]
    missing_center_aft = add_missingness(center_aft, m_rate=missingness)

    return missing_center_aft, sel_coeff


def get_hft_central_window(snps, haps, samp_sizes, win_size, mut_types):
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

    center_idx = int(len(centers) / 2)
    for center in centers:
        if snps[center][2] in mut_types:
            center_idx = center
            break
        else:
            pass

    win_idxs = get_window_idxs(center_idx, win_size)
    window = np.swapaxes(haps[win_idxs, :], 0, 1)
    str_window = haps_to_strlist(window)
    central_hfs = getTSHapFreqs(str_window, samp_sizes)
    sel_coeff = snps[center_idx][3]

    return central_hfs, sel_coeff


def check_freq_increase(ts_afs, min_increase=0.25):
    """Quick test to make sure a given example is increasing properly, can be used to filter training."""
    center = int(ts_afs.shape[0] / 2)
    if ts_afs[-1, center] - ts_afs[0, center] >= min_increase:
        return True
    else:
        return False


def aft_worker(
    in_vcf, mut_types, scenarios, samp_sizes, win_size, missingness, verbose=False,
):
    benchmark = True
    try:
        id = get_rep_id(in_vcf)
        scenario = get_scenario_from_filename(in_vcf, scenarios)
        vcf = su.read_vcf(in_vcf, benchmark)
        genos, snps = su.vcf_to_genos(vcf, benchmark)

        central_aft, sel_coeff = get_aft_central_window(
            snps, genos, samp_sizes, win_size, missingness, mut_types
        )

        return id, scenario, central_aft, sel_coeff

    except UserWarning as Ue:
        print(Ue)
        return None

    except Exception as e:
        if verbose:
            logger.warning(f"Could not process {in_vcf}")
            logger.warning(f"Exception: {e}")
            sys.stdout.flush()
            sys.stderr.flush()
        return None


def hft_worker(
    in_vcf, mut_types, scenarios, samp_sizes, win_size, ploidy=1, verbose=False,
):
    benchmark = True
    try:
        id = get_rep_id(in_vcf)
        scenario = get_scenario_from_filename(in_vcf, scenarios)
        vcf = su.read_vcf(in_vcf, benchmark)
        haps, snps = su.vcf_to_haps(vcf, benchmark)

        central_hft, sel_coeff = get_hft_central_window(
            snps, haps, [ploidy * i for i in samp_sizes], win_size, mut_types,
        )

        return id, scenario, central_hft, sel_coeff

    except UserWarning as Ue:
        # print(Ue)
        return None

    except Exception as e:
        if verbose:
            logger.warning(f"Could not process {in_vcf}")
            logger.warning(f"Exception: {e}")
            sys.stdout.flush()
            sys.stderr.flush()
        return None


def main(ua):
    yaml_data = read_config(ua.yaml_file)
    scenarios, mut_types, work_dir, samp_sizes, ploidy, win_size, threads = (
        yaml_data["scenarios"],
        yaml_data["mut types"],
        yaml_data["work dir"],
        yaml_data["sample sizes"],
        yaml_data["ploidy"],
        yaml_data["win_size"],
        ua.threads,
    )

    filelist = glob(f"{work_dir}/vcfs/*/*/merged.vcf", recursive=True)[:10]

    aft_work_args = zip(
        filelist,
        cycle([mut_types]),
        cycle([scenarios]),
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([ua.missingness]),
        cycle([ua.verbose]),
    )
    hft_work_args = zip(
        filelist,
        cycle([mut_types]),
        cycle([scenarios]),
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([int(ploidy)]),
        cycle([ua.verbose]),
    )

    pool = mp.Pool(threads)
    if ua.no_progress:
        aft_work_res = pool.starmap(aft_worker, aft_work_args, chunksize=4,)

        if ua.hft:
            hft_work_res = pool.starmap(hft_worker, hft_work_args, chunksize=4,)

        pool.close()
    else:
        aft_work_res = pool.starmap(
            aft_worker,
            tqdm(
                aft_work_args, desc="Formatting AFT training data", total=len(filelist),
            ),
            chunksize=4,
        )
        if ua.hft:
            hft_work_res = pool.starmap(
                hft_worker,
                tqdm(
                    hft_work_args,
                    desc="Formatting HFT training data",
                    total=len(filelist),
                ),
                chunksize=4,
            )
        pool.close()

    sys.exit(0)
    pickle_dict = {}
    for s in scenarios:
        pickle_dict[s] = {}

    for res in aft_work_res:
        if res:
            rep, scenario, aft, s = res

            pickle_dict[scenario][rep] = {}
            pickle_dict[scenario][rep]["aft"] = aft
            pickle_dict[scenario][rep]["sel_coeff"] = s

    if ua.hft:
        for res in hft_work_res:
            try:
                if res:
                    rep, scenario, hft, s = res
                    if rep in pickle_dict[scenario].keys():
                        pickle_dict[scenario][rep]["hft"] = hft
                        pickle_dict[scenario][rep]["sel_coeff"] = s

            except KeyError as e:
                print(e)
                print(res)
                pass

    with open(ua.outfile, "wb") as outfile:
        pickle.dump(pickle_dict, outfile)
