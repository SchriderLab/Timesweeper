import logging
import math
import multiprocessing as mp
import os
import pickle
import sys
from glob import glob
from itertools import cycle
from random import sample

import pandas as pd
import allel
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm

from timesweeper.utils import snp_utils as su
from timesweeper.utils.gen_utils import (get_rep_id,
                                         get_scenario_from_filename,
                                         read_config)
from timesweeper.utils.hap_utils import getTSHapFreqs, haps_to_strlist

logging.basicConfig()
logger = logging.getLogger("make_training_feats")
logger.setLevel("INFO")

import warnings

warnings.filterwarnings("error")


def draw_rand_center_offset(max_offset=700):
    rng = default_rng(np.random.seed(int.from_bytes(os.urandom(4), byteorder="little")))
    return int(rng.uniform(-max_offset, max_offset, 1)[0])


def add_missingness(data, m_rate, nan_val=-1):
    rng = default_rng(np.random.seed(int.from_bytes(os.urandom(4), byteorder="little")))
    missing = rng.binomial(1, m_rate, data.shape)
    data[missing == 1] = nan_val

    return data


def check_freq_increase(ts_afs, min_increase=0.25):
    """Quick test to make sure a given example is increasing properly, can be used to filter training."""
    center = int(ts_afs.shape[0] / 2)
    if ts_afs[-1, center] - ts_afs[0, center] >= min_increase:
        return True
    else:
        return False


def subsample_tps(inds_per_tp, og_tps, num_sampled_tps):
    """Subsamples timepoints during VCF by sampling samples list."""
    samples_list = []
    if num_sampled_tps == 1:
        t = og_tps + 1  # VCFs merged samples are 1-indexed, get last
        samples_list.extend([f"{t}:i{i}" for i in range(inds_per_tp)])
    elif num_sampled_tps == 2:
        # This covers the first sampling point
        samples_list.extend([f"i{i}" for i in range(inds_per_tp)])
        t = og_tps + 1  # VCFs merged samples are 1-indexed, get last
        samples_list.extend([f"{t}:i{i}" for i in range(inds_per_tp)])
    else:
        tps = np.linspace(1, og_tps, num_sampled_tps, dtype=int)
        # This covers the first sampling point
        samples_list.extend([f"i{i}" for i in range(inds_per_tp)])
        # This gets the rest
        for t in tps[1:]:
            samples_list.extend([f"{t}:i{i}" for i in range(inds_per_tp)])

    return samples_list


def subsample_inds(inds_per_tp, subsample_size, num_tps):
    """Subsamples individuals from each timepoint to re-use simulations."""
    samples_list = []
    samples_list.extend(
        [f"i{i}" for i in np.random.choice(inds_per_tp, subsample_size, replace=False)]
    )
    for s in range(2, num_tps + 1):
        samples_list.extend(
            [
                f"{s}:i{i}"
                for i in np.random.choice(inds_per_tp, subsample_size, replace=False)
            ]
        )
    return samples_list


def get_window_idxs(center_idx, win_size):
    """
    Gets the win_size number of snps around a central snp.

    Args:
        center_idx (int): Index of the central SNP to use for the window.
        win_size (int): Size of window to use around the SNP, optimally odd number.

    Returns:
        list: Indices of all SNPs to grab for the feature matrix.
    """
    half_window = math.floor(win_size / 2)
    return list(range(center_idx - half_window, center_idx + half_window + 1))


def prep_ts_aft(genos, samp_sizes):
    """
    Iterates through timepoints and creates MAF feature matrices.

    Args:
        genos (allel.GenotypeArray): Genotype array containing all timepoints.
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.

    Returns:
        np.arr: MAF array to use for predictions. Shape is (timepoints, MAF).
    """
    # Prep genos into time-series format and calculate Maft
    ts_genos = su.split_arr(genos, samp_sizes)

    min_alleles, first_genos, last_genos = su.get_vel_minor_alleles(
        ts_genos, np.max(genos)
    )

    ts_maft = []
    for timepoint in ts_genos:
        _genos = []
        _genotypes = allel.GenotypeArray(timepoint).count_alleles(
            max_allele=min_alleles.max()
        )

        for snp, min_allele_idx in zip(_genotypes, min_alleles):
            maf = su.calc_maft(snp, min_allele_idx)
            _genos.append(maf)

        ts_maft.append(_genos)

    return np.stack(ts_maft)


def get_aft_central_window(
    snps, genos, samp_sizes, win_size, missingness, mut_types, offset
):
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

    try:
        for center in centers:
            if snps[center][2] in mut_types:
                center_idx = center
                break
            else:
                pass
    except:
        pass

    sel_coeff = snps[center_idx][3]

    if offset:
        rng = default_rng(
            np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        )
        if rng.uniform(0, 3, 1)[0] > 2:
            rand_offset = draw_rand_center_offset()
            center_idx += rand_offset
        else:
            rand_offset = 0
    else:
        rand_offset = 0

    win_idxs = get_window_idxs(center_idx, win_size)

    window = ts_aft[:, win_idxs]
    center_aft = window

    missing_center_aft = add_missingness(
        center_aft, m_rate=missingness
    )  # If no missingness, will just return

    return missing_center_aft, sel_coeff, rand_offset


def get_hft_central_window(snps, haps, samp_sizes, win_size, mut_types, offset):
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

    try:
        for center in centers:
            if snps[center][2] in mut_types:
                center_idx = center
                break
            else:
                pass
    except:
        pass

    sel_coeff = snps[center_idx][3]

    if offset:
        rng = default_rng(
            np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        )
        if rng.uniform(1, 3, 1)[0] > 2:
            rand_offset = draw_rand_center_offset()
            center_idx += rand_offset

        else:
            rand_offset = 0
    else:
        rand_offset = 0

    win_idxs = get_window_idxs(center_idx, win_size)
    window = np.swapaxes(haps[win_idxs, :], 0, 1)
    str_window = haps_to_strlist(window)
    central_hfs = getTSHapFreqs(str_window, samp_sizes)

    return central_hfs, sel_coeff, rand_offset


def aft_worker(
    in_vcf,
    mut_types,
    scenarios,
    samp_sizes,
    samps_list,
    win_size,
    offset,
    missingness,
    verbose=False,
    params=None,
):
    benchmark = True  # Want to get all the info we can from sims in training
    try:
        id = get_rep_id(in_vcf)
        scenario = get_scenario_from_filename(in_vcf, scenarios)

        vcf = su.read_vcf(in_vcf, samps_list, benchmark)
        genos, snps = su.vcf_to_genos(vcf, benchmark)

        central_aft, sel_coeff, rand_offset = get_aft_central_window(
            snps, genos, samp_sizes, win_size, missingness, mut_types, offset
        )
    
        if params is not None:
            sel_coeff = params[(params["rep"] == int(id)) & (params["sweep"] == scenario)]["selCoeff"].values[0]

        if "neut" not in scenario.lower() and sel_coeff == 0.0:
            raise Exception

        return id, scenario, central_aft, sel_coeff, rand_offset

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
    in_vcf,
    mut_types,
    scenarios,
    samp_sizes,
    samps_list,
    win_size,
    offset,
    ploidy=2,
    verbose=False,
    params=None,
):
    benchmark = True
    try:
        id = get_rep_id(in_vcf)
        scenario = get_scenario_from_filename(in_vcf, scenarios)

        vcf = su.read_vcf(in_vcf, samps_list, benchmark)
        haps, snps = su.vcf_to_haps(vcf, benchmark)

        central_hft, sel_coeff, rand_offset = get_hft_central_window(
            snps, haps, [ploidy * i for i in samp_sizes], win_size, mut_types, offset
        )

        if params is not None:
            sel_coeff = params[(params["rep"] == int(id)) & (params["sweep"] == scenario)]["selCoeff"].values[0]

        return id, scenario, central_hft, sel_coeff, rand_offset

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
    if ua.paramsfile:
        params = pd.read_csv(ua.paramsfile, sep="\t")
    else:
        params = None
    
    if ua.subsample_inds:
        if ua.subsample_tps:
            logger.error("Can't subsample both timepoints and individuals.")

        samps_list = subsample_inds(samp_sizes[0], ua.subsample_inds, len(samp_sizes))
    else:
        samps_list = None

    if ua.subsample_tps:
        if not ua.og_tps:
            logger.error(
                "Must provide original simulation number of tps in order to properly subset."
            )

        samps_list = subsample_tps(samp_sizes[0], ua.og_tps, ua.subsample_tps)

    if ua.allow_shoulders:
        offset = int(ua.allow_shoulders)
    else:
        offset = 0

    filelist = glob(f"{work_dir}/vcfs/*/*/merged.vcf", recursive=True)

    aft_work_args = zip(
        filelist,
        cycle([mut_types]),
        cycle([scenarios]),
        cycle([samp_sizes]),
        cycle([samps_list]),
        cycle([win_size]),
        cycle([offset]),
        cycle([ua.missingness]),
        cycle([ua.verbose]),
        cycle([params]),
    )
    hft_work_args = zip(
        filelist,
        cycle([mut_types]),
        cycle([scenarios]),
        cycle([samp_sizes]),
        cycle([samps_list]),
        cycle([win_size]),
        cycle([offset]),
        cycle([int(ploidy)]),
        cycle([ua.verbose]),
        cycle([params]),
    )
    print("[INFO] Starting run")
    debug = False
    if debug:
        aft_work_res = []
        for i in tqdm(aft_work_args, desc="AFT", total=len(filelist)):
            aft_work_res.append(aft_worker(*i))
        
        hft_work_res = []
        for i in tqdm(hft_work_args, desc="HFT", total=len(filelist)):
            hft_work_res.append(hft_worker(*i))

    else:
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
                    aft_work_args,
                    desc="Formatting AFT training data",
                    total=len(filelist),
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

    pickle_dict = {}
    for s in scenarios:
        pickle_dict[s] = {}

    for res in aft_work_res:
        if res:
            rep, scenario, aft, s, off = res

            pickle_dict[scenario][rep] = {}
            pickle_dict[scenario][rep]["aft"] = aft
            pickle_dict[scenario][rep]["sel_coeff"] = s
            pickle_dict[scenario][rep]["center_offset"] = off

    if ua.hft:
        for res in hft_work_res:
            try:
                if res:
                    rep, scenario, hft, s, off = res
                    if rep in pickle_dict[scenario].keys():
                        pickle_dict[scenario][rep]["hft"] = hft
                        pickle_dict[scenario][rep]["sel_coeff"] = s
                        pickle_dict[scenario][rep]["center_offset"] = off

            except KeyError as e:
                print(e)
                print(res)
                pass

    with open(ua.outfile, "wb") as outfile:
        pickle.dump(pickle_dict, outfile)
