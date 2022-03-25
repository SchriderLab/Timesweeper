import argparse as ap
import multiprocessing as mp
import os
import pickle
from glob import glob
from itertools import cycle
import logging
import numpy as np
from numpy.random import default_rng

import timesweeper as ts
from timesweeper import read_config
from utils import hap_utils as hu
from utils import snp_utils as su
import sys

rng = default_rng()

logging.basicConfig()
logger = logging.getLogger("make_training_feats")
logger.setLevel("INFO")


def add_missingness(data, m_rate, nan_val=-5):
    missing = rng.binomial(1, m_rate, data.shape)
    data[missing == 1] = nan_val

    return data


def get_aft_central_window(snps, genos, samp_sizes, win_size, sweep, missingness):
    """
    Iterates through windows of MAF time-series matrix and gets the central window.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP.
        genos (allel.GenotypeArray): Genotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        missingness (float): Parameter of binomial distribution to pull missingness from.

    Returns:
        np.arr: The central-most window, either based on mutation type or closest to half size of chrom.
    """
    ts_aft = ts.prep_ts_aft(genos, samp_sizes)

    # Iterate over SNP windows and predict
    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in centers:
        if sweep in ["hard", "soft"]:
            if snps[center][2] == 2:  # Check for mut type of 2
                win_idxs = ts.get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window
        elif sweep == "neut":
            if center == centers[int(len(centers) / 2)]:
                win_idxs = ts.get_window_idxs(center, win_size)
                window = ts_aft[:, win_idxs]
                center_aft = window

    missing_center_aft = add_missingness(center_aft, m_rate=missingness)

    return missing_center_aft


def parse_ua():
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
        "-m",
        "--missingness",
        metavar="MISSINGNESS",
        dest="missingness",
        type=float,
        required=False,
        default=0.0,
        help="Missingness rate in range of [0,1], used as the parameter of a binomial distribution for randomly removing known values.",
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
        help="Number of individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling points.",
        required=True,
        nargs="+",
        type=int,
    )
    cli_parser.add_argument(
        "-p",
        "--ploidy",
        dest="ploidy",
        help="Ploidy of organism being sampled.",
        default="2",
        type=int,
    )

    return uap.parse_args()


def worker(in_vcf, samp_sizes, win_size, ploidy, missingness, benchmark=True):
    try:
        id = ts.get_rep_id(in_vcf)
        sweep = ts.get_sweep(in_vcf)
        vcf = su.read_vcf(in_vcf, benchmark)

        # aft
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


def main():
    ua = parse_ua()
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

    work_args = zip(
        glob(f"{work_dir}/vcfs/*/*/merged.vcf", recursive=True),
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([ploidy]),
        cycle([ua.missingness]),
    )

    pool = mp.Pool(threads)
    work_res = pool.starmap(worker, work_args, chunksize=10)

    # Save this way so that if a single piece of data needs to be inspected/plotted it's always identifiable
    pickle_dict = {}
    for res in work_res:
        if res:
            rep, sweep, aft, hfs = res
            if sweep not in pickle_dict.keys():
                pickle_dict[sweep] = {}

            pickle_dict[sweep][rep] = {}
            pickle_dict[sweep][rep]["aft"] = aft

    # with open(, "w") as pklfile:
    pickle.dump(pickle_dict, open(f"{work_dir}/training_data.pkl", "wb"))


if __name__ == "__main__":
    main()
