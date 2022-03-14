import argparse as ap
import multiprocessing as mp
import os
import pickle
from glob import glob
from itertools import cycle
import logging
import numpy as np

import timesweeper as ts
from timesweeper import read_config
from utils import hap_utils as hu
from utils import snp_utils as su
import sys


logging.basicConfig()
logger = logging.getLogger("make_training_feats")
logger.setLevel("INFO")


def get_afs_central_window(snps, genos, samp_sizes, win_size, sweep):
    """
    Iterates through windows of MAF time-series matrix and gets the central window.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP.
        genos (allel.GenotypeArray): Genotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.

    Returns:
        np.arr: The central-most window, either based on mutation type or closest to half size of chrom.
    """
    ts_afs = ts.prep_ts_afs(genos, samp_sizes)

    # Iterate over SNP windows and predict
    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in centers:
        if sweep in ["hard", "soft"]:
            if snps[center][2] == 2:  # Check for mut type of 2
                win_idxs = ts.get_window_idxs(center, win_size)
                window = ts_afs[:, win_idxs]
                center_afs = window
        elif sweep == "neut":
            if center == centers[int(len(centers) / 2)]:
                win_idxs = ts.get_window_idxs(center, win_size)
                window = ts_afs[:, win_idxs]
                center_afs = window

    return center_afs


def get_hfs_central_window(snps, haps, samp_sizes, win_size, sweep):
    """
    Iterates through windows of MAF time-series matrix and gets the central window.

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
                win_idxs = ts.get_window_idxs(center, win_size)
                window = np.swapaxes(haps[win_idxs, :], 0, 1)
                str_window = hu.haps_to_strlist(window)
                central_hfs = hu.getTSHapFreqs(str_window, samp_sizes)
        elif sweep == "neut":
            if center == centers[int(len(centers) / 2)]:
                win_idxs = ts.get_window_idxs(center, win_size)
                window = np.swapaxes(haps[win_idxs, :], 0, 1)
                str_window = hu.haps_to_strlist(window)
                central_hfs = hu.getTSHapFreqs(str_window, samp_sizes)

    return central_hfs


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


def worker(in_vcf, samp_sizes, win_size, ploidy, benchmark=True):
    try:
        id = ts.get_rep_id(in_vcf)
        sweep = ts.get_sweep(in_vcf)

        # AFS
        genos, snps = su.vcf_to_genos(in_vcf, benchmark)
        central_afs = get_afs_central_window(snps, genos, samp_sizes, win_size, sweep)

        # HFS
        haps, snps = su.vcf_to_haps(in_vcf, benchmark)
        central_hfs = get_hfs_central_window(
            snps, haps, [ploidy * i for i in samp_sizes], win_size, sweep
        )
        return id, sweep, central_afs, central_hfs

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
        glob(f"{work_dir}/vcfs/*/*/merged.vcf"),
        cycle([samp_sizes]),
        cycle([win_size]),
        cycle([ploidy]),
    )

    pool = mp.Pool(threads)
    work_res = pool.starmap(worker, work_args, chunksize=10)

    # Save this way so that if a single piece of data needs to be inspected/plotted it's always identifiable
    pickle_dict = {}
    for res in work_res:
        print(res)
        if res:
            rep, sweep, afs, hfs = res
            if sweep not in pickle_dict.keys():
                pickle_dict[sweep] = {}

            pickle_dict[sweep][rep] = {}
            pickle_dict[sweep][rep]["afs"] = afs
            pickle_dict[sweep][rep]["hfs"] = hfs

    # with open(, "w") as pklfile:
    pickle.dump(pickle_dict, open(f"{work_dir}/training_data.pkl", "wb"))


if __name__ == "__main__":
    main()
