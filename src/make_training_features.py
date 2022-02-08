import argparse as ap
import os

import numpy as np
import yaml

import timesweeper as ts
from utils import hap_utils as hu
from utils import snp_utils as su


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
            if snps[center][2] == 2:
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
        description="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window."
    )
    subparsers = uap.add_subparsers(dest="config_format")
    subparsers.required = True
    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        "-y",
        "--yaml",
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    cli_parser = subparsers.add_parser("cli")
    cli_parser.add_argument(
        "-i",
        "--input-vcf",
        dest="input_vcf",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
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


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


def get_sweep(filepath):
    """Grabs the sweep label from filepaths for easy saving."""
    for sweep in ["neut", "hard", "soft"]:
        if sweep in filepath:
            return sweep


def main():
    ua = parse_ua()
    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        input_vcf, samp_sizes, ploidy = (
            yaml_data["vcf"],
            yaml_data["sample_sizes"],
            yaml_data["ploidy"],
        )
    elif ua.config_format == "cli":
        input_vcf, samp_sizes, ploidy = (
            ua.input_vcf,
            ua.samp_sizes,
            ua.ploidy,
        )

    indir = os.path.dirname(input_vcf)
    win_size = 51  # Must be consistent with training data
    sweep = get_sweep(input_vcf)

    # AFS
    genos, snps = su.vcf_to_genos(input_vcf)
    central_afs = get_afs_central_window(snps, genos, samp_sizes, win_size, sweep)
    np.save(f"{indir}/afs_center.npy", central_afs)
    # HFS
    haps, snps = su.vcf_to_haps(input_vcf)
    central_hfs = get_hfs_central_window(
        snps, haps, [ploidy * i for i in samp_sizes], win_size, sweep
    )
    np.save(f"{indir}/hfs_center.npy", central_hfs)


if __name__ == "__main__":
    main()
