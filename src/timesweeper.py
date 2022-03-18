import argparse as ap
import logging
import os

import allel
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.models import load_model
from tqdm import tqdm

from frequency_increment_test import fit
from utils import hap_utils as hu
from utils import snp_utils as su

import sys

logging.basicConfig()
logger = logging.getLogger("timesweeper")
logger.setLevel("INFO")


def get_sweep(filepath):
    """Grabs the sweep label from filepaths for easy saving."""
    for sweep in ["neut", "hard", "soft"]:
        if sweep in filepath:
            return sweep


def get_rep_id(filepath):
    """Searches path for integer, uses as id for replicate."""
    for i in filepath.split("/"):
        try:
            int(i)
            return i
        except ValueError:
            continue


def prep_ts_afs(genos, samp_sizes):
    """
    Iterates through timepoints and creates MAF feature matrices.

    Args:
        genos (allel.GenotypeArray): Genotype array containing all timepoints.
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.

    Returns:
        np.arr: MAF array to use for predictions. Shape is (timepoints, MAF).
    """
    # Prep genos into time-series format and calculate MAFs
    ts_genos = su.split_arr(genos, samp_sizes)
    min_alleles = su.get_minor_alleles(ts_genos)
    ts_mafs = []
    for timepoint in ts_genos:
        _genos = []
        _genotypes = allel.GenotypeArray(timepoint).count_alleles(
            max_allele=min_alleles.max()
        )

        for snp, min_allele_idx in zip(_genotypes, min_alleles):
            maf = su.calc_mafs(snp, min_allele_idx)
            _genos.append(maf)

        ts_mafs.append(_genos)

    return np.stack(ts_mafs)


def run_afs_windows(snps, genos, samp_sizes, win_size, model):
    """
    Iterates through windows of MAF time-series matrix and predicts using NN.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP. Contains mut only if benchmarking == True.
        genos (allel.GenotypeArray): Genotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        model (Keras.model): Keras model to use for prediction.

    Returns:
        dict: Prediction values in the form of dict[snps[center]]
        np.arr: the central-most window, either based on mutation type or closest to half size of chrom.
    """
    ts_afs = prep_ts_afs(genos, samp_sizes)

    # Iterate over SNP windows and predict
    results_dict = {}
    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in tqdm(centers, desc="Predicting on AFS windows"):
        win_idxs = get_window_idxs(center, win_size)
        window = ts_afs[:, win_idxs]

        probs = model.predict(np.expand_dims(window, 0))
        results_dict[snps[center]] = probs

    return results_dict


def run_fit_windows(snps, genos, samp_sizes, win_size, gens):
    """
    Iterates through windows of MAF time-series matrix and predicts using NN.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP. Contains mut only if benchmarking == True.
        genos (allel.GenotypeArray): Genotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        gens (list[int]): List of generations that were sampled.

    Returns:
        dict: P values from FIT.
    """
    ts_afs = prep_ts_afs(genos, samp_sizes)
    results_dict = {}
    buffer = int(win_size / 2)
    for idx in tqdm(range(buffer, len(snps) - buffer), desc="Calculating FIT values"):
        results_dict[snps[idx]] = fit(list(ts_afs[:, idx]), gens)  # tval, pval

    return results_dict


def run_hfs_windows(snps, haps, samp_sizes, win_size, model):
    """
    Iterates through windows of MAF time-series matrix and predicts using NN.

    Args:
        snps (list[tup(chrom, pos,  mut)]): Tuples of information for each SNP. Contains mut only if benchmarking == True.
        haps (np.arr): Haplotypes of all samples. 
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.
        win_size (int): Number of SNPs to use for each prediction. Needs to match how NN was trained.
        model (Keras.model): Keras model to use for prediction.

    Returns:
        dict: Prediction values in the form of dict[snps[center]]
        np.arr: the central-most window, either based on mutation type or closest to half size of chrom.
    """
    results_dict = {}
    buffer = int(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    for center in tqdm(centers, desc="Predicting on HFS windows"):
        win_idxs = get_window_idxs(center, win_size)
        window = np.swapaxes(haps[win_idxs, :], 0, 1)
        str_window = hu.haps_to_strlist(window)
        hfs = hu.getTSHapFreqs(str_window, samp_sizes)

        win_idxs = get_window_idxs(center, win_size)
        window = np.swapaxes(haps[win_idxs, :], 0, 1)
        str_window = hu.haps_to_strlist(window)
        # For plotting
        probs = model.predict(np.expand_dims(hfs, 0))
        results_dict[snps[center]] = probs

    return results_dict


def get_window_idxs(center_idx, win_size):
    """
    Gets the win_size number of snps around a central snp.

    Args:
        center_idx (int): Index of the central SNP to use for the window.
        win_size (int): Size of window to use around the SNP, optimally odd number.

    Returns:
        list: Indices of all SNPs to grab for the feature matrix.
    """
    half_window = int(win_size / 2)
    return list(range(center_idx - half_window, center_idx + half_window + 1))


def load_nn(model_path, summary=False):
    """
    Loads the trained Keras network.

    Args:
        model_path (str): Path to Keras model.
        summary (bool, optional): Whether to print out model summary or not. Defaults to False.

    Returns:
        Keras.model: Trained Keras model to use for prediction.
    """
    model = load_model(model_path)
    if summary:
        print(model.summary())

    return model


def write_fit(fit_dict, outfile, benchmark):
    """
    Writes FIT predictions to file.

    Args:
        fit_dict (dict): FIT p values and SNP information.
        outfile (str): File to write results to.
    """
    inv_pval = [1 - i[1] for i in fit_dict.values()]
    if benchmark:
        chroms, bps, mut_type = zip(*fit_dict.keys())
        predictions = pd.DataFrame(
            {"Chrom": chroms, "BP": bps, "Mut Type": mut_type, "Inv pval": inv_pval}
        )
    else:
        chroms, bps = zip(*fit_dict.keys())
        predictions = pd.DataFrame({"Chrom": chroms, "BP": bps, "Inv pval": inv_pval})

    predictions.sort_values(["Chrom", "BP"], inplace=True)
    predictions.to_csv(
        os.path.join(outfile), header=True, index=False, sep="\t",
    )


def write_preds(results_dict, outfile, benchmark):
    """
    Writes NN predictions to file.

    Args:
        results_dict (dict): SNP NN prediction scores.
        outfile (str): File to write results to.
    """
    lab_dict = {0: "Neut", 1: "Hard", 2: "Soft"}
    if benchmark:
        chroms, bps, mut_type = zip(*results_dict.keys())
    else:
        chroms, bps = zip(*results_dict.keys())

    neut_scores = [i[0][0] for i in results_dict.values()]
    hard_scores = [i[0][1] for i in results_dict.values()]
    soft_scores = [i[0][2] for i in results_dict.values()]

    classes = [lab_dict[np.argmax(i, axis=1)[0]] for i in results_dict.values()]

    if benchmark:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Mut Type": mut_type,
                "Class": classes,
                "Neut Score": neut_scores,
                "Hard Score": hard_scores,
                "Soft Score": soft_scores,
            }
        )
    else:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Class": classes,
                "Neut Score": neut_scores,
                "Hard Score": hard_scores,
                "Soft Score": soft_scores,
            }
        )

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    if not os.path.exists(outfile):
        predictions.to_csv(outfile, header=True, index=False, sep="\t")
    else:
        predictions.to_csv(outfile, mode="a", header=False, index=False, sep="\t")


def add_file_label(filename, label):
    """Injects a model identifier to the outfile name."""
    splitfile = filename.split(".")
    newname = f"{''.join(splitfile[:-1])}_{label}.{splitfile[-1]}"
    return newname


def parse_ua():
    uap = ap.ArgumentParser(
        description="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window."
    )
    uap.add_argument(
        "-i",
        "--input-vcf",
        dest="input_vcf",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )
    uap.add_argument(
        "--benchmark",
        dest="benchmark",
        action="store_true",
        help="If testing on simulated data and would like to report the mutation \
            type stored by SLiM during outputVCFSample, use this flag. \
            Otherwise the mutation type will not be looked for in the VCF entry nor reported with results.",
        required=False,
    )
    uap.add_argument(
        "--afs-model",
        dest="afs_model",
        help="Path to Keras2-style saved model to load for AFS prediction.",
        required=True,
    )
    uap.add_argument(
        "--hfs-model",
        dest="hfs_model",
        help="Path to Keras2-style saved model to load for HFS prediction.",
        required=True,
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
    cli_parser.add_argument(
        "-w",
        "--work-dir",
        metavar="WORKING_DIR",
        dest="work_dir",
        type=str,
        help="Working directory for workflow, should be identical to previous steps.",
    )
    cli_parser.add_argument(
        "--years-sampled",
        required=False,
        type=int,
        nargs="+",
        dest="years_sampled",
        default=None,
        help="Years BP (before 1950) that samples are estimated to be from. Only used for FIT calculations, and is optional if you don't care about those.",
    )
    cli_parser.add_argument(
        "--gen-time",
        required=False,
        type=int,
        dest="gen_time",
        default=None,
        help="Generation time to multiply years_sampled by. Similarly to years_sampled, only used for FIT calculation and is optional.",
    )
    return uap.parse_args()


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


def main():
    ua = parse_ua()
    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        work_dir, input_vcf, samp_sizes, ploidy, outdir, afs_model, hfs_model = (
            yaml_data["work dir"],
            ua.input_vcf,
            yaml_data["sample sizes"],
            yaml_data["ploidy"],
            yaml_data["work dir"],
            load_nn(ua.afs_model),
            load_nn(ua.hfs_model),
        )

        # If you're doing simple sims you probably aren't calculating years out
        if "years sampled" in yaml_data:
            years_sampled = yaml_data["years sampled"]
        else:
            years_sampled = None

        if "gen time" in yaml_data:
            gen_time = yaml_data["gen time"]
        else:
            gen_time = None

    elif ua.config_format == "cli":
        (
            input_vcf,
            samp_sizes,
            years_sampled,
            gen_time,
            ploidy,
            work_dir,
            afs_model,
            hfs_model,
        ) = (
            ua.input_vcf,
            ua.samp_sizes,
            ua.years_sampled,
            ua.gen_time,
            ua.ploidy,
            ua.work_dir,
            load_nn(ua.afs_model),
            load_nn(ua.hfs_model),
        )

    outdir = f"{work_dir}/timesweeper_output"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    win_size = 51  # Must be consistent with training data

    vcf_iter = su.get_vcf_iter(ua.input_vcf, ua.benchmark)
    for id, chunk in enumerate(vcf_iter):
        chunk = chunk[0]  # Why you gotta do me like that, skallel?
        logger.info(f"Predicting on chunk {id}")

        # AFS
        genos, snps = su.vcf_to_genos(chunk, ua.benchmark)
        afs_predictions = run_afs_windows(snps, genos, samp_sizes, win_size, afs_model)

        write_preds(afs_predictions, f"{outdir}/afs_preds.csv", ua.benchmark)

        # HFS
        haps, snps = su.vcf_to_haps(chunk, ua.benchmark)
        hfs_predictions = run_hfs_windows(
            snps, haps, [ploidy * i for i in samp_sizes], win_size, hfs_model
        )

        write_preds(hfs_predictions, f"{outdir}/hfs_preds.csv", ua.benchmark)

        if years_sampled and gen_time:
            # FIT
            gens = [i * gen_time for i in years_sampled]
            genos, snps = su.vcf_to_genos(input_vcf, ua.benchmark)
            fit_predictions = run_fit_windows(snps, genos, samp_sizes, win_size, gens)
            print(fit_predictions)
            write_fit(fit_predictions, f"{outdir}/fit_preds.csv", ua.benchmark)
        else:
            logger.info(
                "Cannot calculate FIT, years sampled and gen time not supplied."
            )


if __name__ == "__main__":
    main()
