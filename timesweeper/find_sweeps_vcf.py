import logging
import math
import os

import allel
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from .utils.frequency_increment_test import fit
from .utils import snp_utils as su
from .utils.gen_utils import read_config, write_fit, write_preds
from .utils import hap_utils as hu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig()
logger = logging.getLogger("timesweeper")
logger.setLevel("INFO")


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
    min_alleles = su.get_vel_minor_alleles(ts_genos, np.max(genos))
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


def run_aft_windows(snps, genos, samp_sizes, win_size, model):
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
    ts_aft = prep_ts_aft(genos, samp_sizes)

    # Iterate over SNP windows and predict
    buffer = math.floor(win_size / 2)

    centers = range(buffer, len(snps) - buffer)
    left_edges = []
    right_edges = []
    data = []
    for center in tqdm(centers, desc="Predicting on AFT windows"):
        try:
            win_idxs = get_window_idxs(center, win_size)
            window = ts_aft[:, win_idxs]
            data.append(window)
            left_edges.append(snps[win_idxs[0]][1])
            right_edges.append(snps[win_idxs[-1]][1])

        except Exception as e:
            logger.warning(f"Center {snps[center]} raised error {e}")

    class_probs, sel_pred = model.predict(np.stack(data))

    results_dict = {}
    for center, class_probs, sel_pred, l_e, r_e in zip(
        centers, class_probs, sel_pred, left_edges, right_edges
    ):
        results_dict[snps[center]] = (class_probs, sel_pred, l_e, r_e)

    return results_dict


def run_hft_windows(snps, haps, ploidy, samp_sizes, win_size, model):
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
    buffer = math.floor(win_size / 2)
    centers = range(buffer, len(snps) - buffer)
    left_edges = []
    right_edges = []
    data = []
    for center in tqdm(centers, desc="Predicting on HFS windows"):
        try:
            win_idxs = get_window_idxs(center, win_size)
            window = np.swapaxes(haps[win_idxs, :], 0, 1)
            str_window = hu.haps_to_strlist(window)
            hfs = hu.getTSHapFreqs(str_window, [i * ploidy for i in samp_sizes])
            data.append(hfs)
            left_edges.append(snps[win_idxs[0]][1])
            right_edges.append(snps[win_idxs[-1]][1])

        except Exception as e:
            logger.warning(f"Center {snps[center]} raised error {e}")
            continue

    probs = model.predict(np.stack(data))

    results_dict = {}
    for center, prob, l_e, r_e in zip(centers, probs, left_edges, right_edges):
        results_dict[snps[center]] = (prob, l_e, r_e)

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
    ts_aft = prep_ts_aft(genos, samp_sizes)
    results_dict = {}
    buffer = int(win_size / 2)
    for idx in tqdm(range(buffer, len(snps) - buffer), desc="Calculating FIT values"):
        results_dict[snps[idx]] = fit(list(ts_aft[:, idx]), gens)  # tval, pval

    return results_dict


def run_fet_windows(genos, samp_sizes):
    ts_genos = su.split_arr(genos, samp_sizes)
    min_alleles = su.get_vel_minor_alleles(ts_genos, np.max(genos))

    #get count values for maj, min using snp utils
    #feed into fet 
    #collect fet values into fet_geno counts for all snps 
    #fet_pvals = list with len(snps)

    for timepoint in [0, -1]:
        _genotypes = allel.GenotypeArray(timepoint).count_alleles(
            max_allele=min_alleles.max()
        )
        min_allele_counts = []
        for snp, min_allele_idx in zip(_genotypes, min_alleles):

        fet_geno_counts.append(min_allele_counts)


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


def main(ua):
    if ua.config_format == "yaml":
        yaml_data = read_config(ua.yaml_file)
        (work_dir, samp_sizes, ploidy, win_size, outdir, aft_model) = (
            yaml_data["work dir"],
            yaml_data["sample sizes"],
            yaml_data["ploidy"],
            yaml_data["win_size"],
            ua.outdir,
            load_nn(ua.aft_model),
        )

        # If you're doing simple sims you probably aren't calculating years out
        if "gens sampled" in yaml_data:
            gens_sampled = yaml_data["gens sampled"]
        else:
            gens_sampled = None

    elif ua.config_format == "cli":
        (samp_sizes, gens_sampled, ploidy, win_size, work_dir, aft_model,) = (
            ua.samp_sizes,
            ua.gens_sampled,
            ua.ploidy,
            ua.win_size,
            ua.work_dir,
            load_nn(ua.aft_model),
        )

    outdir = ua.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Chunk and iterate for NN predictions to not take up too much space
    vcf_iter = su.get_vcf_iter(ua.input_vcf, ua.benchmark)
    for chunk_idx, chunk in enumerate(vcf_iter):
        chunk = chunk[0]  # Why you gotta do me like that, skallel?
        logger.info(f"Processing VCF chunk {chunk_idx}")

        # aft
        try:
            genos, snps = su.vcf_to_genos(chunk, ua.benchmark)
            aft_predictions = run_aft_windows(
                snps, genos, samp_sizes, win_size, aft_model
            )
            write_preds(aft_predictions, f"{outdir}/aft_preds.csv", ua.benchmark)

        except Exception as e:
            logger.error(f"Cannot process chunk {chunk_idx} using AFT due to {e}")

        # hft
        if ua.hft_model:
            try:
                haps, snps = su.vcf_to_haps(chunk, ua.benchmark)
                aft_predictions = run_hft_windows(
                    snps, haps, ploidy, samp_sizes, win_size, load_nn(ua.hft_model)
                )
                write_preds(aft_predictions, f"{outdir}/hft_preds.csv", ua.benchmark)

            except Exception as e:
                logger.error(f"Cannot process chunk {chunk_idx} using HFT due to {e}")

    if gens_sampled:
        vcf_iter = su.get_vcf_iter(ua.input_vcf, ua.benchmark)
        for chunk_idx, chunk in enumerate(vcf_iter):
            chunk = chunk[0]  # Why you gotta do me like that, skallel?

            # FIT
            # try:
            genos, snps = su.vcf_to_genos(chunk, ua.benchmark)
            fit_predictions = run_fit_windows(
                snps, genos, samp_sizes, win_size, gens_sampled
            )
            write_fit(fit_predictions, f"{outdir}/fit_preds.csv", ua.benchmark)
            # except Exception as e:
            #    logger.error(f"Cannot process chunk {chunk_idx} using AFT due to {e}")

    else:
        logger.info("Cannot calculate FIT, years sampled and gen time not supplied.")
