import math
import os

import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

from .make_training_features import prep_ts_aft, get_window_idxs
from .utils import snp_utils as su
from .utils.gen_utils import read_config, write_preds, get_logger
from .utils import hap_utils as hu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = get_logger("find_sweeps")


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
    yaml_data = read_config(ua.yaml_file)
    (work_dir, samp_sizes, ploidy, win_size, outdir, aft_model) = (
        yaml_data["work dir"],
        yaml_data["sample sizes"],
        yaml_data["ploidy"],
        yaml_data["win_size"],
        ua.outdir,
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
