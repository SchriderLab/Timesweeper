import math
import os

import numpy as np
import pickle as pkl
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

from timesweeper.make_training_features import prep_ts_aft, get_window_idxs
from timesweeper.utils import snp_utils as su
from timesweeper.utils.gen_utils import read_config, get_logger
from timesweeper.utils import hap_utils as hu

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = get_logger("find_sweeps")


def write_preds(results_dict, outfile, scaler, benchmark):
    """
    Writes NN predictions to file.

    Args:
        results_dict (dict): SNP NN prediction scores and window edges.
        outfile (str): File to write results to.
    """
    with open(scaler, "rb") as ifile:
        scaler = pkl.load(ifile)

    lab_dict = {0: "Neut", 1: "SSV", 2: "SDN"}
    if benchmark:
        chroms, bps, mut_type, true_sel_coeff = zip(*results_dict.keys())
    else:
        chroms, bps = zip(*results_dict.keys())

    neut_scores = [i[0][0] for i in results_dict.values()]
    sdn_scores = [i[0][1] for i in results_dict.values()]
    ssv_scores = [i[0][2] for i in results_dict.values()]
    sdn_preds = [i[1] for i in results_dict.values()]
    ssv_preds = [i[2] for i in results_dict.values()]
    left_edges = [i[3] for i in results_dict.values()]
    right_edges = [i[4] for i in results_dict.values()]
    classes = [lab_dict[np.argmax(i[0])] for i in results_dict.values()]

    sdn_s = scaler.inverse_transform(sdn_preds).squeeze().flatten()
    ssv_s = scaler.inverse_transform(ssv_preds).squeeze().flatten()

    if benchmark:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Mut_Type": mut_type,
                "Class": classes,
                "Sweep_Prob": [i + j for i, j in zip(sdn_scores, ssv_scores)],
                "True_Sel_Coeff": true_sel_coeff,
                "SDN_s_pred": sdn_s,
                "SSV_s_pred": ssv_s,
                "Neut_Prob": neut_scores,
                "SDN_Prob": sdn_scores,
                "SSV_Prob": ssv_scores,
                "Win_Start": left_edges,
                "Win_End": right_edges,
            }
        )
    else:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Class": classes,
                "Sweep_Prob": [i + j for i, j in zip(sdn_scores, ssv_scores)],
                "SDN_s_pred": sdn_s,
                "SSV_s_pred": ssv_s,
                "Neut_Prob": neut_scores,
                "SDN_Prob": sdn_scores,
                "SSV_Prob": ssv_scores,
                "Win_Start": left_edges,
                "Win_End": right_edges,
            }
        )
        predictions = predictions[predictions["Neut_Prob"] < 0.5]

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    predictions.to_csv(outfile, header=True, index=False, sep="\t")

    bed_df = predictions[["Chrom", "Win_Start", "Win_End", "BP"]]
    bed_df.to_csv(
        outfile.replace(".csv", ".bed"), mode="a", header=False, index=False, sep="\t"
    )


def run_aft_windows(
    snps, genos, samp_sizes, win_size, class_model, sdn_model, ssv_model
):
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

    class_probs = class_model.predict(np.stack(data))
    sdn_sel_preds = sdn_model.predict(np.stack(data))
    ssv_sel_preds = ssv_model.predict(np.stack(data))

    results_dict = {}
    for center, class_probs, sd, sv, l_e, r_e in zip(
        centers, class_probs, sdn_sel_preds, ssv_sel_preds, left_edges, right_edges
    ):
        results_dict[snps[center]] = (
            class_probs,
            sd,
            sv,
            l_e,
            r_e,
        )

    return results_dict


def run_hft_windows(
    snps, haps, ploidy, samp_sizes, win_size, class_model, sdn_model, ssv_model
):
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
    for center in tqdm(centers, desc="Predicting on HFT windows"):
        # try:
        win_idxs = get_window_idxs(center, win_size)
        window = np.swapaxes(haps[win_idxs, :], 0, 1)
        str_window = hu.haps_to_strlist(window)
        hft = hu.getTSHapFreqs(str_window, [i * ploidy for i in samp_sizes])
        data.append(hft)
        left_edges.append(snps[win_idxs[0]][1])
        right_edges.append(snps[win_idxs[-1]][1])

        # except Exception as e:
        #    logger.warning(f"Center {snps[center]} raised error {e}")
        #    continue

    class_probs = class_model.predict(np.stack(data))
    sdn_sel_preds = sdn_model.predict(np.stack(data))
    ssv_sel_preds = ssv_model.predict(np.stack(data))

    results_dict = {}
    for center, class_probs, sd, sv, l_e, r_e in zip(
        centers, class_probs, sdn_sel_preds, ssv_sel_preds, left_edges, right_edges
    ):
        results_dict[snps[center]] = (
            class_probs,
            sd,
            sv,
            l_e,
            r_e,
        )

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
    samp_sizes = yaml_data["sample sizes"]
    ploidy = yaml_data["ploidy"]
    win_size = yaml_data["win_size"]

    class_aft_model = load_nn(ua.aft_class_model)
    if "ssv" in ua.aft_reg_model:
        aft_ssv_model = load_model(ua.aft_reg_model)
        aft_sdn_model = load_model(str(ua.aft_reg_model).replace("ssv", "sdn"))
    elif "sdn" in ua.aft_reg_model:
        aft_ssv_model = load_model(str(ua.aft_reg_model).replace("ssv", "sdn"))
        aft_sdn_model = load_model(ua.aft_reg_model)

    if ua.hft_class_model:
        class_hft_model = load_nn(ua.hft_class_model)
        if "ssv" in ua.hft_reg_model:
            hft_ssv_model = load_model(ua.hft_reg_model)
            hft_sdn_model = load_model(str(ua.hft_reg_model).replace("ssv", "sdn"))
        elif "sdn" in ua.hft_reg_model:
            hft_ssv_model = load_model(str(ua.hft_reg_model).replace("ssv", "sdn"))
            hft_sdn_model = load_model(ua.hft_reg_model)

    outfile = ua.outfile

    # Chunk and iterate for NN predictions to not take up too much space
    vcf_iter = su.get_vcf_iter(ua.input_vcf, ua.benchmark)
    for chunk_idx, chunk in enumerate(vcf_iter):
        chunk = chunk[0]  # Why you gotta do me like that, skallel?
        logger.info(f"Processing VCF chunk {chunk_idx}")

        # aft
        # try:
        genos, snps = su.vcf_to_genos(chunk, ua.benchmark)
        aft_predictions = run_aft_windows(
            snps,
            genos,
            samp_sizes,
            win_size,
            class_aft_model,
            aft_sdn_model,
            aft_ssv_model,
        )
        write_preds(aft_predictions, f"{outfile}_aft.csv", ua.scalar, ua.benchmark)

        # except Exception as e:
        #    logger.error(f"Cannot process chunk {chunk_idx} using AFT due to {e}")

        # hft
        if ua.hft_class_model:
            # try:
            haps, snps = su.vcf_to_haps(chunk, ua.benchmark)
            hft_predictions = run_hft_windows(
                snps,
                haps,
                ploidy,
                samp_sizes,
                win_size,
                class_hft_model,
                hft_sdn_model,
                hft_ssv_model,
            )
            write_preds(hft_predictions, f"{outfile}_hft.csv", ua.scalar, ua.benchmark)

            # except Exception as e:
            #    logger.error(f"Cannot process chunk {chunk_idx} using HFT due to {e}")
