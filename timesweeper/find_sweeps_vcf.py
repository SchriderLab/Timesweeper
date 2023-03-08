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


def write_preds(scenarios, mut_types, results_dict, outfile, scaler, benchmark, true_class):
    """
    Writes NN predictions to file.

    Args:
        results_dict (dict): SNP NN prediction scores and window edges.
        outfile (str): File to write results to.
    """
    lab_dict = {idx: s for idx, s in enumerate(scenarios)}
    if benchmark:
        chroms, bps, mut_type, true_sel_coeff = zip(*results_dict.keys())
    else:
        chroms, bps = zip(*results_dict.keys())

    class_scores = [[i[0][j] for i in results_dict.values()] for j in range(len(scenarios))]
    reg_preds = [[i[j+1] for i in results_dict.values()] for j in range(len(scenarios))]
    left_edges = [i[-2] for i in results_dict.values()]
    right_edges = [i[-1] for i in results_dict.values()]
    classes = [lab_dict[np.argmax(i[0])] for i in results_dict.values()]
    scaled_s = [scaler.inverse_transform(np.array(p).reshape(-1, 1)).squeeze().flatten() for p in reg_preds]

    if benchmark:
        true_classes = []
        for i in mut_type:
            if i in mut_types:
                true_classes.append(true_class)
            else:
                true_classes.append(scenarios[0])

        pred_dict = {                
                "Chrom": chroms,
                "BP": bps,
                "Mut_Type": mut_type,
                "True_Class": true_classes,
                "Pred_Class": classes,
                "True_Sel_Coeff": true_sel_coeff,
                "Win_Start": left_edges,
                "Win_End": right_edges}
        
        for s,c in zip(scenarios, class_scores):
            pred_dict[f"{s}_Prob"] = c
            
        for s,c in zip(scenarios[1:], scaled_s):
            pred_dict[f"{s}_selcoeff_pred"] = c

        predictions = pd.DataFrame(pred_dict)
    
    else:
        pred_dict = {                
                "Chrom": chroms,
                "BP": bps,
                "Pred_Class": classes,
                "Win_Start": left_edges,
                "Win_End": right_edges}
        
        for s,c in zip(scenarios, class_scores):
            pred_dict[f"{s}_Prob"] = c
            
        for s,c in zip(scenarios[1:], scaled_s):
            pred_dict[f"{s}_selcoeff_pred"] = c

        predictions = pd.DataFrame(pred_dict)

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    predictions.to_csv(outfile, header=True, index=False, sep="\t")

    bed_df = predictions[["Chrom", "Win_Start", "Win_End", "BP"]]
    bed_df.to_csv(
        outfile.replace(".csv", ".bed"), mode="a", header=False, index=False, sep="\t"
    )


def run_aft_windows(
    snps, genos, samp_sizes, win_size, class_model, reg_models
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
    reg_preds = [model.predict(np.stack(data)) for model in reg_models.values()]
    
    results_dict = {}
    for center, res in zip(centers, zip(
        class_probs, *reg_preds, left_edges, right_edges
    )):
        results_dict[snps[center]] = res

    return results_dict


def run_hft_windows(
    snps, haps, ploidy, samp_sizes, win_size, class_model, reg_models
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
        win_idxs = get_window_idxs(center, win_size)
        window = np.swapaxes(haps[win_idxs, :], 0, 1)
        str_window = hu.haps_to_strlist(window)
        hft = hu.getTSHapFreqs(str_window, [i * ploidy for i in samp_sizes])
        data.append(hft)
        left_edges.append(snps[win_idxs[0]][1])
        right_edges.append(snps[win_idxs[-1]][1])

    class_probs = class_model.predict(np.stack(data))
    reg_preds = [model.predict(np.stack(data)) for model in reg_models.values()]
    
    results_dict = {}
    for center, res in zip(centers, zip(
        class_probs, *reg_preds, left_edges, right_edges
    )):
        results_dict[snps[center]] = res

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


def get_swp(filename, scenarios):
    for s in scenarios:
        if s in filename:
            return s

def main(ua):
    yaml_data = read_config(ua.yaml_file)
    samp_sizes = yaml_data["sample sizes"]
    ploidy = yaml_data["ploidy"]
    win_size = yaml_data["win_size"]
    scenarios = yaml_data["scenarios"]
    work_dir = yaml_data["work dir"]
    experiment_name = yaml_data["experiment name"]
    mut_types = yaml_data["mut types"]

    class_aft_model = load_model(f"{work_dir}/trained_models/{experiment_name}_Timesweeper_Class_aft")
    reg_aft_models = {scenario: load_model(f"{work_dir}/trained_models/REG_{experiment_name}_{scenario}_Timesweeper_Reg_aft") for scenario in scenarios[1:]}
    
    with open(f"{work_dir}/trained_models/{experiment_name}_selcoeff_scaler.pkl", "rb") as ifile:
        scaler = pkl.load(ifile)

    if ua.benchmark:
        true_class = get_swp(ua.input_vcf, scenarios)
    else:
        true_class = None
    
    # Chunk and iterate for NN predictions to not take up too much space
    vcf_iter = su.get_vcf_iter(ua.input_vcf, ua.benchmark)
    for chunk_idx, chunk in enumerate(vcf_iter):
        chunk = chunk[0]  # Why you gotta do me like that, skallel?
        logger.info(f"Processing VCF chunk {chunk_idx}")

        # aft
        genos, snps = su.vcf_to_genos(chunk, ua.benchmark)
        aft_predictions = run_aft_windows(
            snps,
            genos,
            samp_sizes,
            win_size,
            class_aft_model,
            reg_aft_models,
        )
        write_preds(scenarios, mut_types, aft_predictions, f"{ua.output_dir}/{experiment_name}_aft.csv", scaler, ua.benchmark, true_class)

        # hft
        if ua.hft:
            class_hft_model = load_model(f"{work_dir}/trained_models/{experiment_name}_Timesweeper_Class_hft")
            reg_hft_models = {scenario: load_model(f"{work_dir}/trained_models/REG_{experiment_name}_{scenario}_Timesweeper_Reg_hft") for scenario in scenarios[1:]}
     
            haps, snps = su.vcf_to_haps(chunk, ua.benchmark)
            hft_predictions = run_hft_windows(
                snps,
                haps,
                ploidy,
                samp_sizes,
                win_size,
                class_hft_model,
                reg_hft_models,
            )
            write_preds(scenarios, mut_types, hft_predictions, f"{ua.output_dir}/{experiment_name}_hft.csv", scaler, ua.benchmark, true_class)
