import argparse as ap
import logging
import os
from itertools import cycle
import pickle as pkl
import numpy as np
import pandas as pd
import yaml
from tensorflow.keras.models import load_model
from tqdm import tqdm

from timesweeper import find_sweeps_vcf as fsv
from timesweeper.utils.gen_utils import read_config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig()
logger = logging.getLogger("timesweeper")
logger.setLevel("INFO")

pd.options.display.float_format = "{:.2f}".format


def load_npz(npz_path):
    npz_obj = np.load(npz_path)
    aft = npz_obj["aftIn"]  # (snps, timepoints, window)
    locs = npz_obj["aftInPosition"]  # (snps, window_locs)

    return aft, locs


def parse_npz_name(npz_path):
    splitpath = npz_path.split("_")
    chrom = splitpath[2]
    rep = splitpath[-1].split(".")[0]

    return chrom, rep


def run_aft_windows(ts_aft, locs, chrom, class_model, reg_models, scaler):
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
    left_edges = list(locs[:, 0])
    right_edges = list(locs[:, -1])
    centers = list(locs[:, 25])
    class_probs = class_model.predict(ts_aft)
    reg_preds = [scaler.inverse_transform(model.predict(ts_aft).reshape(-1, 1)) for model in reg_models.values()]
    return [chrom for i in range(len(centers))], centers, left_edges, right_edges, class_probs, reg_preds


def write_preds(chroms, centers, left_edges, right_edges, class_probs, reg_preds, outfile, scenarios):
    """
    Writes NN predictions to file.
    Args:
        results_dict (dict): SNP NN prediction scores and window edges.
        outfile (str): File to write results to.
    """
    lab_dict = {idx: s for idx, s in enumerate(scenarios)}
    class_scores = [[i[j] for i in class_probs] for j in range(len(scenarios))]
    classes = [lab_dict[np.argmax(i)] for i in class_probs]
    
    pred_dict = {
            "Chrom": chroms,
            "BP": centers,
            "Class": classes,
            "Win_Start": left_edges,
            "Win_End": right_edges,
        }

    for s,c in zip(scenarios, class_scores):
        pred_dict[f"{s}_Prob"] = c
        
    for s,c in zip(scenarios[1:], reg_preds):
        pred_dict[f"{s}_selcoeff_pred"] = c.flatten()

    
    predictions = pd.DataFrame(pred_dict)

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    predictions.to_csv(outfile, header=True, index=False, sep="\t", float_format="%.3f")

    bed_df = predictions[["Chrom", "Win_Start", "Win_End", "BP"]]
    bed_df.to_csv(outfile.replace(".csv", ".bed"), header=False, index=False, sep="\t")


def main(ua):
    yaml_data = fsv.read_config(ua.yaml_file)

    scenarios = yaml_data["scenarios"]
    work_dir = yaml_data["work dir"]
    experiment_name = yaml_data["experiment name"]    
    class_aft_model = load_model(f"{work_dir}/trained_models/{experiment_name}_Timesweeper_Class_aft")
    reg_aft_models = {scenario: load_model(f"{work_dir}/trained_models/REG_{experiment_name}_{scenario}_Timesweeper_Reg_aft") for scenario in scenarios[1:]}
    
    try:
        if not os.path.exists(ua.outdir):
            os.makedirs(ua.outdir)
    except:
        # running in high-parallel sometimes it errors when trying to check/create simultaneously
        pass
    
    with open(f"{work_dir}/trained_models/{experiment_name}_selcoeff_scaler.pkl", "rb") as ifile:
        scaler = pkl.load(ifile)

    # Load in everything
    logger.info(f"Loading data from {ua.input_file}")
    chrom, rep = parse_npz_name(ua.input_file)
    ts_aft, locs = load_npz(ua.input_file)

    # aft
    logger.info("Predicting with AFT")
    chroms, centers, left_edges, right_edges, class_probs, reg_preds = run_aft_windows(ts_aft, locs, chrom, class_aft_model, reg_aft_models, scaler)
    write_preds(chroms, centers, left_edges, right_edges, class_probs, reg_preds, f"{ua.outdir}/aft_{chrom}_{rep}_preds.csv", scenarios)
    logger.info(f"Done, results written to {ua.outdir}/aft_{chrom}_{rep}_preds.csv")
