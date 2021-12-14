import argparse as ap
import multiprocessing as mp
import os

import allel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tqdm import tqdm

from feder_fit import fit
from utils.haps import getTSHapFreqs, haps_to_strlist
from utils.vcf import vcf_to_genos, vcf_to_haps


def run_afs_windows(snps, genos, samp_sizes, win_size, model):
    ts_genos = split_arr(genos, samp_sizes)
    afs = np.stack([calc_mafs(allel.GenotypeArray(geno)) for geno in ts_genos])
    padded_freqs = pad_arr(afs, win_size)
    results_dict = {}
    for center in tqdm(range(len(snps)), desc="Predicting on AFS windows"):
        win_idxs = get_window_idxs(center, win_size)
        window = padded_freqs[:, win_idxs]

        probs = model.predict(np.expand_dims(window, 0))
        results_dict[snps[center]] = probs

    return results_dict


def run_fit_windows(snps, genos, samp_sizes, gens):
    ts_genos = split_arr(genos, samp_sizes)
    afs = np.stack([calc_mafs(allel.GenotypeArray(geno)) for geno in ts_genos])
    results_dict = {}
    for idx in tqdm(range(len(snps)), desc="Calculating FIT values"):
        results_dict[snps[idx]] = fit(list(afs[:, idx]), gens)  # tval, pval

    return results_dict


def run_hfs_windows(snps, haps, samp_sizes, win_size, model):
    padded_haps = pad_arr(np.swapaxes(haps, 0, 1), win_size)
    results_dict = {}
    for center in tqdm(range(len(snps)), desc="Predicting on HFS windows"):
        win_idxs = get_window_idxs(center, win_size)
        window = padded_haps[:, win_idxs]
        str_window = haps_to_strlist(np.swapaxes(window, 0, 1))
        hfs = getTSHapFreqs(str_window, samp_sizes)

        probs = model.predict(np.expand_dims(hfs, 0))
        results_dict[snps[center]] = probs

    return results_dict


def pad_arr(arr, win_size):
    """Puts afs in the middle of zeros size of window on either side."""
    padded = np.zeros((arr.shape[0], (2 * win_size) + arr.shape[1]))
    padded[:, win_size : win_size + arr.shape[1]] = arr
    return padded


def split_arr(arr, samp_sizes):
    i = 0
    arr_list = []
    for j in samp_sizes:
        arr_list.append(arr[:, i : i + j])
        i += j

    return np.stack(arr_list)


def calc_mafs(geno_arr):
    alleles = geno_arr.count_alleles()
    return alleles[:, 1] / alleles.sum(axis=1)


### Merge timepoints
def dicts_to_keyset(freq_dicts):
    keys = []
    for i in freq_dicts:
        keys.extend([*i])

    return list(set(keys))


def classify_window(win_snps, model):
    return model.predict(win_snps)


def get_window_idxs(center_idx, win_size):
    half_window = int(win_size / 2)
    return list(range(center_idx - half_window, center_idx + half_window + 1))


def load_nn(model_path, summary=False):
    model = load_model(model_path)
    if summary:
        print(model.summary())

    return model


def write_fit(fit_dict, outfile):
    chroms, bps = zip(*fit_dict.keys())

    inv_pval = [1 - i[1] for i in fit_dict.values()]

    predictions = pd.DataFrame({"Chrom": chroms, "BP": bps, "Inv pval": inv_pval})
    predictions.sort_values(["Chrom", "BP"], inplace=True)

    predictions.to_csv(os.path.join(outfile), header=True, index=False, sep="\t")


def write_preds(pred_dict, outfile):
    lab_dict = {0: "Neut", 1: "Hard", 2: "Soft"}
    chroms, bps = zip(*pred_dict.keys())

    neut_scores = [i[0][0] for i in pred_dict.values()]
    hard_scores = [i[0][1] for i in pred_dict.values()]
    soft_scores = [i[0][2] for i in pred_dict.values()]

    classes = [lab_dict[np.argmax(i, axis=1)[0]] for i in pred_dict.values()]
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

    predictions.to_csv(os.path.join(outfile), header=True, index=False, sep="\t")


def add_file_label(filename, label):
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
        dest="input_vcfs",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )

    uap.add_argument(
        "-s",
        "--sample-sizes",
        dest="samp_sizes",
        help="Number of diploid individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling points.",
        required=True,
        nargs="+",
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
    uap.add_argument(
        "-o",
        "--output",
        dest="outfile",
        help="File to write results to.",
        required=False,
        default="Timesweeper_predictions.csv",
    )
    return uap.parse_args()


def main():
    ua = parse_ua()
    input_vcf, samp_sizes, outfile, afs_model, hfs_model = (
        ua.input_vcfs,
        [int(i) for i in ua.samp_sizes],
        ua.outfile,
        load_nn(ua.afs_model),
        load_nn(ua.hfs_model),
    )
    win_size = 51  # Must be consistent with training data

    # AFS
    genos, snps = vcf_to_genos(input_vcf)
    predictions = run_afs_windows(snps, genos, samp_sizes, win_size, afs_model)
    afs_file = add_file_label(outfile, "afs")
    write_preds(predictions, afs_file)

    # FIT
    GEN_STEP = 10
    GENS = list(range(10060, 10250 + GEN_STEP, GEN_STEP))
    genos, snps = vcf_to_genos(input_vcf)
    predictions = run_fit_windows(snps, genos, samp_sizes, GENS)
    fit_file = add_file_label(outfile, "fit")
    write_fit(predictions, fit_file)

    # HFS
    haps, snps = vcf_to_haps(input_vcf)
    predictions = run_hfs_windows(snps, haps, samp_sizes, win_size, hfs_model)
    hfs_file = add_file_label(outfile, "hfs")
    write_preds(predictions, hfs_file)


# python classify_windows.py -nn ../simple_sims/models/bighaps_TimeSweeper -i ../simple_sims/vcf_sims/hard/pops/1/merged.vcf.gz -s $(printf '10 %.s' {1..20})

if __name__ == "__main__":
    main()
