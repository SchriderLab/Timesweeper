import os, sys
import multiprocessing as mp
import argparse as ap
from utils.vcf import vcf_to_afs, create_afm

# from tensorflow.keras.models import load_model
import numpy as np


def run_windows(snps, afs, model):
    results_dict = {}
    for center in range(len(snps)):
        win_idxs = get_window_idxs(center)
        window = np.swapaxes(afs[win_idxs, :], 0, 1)
        probs = model.predict(window)
        results_dict[snps[center]] = probs


def pad_neg_idxs(idxs, arr):
    pass


def classify_window(win_snps, model):
    return model.predict(win_snps)


def get_window_idxs(center_idx):
    return list(range(center_idx - 25, center_idx + 25 + 1))


def load_nn(model_path, summary=False):
    model = load_model(model_path)
    if summary:
        print(model.summary())

    return model


def worker(args):
    vcfs, model = args
    freq_dicts = [vcf_to_afs(vcf_file) for vcf_file in vcfs]
    snps, afs = create_afm(freq_dicts)


def parse_ua():
    uap = ap.ArgumentParser(
        description="Module for iterating across windows in a time-series vcf file and predicting whether a sweep is present at each snp-centralized window."
    )
    uap.add_argument(
        "-i",
        "--input-vcfs",
        dest="input_vcfs",
        help="VCFs to scan for sweeps. Provide in order of earliest to most recent samples.",
        nargs="+",
        required=True,
    )
    uap.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        help="Directory to write results to.",
        required=False,
        default=os.getcwd(),
    )
    uap.add_argument(
        "-nn",
        "--network-model",
        dest="nn_model",
        help="Path to Keras2-style saved model to load for prediction.",
        default=None,
    )
    uap.add_argument(
        "--threads",
        dest="threads",
        help="Threads to use for multiprocessing.",
        default=mp.cpu_count() - 1 or 1,
    )

    return uap.parse_args()


def main():
    ua = parse_ua()
    input_vcfs, output_dir, nn_model, threads = (
        ua.input_vcfs,
        ua.output_dir,
        ua.nn_model,
        ua.threads,
    )
    model = load_nn(nn_model)
    worker(input_vcfs)


if __name__ == "__main__":
    main()
