import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import logging
from bisect import bisect_left


class BPError(Exception):
    pass


def load_data(infile):
    # Load dataframes
    return pd.read_csv(infile)


def get_middle_bp(physLen=100000):
    # Get spot in the middle of the window
    return int(physLen / 2)


def take_closest(myList, myNumber):
    """
    https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def get_window_idxs(snp_df, mid_bp, sweep_type):
    # Get the bp labels for the 25 on each side of mid_bp
    # If the physical half isn't in the list take the middle of the range of sorted values
    bps = sorted(list(snp_df["bp"].unique()))
    if sweep_type == "neut":
        closest_bp_phys = take_closest(bps, mid_bp)
        center_idx = bps.index(closest_bp_phys)
    else:
        mut_bp = list(snp_df[snp_df["mut_type"] == "m2"]["bp"])[0]
        center_idx = bps.index(mut_bp)

    return bps[center_idx - 25 : center_idx + 25 + 1]


def sort_gens(snp_df):
    # Get generations to build dict
    return sorted(list(snp_df["gen_sampled"].unique()))


def build_freq_mat(snp_df, window_bps, sorted_gens):
    # Build the frequency over time matrix
    ts_freqs = {}
    for gen in sorted_gens:
        ts_freqs[gen] = []
        _gendf = snp_df[snp_df["gen_sampled"] == gen]
        for bp in window_bps:
            if bp in list(_gendf["bp"]):
                ts_freqs[gen].append(_gendf[_gendf["bp"] == bp]["freq"].iloc[0])
            else:
                ts_freqs[gen].append(0)

    return ts_freqs


def convert_to_np(dflist):
    # Convert dfs to np arrays and save to npz for easy network loading
    return [np.array(i) for i in dflist]


def get_file_label(filename):
    # Creates a label in the format of "sweeptype/ID" for use in NPZ labelling
    # This format is identical to the other NPZ merge format used for other data types
    if "hard" in filename:
        sweeplab = "hard"
    elif "neut" in filename:
        sweeplab = "neut"
    elif "soft" in filename:
        sweeplab = "soft"
    else:
        print("Labels aren't correct, check filenames.")
        sys.exit()

    fileid = filename.split("/")[-1].split(".")[0]

    return f"{sweeplab}/{fileid}"


def worker(snpfile):
    try:
        snpdf = load_data(snpfile)
        arr_label = get_file_label(snpfile)

        mid_bp = get_middle_bp()
        window_idxs = get_window_idxs(snpdf, mid_bp, arr_label.split("/")[0])
        sys.stdout.flush()
        sorted_gens = sort_gens(snpdf)
        freqmat = build_freq_mat(snpdf, window_idxs, sorted_gens)

        return arr_label, pd.DataFrame.from_dict(freqmat)

    except Exception as e:
        logging.warning(f"Couldn't process {snpfile} because of {e}.")


def main():
    filelist = glob(os.path.join(sys.argv[1], "*.freqs"))
    base_dir = os.path.join(os.path.dirname(filelist[0]).split("freqs")[0])

    id_arrs = []
    pool = mp.Pool(processes=mp.cpu_count())
    for proc_result in tqdm(
        pool.imap_unordered(worker, filelist),
        desc="Creating freqmats...",
        total=len(filelist),
    ):
        if proc_result is not None:
            id_arrs.append(proc_result)
    # id_arrs = [worker(i) for i in filelist]

    arr_labels = [i[0] for i in id_arrs]
    df_list = [i[1] for i in id_arrs]

    arr_list = convert_to_np(df_list)

    np.savez(
        os.path.join(base_dir, arr_labels[0].split("/")[0] + "_freqmats.npz"),
        **dict(zip(arr_labels, arr_list)),
    )


if __name__ == "__main__":
    main()
