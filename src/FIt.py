import multiprocessing as mp
import os
import sys
import warnings
from argparse import ArgumentParser
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)
# https://www.genetics.org/content/196/2/509

df = pd.DataFrame


def getRescaledIncs(freqs: List[int], gens: List[int]) -> List[float]:
    """
    Calculates allele increments for the FIT.

    Args:
        freqs (List[int]): Frequency of a given allele at each site.
        gens (List[int]): Generations that each allele is sampled.

    Returns:
        List[float]: Rescaled frequency increments that can be used in t-test.
    """
    incs = []
    i = 0
    # advance to first non-zero freq
    while i < len(freqs) and (freqs[i] == 0 or freqs[i] == 1):
        i += 1
    if i < len(freqs):
        prevFreq = freqs[i]
        prevGen = gens[i]
        i += 1
        while i < len(freqs):
            if freqs[i] != 0 and freqs[i] != 1:
                num = freqs[i] - prevFreq
                denom = ((2 * prevFreq) * (1 - prevFreq) * (gens[i] - prevGen)) ** 0.5
                incs.append(num / denom)
                prevFreq = freqs[i]
                prevGen = gens[i]
            i += 1

    return incs


def remove_restarts(lines) -> List[str]:
    """
    Gets rid of restarts in the simulation output using generation labels.

    Args:
        lines (List[str]): Parsed lines from slim simulation output.

    Returns:
        List[str]: Slim output without any restarts at any point in the simulation.
    """
    gens = []
    out_idxs = []
    for idx, line in enumerate(lines):
        if "#OUT" in line:
            gens.append(int(line[1]))
            out_idxs.append(idx)

    for gen in gens:
        gen_inds = [i for i, x in enumerate(gens) if x == gen]
        if (
            len(gen_inds) > 1
        ):  # Get indices of any duplicated gens - spans gens and out_idxs
            # Remove lines between the first and last occurence
            # Only want to keep the ones after the restart
            # Technically only restarts should happen at the dumpfile gen, but this is flexible for anything I suppose
            del lines[out_idxs[gen_inds[0]] : out_idxs[gen_inds[-1]]]

    return lines


def fit(freqs, gens):
    """
    Calculate FIT by performing 1-Sided Student t-test on frequency increments.

    Args:
        freqs (List[int]): Frequencies at all given generations for targeted alleles.
        gens (List[int]): Generations sampled.

    Returns:
        List[int]: t-test results. 
    """
    rescIncs = getRescaledIncs(freqs, gens)
    return ttest_1samp(rescIncs, 0)


def bin_samps(
    samp_sizes, gens_sampled, gen_threshold=25, size_threshold=3
) -> Tuple[List[int], List[int], List[int]]:
    """
    Bins a list of ints into condensed bins where the minimum value is equal to <size_threshold>.
    Each bin must also not be larger than <gen_threshold>.

    Args:
        samp_sizes (list[int]): List of ints to bin.
        gens_sampled (list[int]): List of generations sampled.
        gen_threshold (int, optional): Minimum size of generation window any given bin can be. Defaults to 25.
        size_threshold (int, optional): Minimum number of samples in any given bin. Defaults to 3.

    Returns:
        list[int]: Binned values.
    """
    bin_inds = []  # End of each bin

    i = 0
    while i < len(samp_sizes):
        if samp_sizes[i] >= size_threshold:
            i += 1
            bin_inds.append(i)

        else:
            j = 0
            while sum(samp_sizes[i : i + j]) < size_threshold:
                if i + j == len(gens_sampled):
                    # Hit the end before it's good, just take whatever's left
                    break
                elif gens_sampled[i + j] - gens_sampled[i] > gen_threshold:
                    # Good to go, append sample
                    break
                else:
                    # Need more samples, add the next timepoint
                    j += 1

            if i + j == len(gens_sampled):
                bin_inds.append(i + j)
            else:
                bin_inds.append(i + j)

            i += j

    binned_gens = []
    binned_sizes = []
    i = 0
    for j in bin_inds:
        binned_sizes.append(sum(samp_sizes[i:j]))
        binned_gens.append(int(np.mean(gens_sampled[i:j])))
        i = j

    return binned_gens, bin_inds, binned_sizes


def bin_dfs(df_list: List[df], bin_inds: List[int], bin_gens: List[int],) -> List[df]:
    """
    Uses indices of bins from bin_samps() to combine DFs and calculate adjusted frequency.

    Args:
        df_list (List[df]): DFs needing to be binned
        binned_gens (List[int]): Generation of the last sample of each bin, used as the gen for entire bin
        bin_inds (List[int]): Indices of last sampling for each bin, used to iterate through samples
        binned_sizes (List[int]): Number of chroms per bin

    Returns:
        List[df]: DFs with frequencies calculated based on binned values
    """
    binned_dfs = []
    i = 0
    for j in range(len(bin_inds)):
        if i == bin_inds[j] or bin_inds[j] == bin_inds[-1]:
            if i == len(df_list):
                _df = df_list[-1]
            else:
                _df = df_list[i]

            _df["gen_sampled"] = bin_gens[j]
            binned_dfs.append(_df)
            i += 1
        else:
            # Join dfs that are within a bin
            # print(bin_inds[-1])
            # print(i, bin_inds[j])
            _df = pd.concat(df_list[i : bin_inds[j]])
            _df = (
                _df.groupby(["perm_ID", "bp", "mut_type"])
                .agg({"prevalence": "sum", "sampsize": "sum", "freq": "mean"})
                .reset_index()
            )
            _df["gen_sampled"] = bin_gens[j]
            binned_dfs.append(_df)
            i = bin_inds[j]

    return binned_dfs


def get_muts(filename: str) -> Union[df, None]:
    """
    Parses SLiM output file, bins, and converts to dataframe for easy calculation downstream.

    Args:
        filename (str): SLiM output file. Must be from some form of slim's "outputsample" format containing individuals and genomes.

    Returns:
        Union[df, None]: Binned dataframe containing all information about each mutation at every timepoint.
    """
    with open(filename, "r") as mutfile:
        rawlines = [i.strip().split(" ") for i in mutfile.readlines()]
    cleaned_lines = remove_restarts(rawlines)

    gens_sampled = [int(line[1]) for line in cleaned_lines if "#OUT:" in line]
    samp_sizes = [int(line[4]) for line in cleaned_lines if "#OUT:" in line]

    binned_gens, bin_inds, binned_sizes = bin_samps(samp_sizes, gens_sampled)

    # Stack a bunch of dfs in a list, concat
    gen_dfs = []
    mutlist = []

    entry_tracker = 0
    for i in range(len(cleaned_lines)):
        if "#OUT:" in cleaned_lines[i]:
            if entry_tracker > 0:
                mutdf = df(
                    mutlist,
                    columns=[
                        "tmp_ID",
                        "perm_ID",
                        "mut_type",
                        "bp",
                        "sel_coeff",
                        "dom_coeff",
                        "subpop_ID",
                        "gen_arose",
                        "prevalence",
                    ],
                )
                mutdf = mutdf.astype(
                    {
                        "tmp_ID": int,
                        "perm_ID": int,
                        "mut_type": str,
                        "bp": int,
                        "sel_coeff": float,
                        "dom_coeff": float,
                        "subpop_ID": str,
                        "gen_arose": int,
                        "prevalence": int,
                    }
                )
                cleaned_mutdf = mutdf.drop(
                    ["tmp_ID", "sel_coeff", "dom_coeff", "subpop_ID", "gen_arose"],
                    axis=1,
                )
                gen_dfs.append(cleaned_mutdf)
            mutlist = []  # Reset for the next round
            entry_tracker += 1

        elif (
            (len(cleaned_lines[i]) == 9)
            and (cleaned_lines[i][2] == "m1" or cleaned_lines[i][2] == "m2")
            and ("p1" not in cleaned_lines[i][1])
        ):
            mutlist.append(cleaned_lines[i])
        else:
            continue

    for i in range(len(gen_dfs)):
        gen_dfs[i]["sampsize"] = samp_sizes[i]
        gen_dfs[i]["freq"] = gen_dfs[i]["prevalence"] / gen_dfs[i]["sampsize"]

    binned_dfs = bin_dfs(gen_dfs, bin_inds, binned_gens)
    try:
        clean_gens_df = (
            pd.concat(binned_dfs, axis=0, ignore_index=True)
            .drop_duplicates()
            .reset_index()
        )
        clean_gens_df["bp"] = clean_gens_df["bp"].astype(int)

        return clean_gens_df

    except:
        print(f"Couldn't process {filename}.")


def write_fitfile(mutdf: df, outfilename: str) -> None:
    """
    Writes the resulting FIt values to csv.
    #TODO should probably separate this out

    Args:
        mutdf (df): DF with all mutations from pop file, cleaned and binned.
        outfilename (str): Name of csv file to output.
    """
    cut_labs = list(range(11))
    cut_bins = np.linspace(
        np.min(mutdf["bp"]), np.max(mutdf["bp"]), 12
    )  # Divide into windows across the genomic region
    mutdf["window"] = pd.cut(mutdf["bp"], bins=cut_bins, labels=cut_labs)
    # print(mutdf)

    mut_dict = {
        "mut_ID": [],
        "mut_type": [],
        "location": [],
        "window": [],
        "fit_t": [],
        "fit_p": [],
        "selection_detected": [],
    }

    # Do iterations for calculations, it's just simpler
    for mutID in mutdf["perm_ID"].unique():
        subdf = mutdf[mutdf["perm_ID"] == mutID].reset_index()

        if len(subdf) > 2:
            try:
                fit_t, fit_p = fit(list(subdf["freq"]), list(subdf["gen_sampled"]))

                if not np.isnan(fit_t):
                    mut_dict["mut_ID"].append(int(mutID))
                    mut_dict["mut_type"].append(subdf.mut_type[0])
                    mut_dict["location"].append(subdf.bp[0])
                    mut_dict["window"].append(subdf.window[0])
                    mut_dict["fit_t"].append(float(fit_t))
                    mut_dict["fit_p"].append(fit_p)
                    if fit_p <= 0.05:
                        mut_dict["selection_detected"].append(1)
                    else:
                        mut_dict["selection_detected"].append(0)

            except:
                continue

        else:
            continue

    outdf = df(mut_dict)
    outdf = outdf[outdf["window"] == 5]

    leadup, base = os.path.split(outfilename)
    newoutfilename = os.path.join(leadup, "fit", base)
    if not os.path.exists(os.path.join(leadup, "fit")):
        os.makedirs(os.path.join(leadup, "fit"), exist_ok=True)

    outdf.to_csv(newoutfilename + ".fit", header=True, index=False)
    sys.stdout.flush()
    sys.stderr.flush()


def fit_gen(mutfile: str) -> None:
    """
    Worker function for multiprocessing.

    Args:
        mutfile (str): SLiM output file to parse.
    """
    mut_df = get_muts(mutfile)
    if mut_df is not None:
        write_fitfile(mut_df, mutfile)
    else:
        print("Nothin")
        pass


def parse_args():
    agp = ArgumentParser(
        description="Reads in *.pop files from slim output and performs Frequency Increment Test as described by Feder et al. 2014. \
        Writes files to a new /fit/ directory in the input directory."
    )
    agp.add_argument(
        "-i",
        "--input-dir",
        metavar="INPUT_DIRECTORY",
        help="Base mutation type (hard/soft/etc) directory with *.pop files to create feature vectors from. Defaults to pwd.",
        dest="in_dir",
        type=str,
        required=False,
        default=".",
    )

    agp.add_argument(
        "-t",
        "--threads",
        metavar="THREADS",
        help="Threads to parallelize across.",
        dest="threads",
        type=int,
        required=False,
        default=4,
    )

    return agp.parse_args()


def main():
    agp = parse_args()
    popfiles = glob(os.path.join(agp.in_dir, "*.pop"))

    with mp.Pool(agp.threads) as p:
        for _ in tqdm(
            p.imap_unordered(fit_gen, popfiles),
            total=len(popfiles),
            desc=f"Creating fitfiles in {agp.in_dir}",
        ):
            pass

    # for i in popfiles:
    #    fit_gen(i)


if __name__ == "__main__":
    main()
