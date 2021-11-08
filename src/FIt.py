import multiprocessing as mp
import os
import sys
import warnings
from argparse import ArgumentParser
from glob import glob
from typing import List, Union

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


def get_muts(filename: str) -> Union[df, None]:
    """
    Parses SLiM output file, bins, and converts to dataframe for easy calculation downstream.

    Args:
        filename (str): SLiM output file. Must be from some form of slim's "outputsample" format containing individuals and genomes.
        bin_sizes (List[int]): Size of each bin, precalculated before running this and fed as a list of ints.
    Returns:
        Union[df, None]: Binned dataframe containing all information about each mutation at every timepoint.
    """
    with open(filename, "r") as mutfile:
        rawlines = [i.strip().split(" ") for i in mutfile.readlines()]
    cleaned_lines = remove_restarts(rawlines)

    samp_sizes = [int(line[4]) for line in cleaned_lines if "#OUT:" in line]
    gens_sampled = [int(line[1]) for line in cleaned_lines if "#OUT:" in line]

    # Stack a bunch of dfs in a list, concat
    gen_dfs = []
    mutlist = []

    header_list = [
        "tmp_ID",
        "perm_ID",
        "mut_type",
        "bp",
        "sel_coeff",
        "dom_coeff",
        "subpop_ID",
        "gen_arose",
        "prevalence",
    ]

    entry_tracker = 0
    for i in range(len(cleaned_lines)):
        if "#OUT:" in cleaned_lines[i]:
            if entry_tracker > 0:
                mutdf = df(mutlist, columns=header_list)
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
            entry_tracker += 1
            mutlist = []  # Reset for the next round

        elif (
            (len(cleaned_lines[i]) == 9)
            and (cleaned_lines[i][2] == "m1" or cleaned_lines[i][2] == "m2")
            and ("p1" not in cleaned_lines[i][1])
        ):
            mutlist.append(cleaned_lines[i])
        else:
            continue

    # Last one
    mutdf = df(mutlist, columns=header_list)
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
        ["tmp_ID", "sel_coeff", "dom_coeff", "subpop_ID", "gen_arose"], axis=1,
    )
    gen_dfs.append(cleaned_mutdf)

    # Iterate through and get freqs
    for i in range(len(gen_dfs)):
        gen_dfs[i]["sampsize"] = samp_sizes[i]
        gen_dfs[i]["gen_sampled"] = gens_sampled[i]
        gen_dfs[i]["freq"] = gen_dfs[i]["prevalence"] / gen_dfs[i]["sampsize"]

    try:
        clean_gens_df = (
            pd.concat(gen_dfs, axis=0, ignore_index=True)
            .drop_duplicates()
            .reset_index()
        )
        clean_gens_df["bp"] = clean_gens_df["bp"].astype(int)
        clean_gens_df = clean_gens_df.drop("index", axis=1)

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
    leadup = os.path.join(leadup.split("pops")[0])

    newoutfilename = os.path.join(leadup, "fit", base)
    if not os.path.exists(os.path.join(leadup, "fit")):
        os.makedirs(os.path.join(leadup, "fit"), exist_ok=True)

    outdf.to_csv(newoutfilename + ".fit", header=True, index=False)
    sys.stdout.flush()
    sys.stderr.flush()


def write_freqfile(mutfile: str, freqdf: df) -> None:
    leadup, base = os.path.split(mutfile)
    leadup = os.path.join(leadup.split("pops")[0])
    newoutfilename = os.path.join(leadup, "freqs", base)
    if not os.path.exists(os.path.join(leadup, "freqs")):
        os.makedirs(os.path.join(leadup, "freqs"), exist_ok=True)

    freqdf.to_csv(newoutfilename + ".freqs", header=True, index=False)


def fit_gen(mutfile: str) -> None:
    """
    Worker function for multiprocessing.

    Args:
        mutfile (str): SLiM output file to parse.
    """
    mut_df = get_muts(mutfile)

    if mut_df is not None:
        write_fitfile(mut_df, mutfile)
        write_freqfile(mutfile, mut_df)
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
        default=mp.cpu_count(),
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

    #for i in popfiles:
    #    print(i)
    #    fit_gen(i)
    #    sys.exit()


#

if __name__ == "__main__":
    main()
