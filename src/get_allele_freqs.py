import multiprocessing as mp
import os, sys
from argparse import ArgumentParser
from glob import glob
import pandas as pd
from tqdm import tqdm
from frequency_increment_test import write_fitfile
import logging

df = pd.DataFrame


def remove_restarts(lines):
    gens = []
    out_idxs = []
    for idx, line in enumerate(lines):
        if "#OUT" in line[0]:
            gens.append(line[1])
            out_idxs.append(idx)
    min_gen = min(gens)
    gen_inds = [i for i, x in enumerate(gens) if x == min_gen]
    if (
        len(gen_inds) > 1
    ):  # Get indices of any duplicated gens - spans gens and out_idxs
        # Remove lines between the first and last occurence
        # Only want to keep the ones after the restart
        # Technically only restarts should happen at the dumpfile ggen, but this is flexible for anything I suppose
        # print("\n".join(lines[out_idxs[gen_inds[0]] : out_idxs[gen_inds[-1]]]))
        del lines[0 : out_idxs[max(gen_inds)]]
    return lines


def get_sampling_info(cleaned_lines):
    samp_sizes = [int(line[4]) for line in cleaned_lines if "#OUT:" in line]
    gens_sampled = [int(line[1]) for line in cleaned_lines if "#OUT:" in line]

    return samp_sizes, gens_sampled


def df_from_lines(header_list, mutlist):
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

    return cleaned_mutdf


def calc_freq(gen_dfs, samp_sizes, gens_sampled):
    # Iterate through and get freqs
    for i in range(len(gen_dfs)):
        gen_dfs[i]["sampsize"] = samp_sizes[i]
        gen_dfs[i]["gen_sampled"] = gens_sampled[i]
        gen_dfs[i]["freq"] = gen_dfs[i]["prevalence"] / gen_dfs[i]["sampsize"]

    return gen_dfs


def get_muts(filename):
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
    samp_sizes, gens_sampled = get_sampling_info(cleaned_lines)

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
                cleaned_mutdf = df_from_lines(header_list, mutlist)
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
    cleaned_mutdf = df_from_lines(header_list, mutlist)
    gen_dfs.append(cleaned_mutdf)

    gen_dfs = calc_freq(gen_dfs, samp_sizes, gens_sampled)

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


def write_freqfile(mutfile, freqdf):
    leadup, base = os.path.split(mutfile)
    leadup = os.path.join(leadup.split("pops")[0])
    newoutfilename = os.path.join(leadup, "freqs", base)
    if not os.path.exists(os.path.join(leadup, "freqs")):
        os.makedirs(os.path.join(leadup, "freqs"), exist_ok=True)

    freqdf.to_csv(newoutfilename + ".freqs", header=True, index=False)


def worker(mutfile):
    """
    Worker function for multiprocessing.

    Args:
        mutfile (str): SLiM output file to parse.
    """
    mut_df = get_muts(mutfile)

    if mut_df is not None:
        try:
            write_fitfile(mut_df, mutfile)
            write_freqfile(mutfile, mut_df)
        except Exception as e:
            logging.warning(f"Couldn't write fit/freq files because of {e}")
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
    popfiles = list(glob(os.path.join(agp.in_dir, "*.pop")))
    print(f"Using {agp.threads} threads.")

    with mp.Pool(agp.threads) as p:
        for _ in tqdm(
            p.imap_unordered(worker, popfiles),
            total=len(popfiles),
            desc=f"Creating freqfiles in {agp.in_dir}",
        ):
            pass

    # for i in popfiles:
    #    print(i)
    #    worker(i)
    #    sys.exit()


if __name__ == "__main__":
    main()
