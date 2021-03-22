import multiprocessing as mp
import sys
from glob import glob
from math import sqrt
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
import os
from tqdm import tqdm

# https://www.genetics.org/content/196/2/509

"""
L: Number of sampled points
v[i]: Allele frequency at point i in L
t[i]: Generation at point i in L
"""


def calc_FIT(freqs: List[float], gens: List[int]) -> Tuple[float, float]:
    L = len(gens)
    Yi_list = []
    for i in range(1, L):
        Yi_list.append(
            (freqs[i] - freqs[i - 1])
            / (sqrt((2 * freqs[i - 1]) * (1 - freqs[i - 1]) * (gens[i] - gens[i - 1])))
        )

    t_stat, p_val = ttest_1samp(Yi_list, popmean=0)

    return t_stat, p_val


def get_muts(filename: str) -> pd.DataFrame:
    with open(filename, "r") as mutfile:
        rawlines = [i.strip().split(" ") for i in mutfile.readlines()]

    gen_dfs = []
    mutlist = []

    for i in range(len(rawlines)):
        if "#OUT:" in rawlines[i][0]:
            sampgen = int(rawlines[i][1])
            popsize = int(rawlines[i][-1])  # Should be 20

            if sampgen == 10000:  # Restart!
                gen_dfs = []
                mutlist = []
                continue

            mutdf = pd.DataFrame(
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
            mutdf["gen_sampled"] = sampgen
            mutdf["freq"] = mutdf["prevalence"].astype(int) / popsize
            gen_dfs.append(mutdf)
            mutlist = []  # Reset for the next round

        elif (
            (len(rawlines[i]) == 9)
            and (rawlines[i][2] == "m1" or rawlines[i][2] == "m2")
            and ("p1" not in rawlines[i][1])
        ):
            mutlist.append(rawlines[i])
        else:
            continue

    try:
        clean_gens_df = (
            pd.concat(gen_dfs, axis=0, ignore_index=True)
            .drop_duplicates()
            .reset_index()
        )
        clean_gens_df["bp"] = clean_gens_df["bp"].astype(int)

        return clean_gens_df

    except:
        print(f"Couldn't process {filename}.")


def write_fitfile(mutdf: pd.DataFrame, outfilename: str) -> None:
    cut_labs = list(range(11))
    cut_bins = np.linspace(
        np.min(mutdf["bp"]), np.max(mutdf["bp"]), 12
    )  # Divide into windows across the genomic region
    mutdf["window"] = pd.cut(mutdf["bp"], bins=cut_bins, labels=cut_labs)

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
        # try:
        subdf = mutdf[mutdf["perm_ID"] == mutID].reset_index()
        # For some reason gen 1000 gets sampled twice, is just 1200 so drop it
        # subdf.drop_duplicates(subset="gen_sampled", keep="first", inplace=True)

        print(subdf)

        if len(subdf) > 2:
            try:
                fit_t, fit_p = calc_FIT(list(subdf["freq"]), list(subdf["gen_sampled"]))

                if not np.isnan(fit_t):
                    mut_dict["mut_ID"].append(int(mutID))
                    mut_dict["mut_type"].append(subdf.mut_type[0])
                    mut_dict["location"].append(subdf.bp[0])
                    mut_dict["window"].append(subdf.window[0])
                    mut_dict["fit_t"].append(fit_t)
                    mut_dict["fit_p"].append(fit_p)
                    if fit_p <= 0.05:
                        mut_dict["selection_detected"].append(1)
                    else:
                        mut_dict["selection_detected"].append(0)

            except:
                continue

        else:
            continue
        # except Exception as e:
        #    print(e)
        #    continue

    outdf = pd.DataFrame(mut_dict)
    newfilename = outfilename + ".fit"
    outdf.to_csv(newfilename, header=True, index=False)

    print(newfilename)
    sys.stdout.flush()
    sys.stderr.flush()


def fit_gen(mutfile: str) -> None:
    # Exists in case of multiprocessing implementation
    # if not os.path.exists(mutfile):
    mut_df = get_muts(mutfile)
    if mut_df is not None:
        write_fitfile(mut_df, mutfile)
    else:
        pass


def main():
    target_dirs = glob(os.path.join(sys.argv[1], "muts/*/*.pop"))
    ts_files = [i for i in target_dirs if "1Samp" not in i]

    # for i in tqdm(ts_files):
    #    print(i)
    #    fit_gen(i)

    # print("Done with {}, no errors.".format(ts_files))

    with mp.Pool(mp.cpu_count()) as p:
        p.map(fit_gen, ts_files)


if __name__ == "__main__":
    main()
