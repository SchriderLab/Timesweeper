from math import sqrt
from typing import List, Tuple
import sys
from scipy.stats import ttest_1samp
import pandas as pd
from glob import glob
import os
from tqdm import tqdm
import multiprocessing as mp

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
            if i > 0:  # Don't have any muts first iteration or repeats
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
                mutdf["freq"] = mutdf["prevalence"].astype(int) / (popsize * 2)
                gen_dfs.append(mutdf)
                mutlist = []  # Reset for the next round

        elif "Populations:" in rawlines[i][0]:
            popsize = int(rawlines[i + 1][1])

        elif (
            (len(rawlines[i]) == 9)
            and (rawlines[i][2] == "m1" or rawlines[i][2] == "m2")
            and ("p1" not in rawlines[i][1])
        ):
            mutlist.append(rawlines[i])
            if rawlines[i][2] == "m2":
                print("FOUND ONE")
        else:
            continue

    clean_gens_df = (
        pd.concat(gen_dfs, axis=0, ignore_index=True).drop_duplicates().reset_index()
    )

    return clean_gens_df


def write_fitfile(mutdf: pd.DataFrame, outfilename: str) -> None:
    mut_dict = {
        "mut_ID": [],
        "mut_type": [],
        "fit_t": [],
        "fit_p": [],
        "selection_detected": [],
    }

    for mutID in mutdf["perm_ID"].unique():
        subdf = mutdf[mutdf["perm_ID"] == mutID]
        if len(subdf) > 2:
            try:
                fit_t, fit_p = calc_FIT(list(subdf["freq"]), list(subdf["gen_sampled"]))
                mut_dict["mut_ID"].append(int(mutID))
                mut_dict["mut_type"].append(list(subdf["mut_type"])[0])
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

    outdf = pd.DataFrame(mut_dict)

    newfilename = outfilename + ".fit"
    outdf.to_csv(newfilename, header=True, index=False)


def fit_gen(mutfile: str) -> None:
    mut_df = get_muts(mutfile)
    write_fitfile(mut_df, mutfile)


def main():
    with mp.Pool(mp.cpu_count()) as p:
        p.map(fit_gen, glob(os.path.join(sys.argv[1], "sims/*/muts/*/*.muts")))


if __name__ == "__main__":
    main()