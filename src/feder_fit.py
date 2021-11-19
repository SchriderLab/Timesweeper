import os
import sys
import warnings
import numpy as np
import pandas as pd
from logging import warning
from scipy.stats import ttest_1samp
import allele_freq_mat as afm

warnings.filterwarnings("ignore", category=RuntimeWarning)
# https://www.genetics.org/content/196/2/509

df = pd.DataFrame


def getRescaledIncs(freqs, gens):
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


def write_fitfile(mutdf, outfilename):
    """
    Writes the resulting FIt values to csv.
    #TODO should probably separate this out

    Args:
        mutdf (df): DF with all mutations from pop file, cleaned and binned.
        outfilename (str): Name of csv file to output.
    """
    _, sweep = afm.get_file_label(outfilename)

    mut_dict = {
        "mut_ID": [],
        "mut_type": [],
        "location": [],
        "fit_t": [],
        "fit_p": [],
        "selection_detected": [],
    }

    # Take center mutation, identical to AFS method
    center_bp = afm.get_middle_bp(physLen=1e5)
    center_win = afm.get_window_idxs(mutdf, center_bp, sweep)
    sel_bp = center_win[int(len(center_win) / 2)]
    subdf = mutdf[mutdf["bp"] == sel_bp].reset_index()

    if len(subdf) > 2:
        try:
            fit_t, fit_p = fit(list(subdf["freq"]), list(subdf["gen_sampled"]))
            if not np.isnan(fit_t):
                mut_dict["mut_ID"].append(int(subdf.perm_ID[0]))
                mut_dict["mut_type"].append(subdf.mut_type[0])
                mut_dict["location"].append(subdf.bp[0])
                mut_dict["fit_t"].append(float(fit_t))
                mut_dict["fit_p"].append(fit_p)
                if fit_p <= 0.05:
                    mut_dict["selection_detected"].append(1)
                else:
                    mut_dict["selection_detected"].append(0)

        except:
            warning("Can't calculate FIT because not enough timepoints.")
            pass

    outdf = df(mut_dict)

    leadup, base = os.path.split(outfilename)
    leadup = os.path.join(leadup.split("pops")[0])

    newoutfilename = os.path.join(leadup, "fit", base)
    if not os.path.exists(os.path.join(leadup, "fit")):
        os.makedirs(os.path.join(leadup, "fit"), exist_ok=True)

    outdf.to_csv(newoutfilename + ".fit", header=True, index=False)
    sys.stdout.flush()
    sys.stderr.flush()

