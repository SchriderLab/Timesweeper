import warnings

import pandas as pd
from scipy.stats import ttest_1samp

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
