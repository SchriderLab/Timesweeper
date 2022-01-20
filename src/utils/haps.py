import numpy as np


def getTSHapFreqs(haps, samp_sizes):
    """
    Build haplotype frequency spectrum for a single timepoint.

    Args:
        haps (list[str]): List of haplotypes read from MS entry.

    Returns:
        list[float]: Haplotype frequency spectrum for a single timepoint sorted by the most common hap in entire set.
    """
    tsHapDicts = calcFreqs(haps, samp_sizes)
    winningHap = getHighestVelHap(tsHapDicts, haps)
    hapBag = list(set(haps))
    hapBag.pop(hapBag.index(winningHap))

    hapToIndex = {}
    index = 0
    hapToIndex[winningHap] = index

    while len(hapBag) > 0:
        index += 1
        mostSimilarHapIndex = getMostSimilarHapIndex(hapBag, winningHap)
        mostSimilarHap = hapBag.pop(mostSimilarHapIndex)
        hapToIndex[mostSimilarHap] = index

    hapFreqMat = []
    i = len(haps) - sum(samp_sizes)  # Skip restarts for sims
    for j in samp_sizes:
        currHapFreqs = getHapFreqsForTimePoint(
            haps[i : i + j], hapToIndex, sum(samp_sizes)
        )
        hapFreqMat.append(currHapFreqs)
        i += j

    return np.array(hapFreqMat)


def calcFreqs(haps, samp_sizes):
    """
    Calculates haplotype frequencies for all present haplotypes in each timepoint.

    Args:
        haps (list[str]): List of haplotypes, each is a 1D genotype string

    Returns:
        List[Dict[str: float]]: List of dictionaries for each timepoint in the form of {hap: freq}
    """
    allFreqs = []
    i = 0

    # Populate with 0s
    for j in samp_sizes:
        freqsInSamp = {}
        for hap in haps:
            freqsInSamp[hap] = 0.0
        for hap in haps[i : i + j]:
            freqsInSamp[hap] += 1 / j
        allFreqs.append(freqsInSamp)
        i += j

    return allFreqs


def getHighestVelHap(tsHapDicts, haps):
    """
    Calculates hap frequency differences between min/max freqs for all haps, returns hap with largest change.

    Args:
        tsHapDicts (List[Dict{str: float}]): List of timepoint dicts with structure {hap: freq}
        haps (List[str]): Haplotype strings

    Returns:
        str: Haplotype with biggest change in frequency from min to max freq.
    """
    freqChanges = {}
    for hap in haps:
        hapFreqs = [
            tsHapDicts[i][hap] for i in range(len(tsHapDicts)) if hap in tsHapDicts[i]
        ]
        # Ensures min is prior to max, set to max if max is first value or only
        max_val = max(hapFreqs)
        max_loc = hapFreqs.index(max_val)

        if len(list(hapFreqs[:max_loc])) == 0:
            min_val = 0
        else:
            min_val = min(hapFreqs[:max_loc])

        freqChanges[hap] = max_val - min_val

    return max(freqChanges, key=freqChanges.get)


def getHapFreqsForTimePoint(currSample, hapToIndex, maxPossibleHaps):
    """
    Create haplotype freq spectrum for a given sample and haplotype.

    Args:
        currSample (list[str]): Set of haplotypes in current time-sample.
        hapToIndex (int): Index of hap from hap-bag to calculate with.
        maxPossibleHaps (int): Number of total possible haplotypes.

    Returns:
        list[float]: Haplotype frequency spectrum for a given set of haplotypes.
    """
    hfs = [0] * maxPossibleHaps
    for hap in currSample:
        hfs[hapToIndex[hap]] += 1

    hfs = [x / len(currSample) for x in hfs]

    return hfs


def getMostSimilarHapIndex(haps, targHap):
    """
    Calculate distances between a current haplotype and all given haps in sample.

    Args:
        haps (list[str]): Haplotypes for a given sample point.
        targHap (str): Haplotype to calculate distance from.

    Returns:
        int: Index of the haplotype in the hapbag that has the min distance from targHap.
    """
    minDist = float("inf")
    for i in range(len(haps)):
        dist = seqDist(haps[i], targHap)
        if dist < minDist:
            minDist = dist
            minIndex = i

            return minIndex


def seqDist(hap1, hap2):
    """
    Calculates pairwise distance between two haplotypes

    Args:
        hap1 (list): Haplotype 1
        hap2 (list): Haplotype 2

    Returns:
        int: Number of pairwise differences (sequence distance) between haps
    """
    assert len(hap1) == len(hap2)
    numDiffs = 0
    for i in range(len(hap1)):
        if hap1[i] != hap2[i]:
            numDiffs += 1

    return numDiffs


def haps_to_strlist(haps_arr):
    haplist = haps_arr.tolist()
    return ["".join([str(int(i)) for i in hap]) for hap in haplist]
