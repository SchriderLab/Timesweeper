import numpy as np
import sys, os, gzip, random
import sklearn.model_selection
from collections import Counter
from glob import glob
from tqdm import tqdm


def getMostCommonHapInLastBunch(haps, sampleSizePerTimeStep):
    counts = Counter(haps[-sampleSizePerTimeStep:])
    if len(counts) == 1:
        return counts.most_common(1)[0][0]

    winner, runnerUp = counts.most_common(2)

    if winner[1] == runnerUp[1]:
        initCounts = Counter(haps[:sampleSizePerTimeStep])
        maxFreqChange = float("-inf")
        for hap in counts:
            if counts[hap] == winner[1]:
                freqChange = counts[hap] - initCounts.get(hap, 0)
                if freqChange > maxFreqChange:
                    maxFreqChange = freqChange
                    maxFreqChangeHap = hap

        return maxFreqChangeHap

    else:
        return counts.most_common(1)[0][0]


def getHapFreqsForTimePoint(currSample, hapToIndex, maxPossibleHaps):
    hfs = [0] * maxPossibleHaps
    for hap in currSample:
        hfs[hapToIndex[hap]] += 1

    hfs = [x / len(currSample) for x in hfs]

    return hfs


def seqDist(hap1, hap2):
    assert len(hap1) == len(hap2)
    numDiffs = 0
    for i in range(len(hap1)):
        if hap1[i] != hap2[i]:
            numDiffs += 1

    return numDiffs


def getMostSimilarHapIndex(haps, targHap):
    minDist = float("inf")
    for i in range(len(haps)):
        dist = seqDist(haps[i], targHap)
        if dist < minDist:
            minDist = dist
            minIndex = i

    return minIndex


def getTimeSeriesHapFreqs(currHaps, sampleSizePerTimeStep):
    winningFinalHap = getMostCommonHapInLastBunch(currHaps, sampleSizePerTimeStep)
    hapBag = list(set(currHaps))
    hapBag.pop(hapBag.index(winningFinalHap))

    hapToIndex = {}
    index = 0
    hapToIndex[winningFinalHap] = index

    while len(hapBag) > 0:
        index += 1
        mostSimilarHapIndex = getMostSimilarHapIndex(hapBag, winningFinalHap)
        mostSimilarHap = hapBag.pop(mostSimilarHapIndex)
        hapToIndex[mostSimilarHap] = index

    if sampleSizePerTimeStep == len(currHaps):
        hapFreqMat = getHapFreqsForTimePoint(currHaps, hapToIndex, len(currHaps))
    else:
        hapFreqMat = []
        for i in range(0, len(currHaps), sampleSizePerTimeStep):
            currHapFreqs = getHapFreqsForTimePoint(
                currHaps[i : i + sampleSizePerTimeStep], hapToIndex, len(currHaps)
            )
            hapFreqMat.append(currHapFreqs)

    return hapFreqMat


def readMsData(msFileName, maxSnps, sampleSizePerTimeStep):
    if msFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(msFileName, "rt") as msFile:
        readMode = 0
        hapMats = []
        for line in msFile:
            if readMode == 0:
                if line.startswith("positions:"):
                    readMode = 1
                    currHaps = []
                elif line.startswith("segsites:"):
                    numSnps = int(line.strip().split()[-1])
                    if numSnps >= maxSnps:
                        start = int((numSnps - maxSnps) / 2)
                        end = start + maxSnps
                    else:
                        start, end = 0, numSnps
            elif readMode == 1:
                line = line.strip()
                if not line:
                    pass
                elif line.startswith("//"):
                    # if len(hapMats) % 100 == 0:
                    # sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                    hapMats.append(
                        getTimeSeriesHapFreqs(currHaps, sampleSizePerTimeStep)
                    )
                    readMode = 0
                else:
                    if line.strip().startswith("blarg"):
                        pass
                    if line[0] in ["0", "1"]:
                        currHaps.append(line[start:end])
        hapMats.append(getTimeSeriesHapFreqs(currHaps, sampleSizePerTimeStep))
        # sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))

    return hapMats


def readAndSplitMsData(
    inDir, maxSnps, sampleSizePerTimeStep, testProp=0.1, valProp=0.1
):
    classNameToLabel = {"hard": 0, "neut": 1, "soft": 2}

    sfsX = []
    y = []
    for inFileName in tqdm(glob(inDir + "/*msCombo")):
        print(inFileName)
        # try:
        # sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split("/")[-4]
        classLabel = classNameToLabel[className]
        currTimeSeriesHFS = readMsData(inFileName, maxSnps, sampleSizePerTimeStep)

        y += [classLabel] * len(currTimeSeriesHFS)
        sfsX += currTimeSeriesHFS
        # except:
        #    continue
    print(len(sfsX))
    print(set(y))
    X = np.array(sfsX, dtype="float32")
    if len(X.shape) == 3:
        X = X.transpose(0, 2, 1)
    print(X.shape)
    y = np.array(y, dtype="int8")

    return X, y


def main():
    # maxSnps = # of snps we take from center
    inDir, outFileName = sys.argv[1:]
    maxSnps = 50  # int(maxSnps)
    sampleSizePerTimeStep = 20  # int(sampleSizePerTimeStep)

    X, y = readAndSplitMsData(inDir, maxSnps, sampleSizePerTimeStep)
    print(X.shape)
    print(len(y))

    np.savez_compressed(
        outFileName,
        X=X,
        y=y,
    )


if __name__ == "__main__":
    main()