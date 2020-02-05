import numpy as np
import sys, os, gzip, random
import sklearn.model_selection
from collections import Counter

inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, outFileName = sys.argv[1:]
maxSnps = int(maxSnps)
sampleSize1PerTimeStep=int(sampleSize1PerTimeStep)
sampleSize2PerTimeStep=int(sampleSize2PerTimeStep)

def getMostCommonHapInLastBunch(haps, sampleSize2PerTimeStep):
    counts = Counter(haps[-sampleSize2PerTimeStep:])
    if len(counts) == 1:
        return counts.most_common(1)[0][0]

    winner, runnerUp = counts.most_common(2)

    if winner[1] == runnerUp[1]:
        initCounts = Counter(haps[:sampleSize2PerTimeStep])
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

def getJointHapFreqsForTimePoint(currSample, hapToIndex, sampleSize1PerTimeStep, sampleSize2PerTimeStep, maxPossibleHaps):
    hfs = [(0, 0)]*maxPossibleHaps
    pop1 = currSample[:sampleSize1PerTimeStep]
    pop2 = currSample[sampleSize1PerTimeStep:sampleSize1PerTimeStep+sampleSize2PerTimeStep]
    for hap in hapToIndex:
        hfs[hapToIndex[hap]] = (pop1.count(hap), pop2.count(hap))
    hfs = [[x[0]/len(pop1),x[1]/len(pop2)] for x in hfs]
    return hfs

def seqDist(hap1, hap2):
    assert len(hap1) == len(hap2)
    numDiffs = 0
    for i in range(len(hap1)):
        if hap1[i] != hap2[i]:
            numDiffs += 1
    return numDiffs

def getMostSimilarHapIndex(hapBag, winningFinalHap):
    minDist = float('inf')
    for i in range(len(hapBag)):
        dist = seqDist(hapBag[i], winningFinalHap)
        if dist < minDist:
            minDist = dist
            minIndex = i
    return minIndex

def getTimeSeriesHapFreqs(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep):
    winningFinalHap = getMostCommonHapInLastBunch(currHaps, sampleSize2PerTimeStep) #recipient pop is listed second
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

    if sampleSize1PerTimeStep+sampleSize2PerTimeStep == len(currHaps):
        hapFreqMat = getJointHapFreqsForTimePoint(currHaps, hapToIndex, sampleSize1PerTimeStep, sampleSize2PerTimeStep, len(currHaps))
    else:
        hapFreqMat = []
        for i in range(0, len(currHaps), sampleSize1PerTimeStep+sampleSize2PerTimeStep):
            currHapFreqs = getJointHapFreqsForTimePoint(currHaps[i:i+sampleSize1PerTimeStep+sampleSize2PerTimeStep], hapToIndex, sampleSize1PerTimeStep, sampleSize2PerTimeStep, len(currHaps))
            hapFreqMat.append(currHapFreqs)
    return hapFreqMat

def readMsData(msFileName, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep):
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
                    if len(hapMats) % 100 == 0:
                        sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                    hapMats.append(getTimeSeriesHapFreqs(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
                    readMode = 0
                else:
                    assert line[0] in ["0","1"]
                    currHaps.append(line[start:end])
        hapMats.append(getTimeSeriesHapFreqs(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
        sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
    return hapMats

classNameToLabel = {'soft': 3, 'hard': 1, 'neut': 0}
def readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, testProp=0.1, valProp=0.1):
    sfsX = []
    y = []
    for inFileName in os.listdir(inDir):
        sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split(".")[0]
        classLabel = classNameToLabel[className]
        currTimeSeriesHFS = readMsData(inDir + "/" + inFileName, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)

        y += [classLabel]*len(currTimeSeriesHFS)
        sfsX += currTimeSeriesHFS

    X = np.array(sfsX, dtype='float32')
    #if len(X.shape) == 4:
    #    X = X.transpose(0, 2, 3, 1)
    y = np.array(y, dtype='int8')

    trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
    testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

    return trainX, trainy, testX, testy, valX, valy

# assume that each input file is a separate class
trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)
print(trainX.shape, testX.shape, valX.shape)
print(len(trainy), len(testy), len(valy))

np.savez_compressed(outFileName, trainX=trainX, trainy=trainy, testX=testX, testy=testy, valX=valX, valy=valy)
