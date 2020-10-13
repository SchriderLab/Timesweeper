import numpy as np
import sys
import os
import gzip
import random
import sklearn.model_selection


class DataPrepper():
    def __init__(self, inDir, maxSnps, sampleSizePerTimeStep, outFileName, joint):
        self.inDir = inDir
        self.maxSnps = int(maxSnps)
        self.sampleSizePerTimeStep = int(sampleSizePerTimeStep)
        self.outFileName = outFileName
        self.joint = joint

    def readMsData(self, msFileName, maxSnps, sampleSizePerTimeStep):
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
                            sys.stderr.write(
                                "read {} hap matrices\r".format(len(hapMats)))
                        hapMats.append(getTimeSeriesSFS(
                            currHaps, sampleSizePerTimeStep))
                        readMode = 0
                    else:
                        assert line[0] in ["0", "1"]
                        currHaps.append(list(line[start:end]))
            hapMats.append(getTimeSeriesSFS(currHaps, sampleSizePerTimeStep))
            sys.stderr.write(
                "read {} hap matrices. done!\n".format(len(hapMats)))
        return hapMats

    def readAndSplitMsData(self, inDir, maxSnps, sampleSizePerTimeStep, testProp=0.1, valProp=0.1):
        sfsX = []
        y = []
        classNameToLabel = {'hard': 1, 'neut': 0, 'soft': 2}

        for inFileName in os.listdir(inDir):
            sys.stderr.write("reading {}\n".format(inFileName))
            className = inFileName.split(".")[0]
            classLabel = classNameToLabel[className]
            currTimeSeriesSFS = readMsData(
                inDir + "/" + inFileName, maxSnps, sampleSizePerTimeStep)

            y += [classLabel]*len(currTimeSeriesSFS)
            sfsX += currTimeSeriesSFS

        X = np.array(sfsX, dtype='float32')
        if len(X.shape) == 3:
            X = X.transpose(0, 2, 1)
        y = np.array(y, dtype='int8')

        trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(
            X, y, stratify=y, test_size=testProp)
        testX, valX, testy, valy = sklearn.model_selection.train_test_split(
            tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

        return trainX, trainy, testX, testy, valX, valy

    def seqDist(self, hap1, hap2):
        assert len(hap1) == len(hap2)
        numDiffs = 0
        for i in range(len(hap1)):
            if hap1[i] != hap2[i]:
                numDiffs += 1
        return numDiffs

    def writeNPZ(self):
        # assume that each input file is a separate class
        trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(
            inDir, maxSnps, sampleSizePerTimeStep)
        print(trainX.shape, testX.shape, valX.shape)
        print(len(trainy), len(testy), len(valy))

        np.savez_compressed(outFileName, trainX=trainX, trainy=trainy,
                            testX=testX, testy=testy, valX=valX, valy=valy)



class SFSPrepper(DataPrepper):
    def getTimeSeriesSFS(self, currHaps, sampleSizePerTimeStep):
        sampleSize = len(currHaps)

        if sampleSize == sampleSizePerTimeStep:
            sfsMat = calcSFSForTimePoint(currHaps)
        else:
            sfsMat = []
            for i in range(0, sampleSize, sampleSizePerTimeStep):
                currSFS = calcSFSForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep])
                sfsMat.append(currSFS)
        return sfsMat

    def calcSFSForTimePoint(self, currSample):
        sfs = np.zeros(len(currSample)-1)
        for site in range(len(currSample[0])):
            daCount = sum([int(currSample[x][site])
                           for x in range(0, len(currSample))])
            if daCount > 0 and daCount < len(currSample):
                sfs[daCount-1] += 1
        denom = np.sum(sfs)
        sfs = sfs/denom
        return sfs


class SFSPrepper(DataPrepper):
    def getTimeSeriesJSFS(self, currHaps, sampleSizePerTimeStep):
        totSteps = int(len(currHaps)/sampleSizePerTimeStep)
        numSteps = 3
        assert numSteps*sampleSizePerTimeStep <= len(currHaps)
        midStep = totSteps/2
        shape = [sampleSizePerTimeStep+1]*numSteps
        tsJSFS = np.zeros(shape)

        for j in range(len(currHaps[0])):
            freqs = []
            for i in range(0, len(currHaps), sampleSizePerTimeStep):
                if i in [0, midStep, numSteps-1]:
                    freq = countFreqOfSegsite(
                        currHaps[i:i+sampleSizePerTimeStep], j)
                    freqs.append(freq)
            tsJSFS[tuple(freqs)] += 1
        return tsJSFS

    def countFreqOfSegsite(self, currSample, segsite):
        return sum([int(x[segsite]) for x in currSample])


class AliPrepper(DataPrepper):
    def getMinimalEditDistanceHap(self, ali):
        distances = np.zeros((len(ali), len(ali)))
        for i in range(len(ali)-1):
            for j in range(i+1, len(ali)):
                d = seqDist(ali[i], ali[j])
                distances[i, j] = d
                distances[j, i] = d

        minDistSum = len(ali[0])*len(ali) + 1
        minDistHapIndex = -1
        for i in range(len(ali)):
            currDistSum = sum(distances[i])
            if currDistSum < minDistSum:
                minDistSum = currDistSum
                minDistHapIndex = i
        assert minDistHapIndex >= 0
        return ali[minDistHapIndex]

    def sortAliBySimilarityToHap(self, ali, targHap):
        sortedAli = list(
            reversed(sorted(ali, key=lambda hap: seqDist(hap, targHap))))
        return sortedAli

    def sortAliAllTimePoints(self, alis):
        newAlis = []
        targHap = getMinimalEditDistanceHap(alis[-1])
        for i in range(len(alis)):
            newAli = sortAliBySimilarityToHap(alis[i], targHap)
            newAlis.append(newAli)
        return newAlis

    def sortAliOneTimePoint(self, ali):
        targHap = getMinimalEditDistanceHap(ali)
        newAli = sortAliBySimilarityToHap(ali, targHap)
        return newAli

    def getAliForTimePoint(self, currSample, maxSnps):
        numChroms = len(currSample)
        a = np.array(currSample, dtype='int8')
        b = np.full((numChroms, maxSnps), -2, dtype='int8')
        b[:, :a.shape[1]] = a
        return b

    def getTimeSeriesAlis(self, currHaps, sampleSizePerTimeStep):
        sampleSize = len(currHaps)

        if sampleSize == sampleSizePerTimeStep:
            return sortAliOneTimePoint(getAliForTimePoint(currHaps))
        else:
            alis = []
            for i in range(0, sampleSize, sampleSizePerTimeStep):
                currAli = getAliForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep])
                alis.append(currAli)
            return sortAliAllTimePoints(alis)


class HapsPrepper(DataPrepper):
    def getMostCommonHapInLastBunch(self, haps, sampleSizePerTimeStep):
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

    def getHapFreqsForTimePoint(self, currSample, hapToIndex, maxPossibleHaps):
        hfs = [0]*maxPossibleHaps
        for hap in currSample:
            hfs[hapToIndex[hap]] += 1

        hfs = [x/len(currSample) for x in hfs]
        return hfs

    def getMostSimilarHapIndex(self, haps, targHap):
        minDist = float('inf')
        for i in range(len(haps)):
            dist = seqDist(haps[i], targHap)
            if dist < minDist:
                minDist = dist
                minIndex = i
        return minIndex

    def getTimeSeriesHapFreqs(self, currHaps, sampleSizePerTimeStep):
        winningFinalHap = getMostCommonHapInLastBunch(
            currHaps, sampleSizePerTimeStep)
        hapBag = list(set(currHaps))
        hapBag.pop(hapBag.index(winningFinalHap))

        hapToIndex = {}
        index = 0
        hapToIndex[winningFinalHap] = index

        while len(hapBag) > 0:
            index += 1
            mostSimilarHapIndex = getMostSimilarHapIndex(
                hapBag, winningFinalHap)
            mostSimilarHap = hapBag.pop(mostSimilarHapIndex)
            hapToIndex[mostSimilarHap] = index

        if sampleSizePerTimeStep == len(currHaps):
            hapFreqMat = getHapFreqsForTimePoint(
                currHaps, hapToIndex, len(currHaps))
        else:
            hapFreqMat = []
            for i in range(0, len(currHaps), sampleSizePerTimeStep):
                currHapFreqs = getHapFreqsForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep], hapToIndex, len(currHaps))
                hapFreqMat.append(currHapFreqs)
        return hapFreqMat
