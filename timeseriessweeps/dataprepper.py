import gzip
import os
import random
import sys

import numpy as np
import sklearn.model_selection


class DataPrepper():
    """General format is 
    1. Object gets instantiated
    2. obj.writeNPZ is called, which in turn calls:
        -> readAndSplitMsData
            -> readMsData
                -> getTimeSeries<type>
                    -> calc<type>ForTimePoint
                        -> Other utility functions required for that type


    TODO:
        - currHaps variable unbound in all methods after the if/else statement
        - Re-parameterize readMsData and readAndSplitMsData so they are inherited \
            rather than being overwrote in each class for 2 variable changes

    """
    def __init__(self, inDir, maxSnps, sampleSizePerTimeStep, outFileName):
        self.inDir = inDir
        self.maxSnps = int(maxSnps)
        self.outFileName = outFileName
        self.sampleSizePerTimeStep = sampleSizePerTimeStep
        self.classNameToLabel = {'hard': 1, 'neut': 0, 'soft': 2}

    def writeNPZ(self, sampleSizePerTimeStep): #Should be callable by all inherited objs
        # assume that each input file is a separate class
        (trainX, trainy, 
         testX, testy, 
         valX, valy) = self.readAndSplitMsData(self.inDir, 
                                              self.maxSnps, 
                                              sampleSizePerTimeStep)
        
        print("Train Set Size   Test Set Size   Val Set Size")
        #print(trainX.shape, testX.shape, valX.shape)
        print(len(trainy), "\t", len(testy), "\t", len(valy))

        np.savez_compressed(self.outFileName, trainX=trainX, trainy=trainy,
                            testX=testX, testy=testy, valX=valX, valy=valy)

    def seqDist(self, hap1, hap2):
        assert len(hap1) == len(hap2)
        numDiffs = 0
        for i in range(len(hap1)):
            if hap1[i] != hap2[i]:
                numDiffs += 1
        return numDiffs


class SFSPrepper(DataPrepper):
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
        
    def getTimeSeriesSFS(self, currHaps, sampleSizePerTimeStep):
        sampleSize = len(currHaps)

        if sampleSize == sampleSizePerTimeStep:
            sfsMat = self.calcSFSForTimePoint(currHaps)
        else:
            sfsMat = []
            for i in range(0, sampleSize, sampleSizePerTimeStep):
                currSFS = self.calcSFSForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep])
                sfsMat.append(currSFS)
        return sfsMat

    def readMsData(self, msFileName):
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
                        if numSnps >= self.maxSnps:
                            start = int((numSnps - self.maxSnps) / 2)
                            end = start + self.maxSnps
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
                        hapMats.append(self.getTimeSeriesSFS(currHaps, self.sampleSizePerTimeStep))
                        readMode = 0
                    else:
                        assert line[0] in ["0", "1"]
                        currHaps.append(list(line[start:end])) #TODO currhaps/start/end are unbound when used here
            hapMats.append(self.getTimeSeriesSFS(currHaps, self.sampleSizePerTimeStep))
            sys.stderr.write(
                "read {} hap matrices. done!\n".format(len(hapMats)))
        return hapMats

    def readAndSplitMsData(self, sampleSizePerTimeStep, testProp=0.1, valProp=0.1):
        sfsX = []
        y = []

        for inFileName in os.listdir(self.inDir):
            sys.stderr.write("reading {}\n".format(inFileName))
            className = inFileName.split(".")[0]
            classLabel = self.classNameToLabel[className]
            currTimeSeriesSFS = self.readMsData(self.inDir + "/" + inFileName, 
                                                self.maxSnps, 
                                                sampleSizePerTimeStep)

            y += [classLabel]*len(currTimeSeriesSFS)
            sfsX += currTimeSeriesSFS

        X = np.array(sfsX, dtype='float32')
        if len(X.shape) == 3:
            X = X.transpose(0, 2, 1)
        y = np.array(y, dtype='int8')

        trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, 
                                                                            stratify=y, 
                                                                            test_size=testProp)

        testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, 
                                                                            stratify=tvy, 
                                                                            test_size=valProp/(valProp+testProp))

        return trainX, trainy, testX, testy, valX, valy


class JSFSPrepper(DataPrepper):#Looks like it'll be inheriting most stuff from SFS
    def countFreqOfSegsite(currSample, segsite):
        return sum([int(x[segsite]) for x in currSample])

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
                    freq = self.countFreqOfSegsite(
                        currHaps[i:i+sampleSizePerTimeStep], j)
                    freqs.append(freq)
            tsJSFS[tuple(freqs)] += 1
        return tsJSFS

    def readMsData(self, msFileName):
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
                        if numSnps >= self.maxSnps:
                            start = int((numSnps - self.maxSnps) / 2)
                            end = start + self.maxSnps
                        else:
                            start, end = 0, numSnps
                elif readMode == 1:
                    line = line.strip()
                    if not line:
                        pass
                    elif line.startswith("//"):
                        if len(hapMats) % 100 == 0:
                            sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                        hapMats.append(self.getTimeSeriesJSFS(currHaps, self.sampleSizePerTimeStep))
                        readMode = 0
                    else:
                        assert line[0] in ["0","1"]
                        currHaps.append(list(line[start:end]))
            hapMats.append(self.getTimeSeriesJSFS(currHaps, self.sampleSizePerTimeStep))
            sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
        return hapMats

class AliPrepper(DataPrepper):
    def getMinimalEditDistanceHap(self, ali):
        distances = np.zeros((len(ali), len(ali)))
        for i in range(len(ali)-1):
            for j in range(i+1, len(ali)):
                d = self.seqDist(ali[i], ali[j])
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
            reversed(sorted(ali, key=lambda hap: self.seqDist(hap, targHap))))
        return sortedAli

    def sortAliAllTimePoints(self, alis):
        newAlis = []
        targHap = self.getMinimalEditDistanceHap(alis[-1])
        for i in range(len(alis)):
            newAli = self.sortAliBySimilarityToHap(alis[i], targHap)
            newAlis.append(newAli)
        return newAlis

    def sortAliOneTimePoint(self, ali):
        targHap = self.getMinimalEditDistanceHap(ali)
        newAli = self.sortAliBySimilarityToHap(ali, targHap)
        return newAli

    def getAliForTimePoint(self, currSample):
        numChroms = len(currSample)
        a = np.array(currSample, dtype='int8')
        b = np.full((numChroms, self.maxSnps), -2, dtype='int8')
        b[:, :a.shape[1]] = a
        return b

    def getTimeSeriesAlis(self, currHaps, sampleSizePerTimeStep):
        sampleSize = len(currHaps)

        if sampleSize == sampleSizePerTimeStep:
            return self.sortAliOneTimePoint(getAliForTimePoint(currHaps))
        else:
            alis = []
            for i in range(0, sampleSize, sampleSizePerTimeStep):
                currAli = self.getAliForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep])
                alis.append(currAli)
            return self.sortAliAllTimePoints(alis)

    def readMsData(self, msFileName):
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
                        if numSnps >= self.maxSnps:
                            start = int((numSnps - self.maxSnps) / 2)
                            end = start + self.maxSnps
                        else:
                            start, end = 0, numSnps
                elif readMode == 1:
                    line = line.strip()
                    if not line:
                        pass
                    elif line.startswith("//"):
                        if len(hapMats) % 100 == 0:
                            sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                        hapMats.append(self.getTimeSeriesAlis(currHaps, self.sampleSizePerTimeStep))
                        readMode = 0
                    else:
                        assert line[0] in ["0","1"]
                        currHaps.append(list(line[start:end]))
            hapMats.append(self.getTimeSeriesAlis(currHaps, self.sampleSizePerTimeStep))
            sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
        return hapMats

    def readAndSplitMsData(self, testProp=0.1, valProp=0.1):
        aliX = []
        y = []
        for inFileName in os.listdir(self.inDir):
            sys.stderr.write("reading {}\n".format(inFileName))
            className = inFileName.split(".")[0]
            classLabel = self.classNameToLabel[className]
            currAlis = self.readMsData(self.inDir + "/" + inFileName, self.maxSnps, self.sampleSizePerTimeStep)

            y += [classLabel]*len(currAlis)
            aliX += currAlis

        X = np.array(aliX, dtype='int8')
        y = np.array(y, dtype='int8')

        trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
        testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

        return trainX, trainy, testX, testy, valX, valy


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
            dist = self.seqDist(haps[i], targHap)
            if dist < minDist:
                minDist = dist
                minIndex = i
                return minIndex
            else:
                return None #This should be clarified, minIndex unbound if not

    def getTimeSeriesHapFreqs(self, currHaps, sampleSizePerTimeStep):
        winningFinalHap = self.getMostCommonHapInLastBunch(
            currHaps, sampleSizePerTimeStep)
        hapBag = list(set(currHaps))
        hapBag.pop(hapBag.index(winningFinalHap))

        hapToIndex = {}
        index = 0
        hapToIndex[winningFinalHap] = index

        while len(hapBag) > 0:
            index += 1
            mostSimilarHapIndex = self.getMostSimilarHapIndex(
                hapBag, winningFinalHap)
            mostSimilarHap = hapBag.pop(mostSimilarHapIndex)
            hapToIndex[mostSimilarHap] = index

        if sampleSizePerTimeStep == len(currHaps):
            hapFreqMat = self.getHapFreqsForTimePoint(
                currHaps, hapToIndex, len(currHaps))
        else:
            hapFreqMat = []
            for i in range(0, len(currHaps), sampleSizePerTimeStep):
                currHapFreqs = self.getHapFreqsForTimePoint(
                    currHaps[i:i+sampleSizePerTimeStep], hapToIndex, len(currHaps))
                hapFreqMat.append(currHapFreqs)
        return hapFreqMat

    def readMsData(self, msFileName):
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
                        if numSnps >= self.maxSnps:
                            start = int((numSnps - self.maxSnps) / 2)
                            end = start + self.maxSnps
                        else:
                            start, end = 0, numSnps
                elif readMode == 1:
                    line = line.strip()
                    if not line:
                        pass
                    elif line.startswith("//"):
                        if len(hapMats) % 100 == 0:
                            sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                        hapMats.append(self.getTimeSeriesHapFreqs(currHaps, self.sampleSizePerTimeStep))
                        readMode = 0
                    else:
                        assert line[0] in ["0","1"]
                        currHaps.append(line[start:end])
            hapMats.append(self.getTimeSeriesHapFreqs(currHaps, self.sampleSizePerTimeStep))
            sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
        return hapMats

    def readAndSplitMsData(self, testProp=0.1, valProp=0.1):
        sfsX = []
        y = []
        for inFileName in os.listdir(self.inDir):
            sys.stderr.write("reading {}\n".format(inFileName))
            className = inFileName.split(".")[0]
            classLabel = self.classNameToLabel[className]
            currTimeSeriesHFS = self.readMsData(self.inDir + "/" + inFileName, self.maxSnps, self.sampleSizePerTimeStep)

            y += [classLabel]*len(currTimeSeriesHFS)
            sfsX += currTimeSeriesHFS

        X = np.array(sfsX, dtype='float32')
        if len(X.shape) == 3:
            X = X.transpose(0, 2, 1)
        y = np.array(y, dtype='int8')

        trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
        testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

        return trainX, trainy, testX, testy, valX, valy
