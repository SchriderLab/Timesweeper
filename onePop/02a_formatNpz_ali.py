import numpy as np
import sys, os, gzip, random
import sklearn.model_selection

inDir, maxSnps, sampleSizePerTimeStep, outFileName = sys.argv[1:]
maxSnps = int(maxSnps)
sampleSizePerTimeStep=int(sampleSizePerTimeStep)

def seqDist(hap1, hap2):
    assert len(hap1) == len(hap2)
    numDiffs = 0
    for i in range(len(hap1)):
        if hap1[i] != hap2[i]:
            numDiffs += 1
    return numDiffs

def getMinimalEditDistanceHap(ali):
    distances = np.zeros((len(ali),len(ali)))
    for i in range(len(ali)-1):
        for j in range(i+1, len(ali)):
            d = seqDist(ali[i], ali[j])
            distances[i,j]=d
            distances[j,i]=d

    minDistSum = len(ali[0])*len(ali) + 1
    minDistHapIndex = -1
    for i in range(len(ali)):
        currDistSum = sum(distances[i])
        if currDistSum < minDistSum:
            minDistSum = currDistSum
            minDistHapIndex = i
    assert minDistHapIndex >= 0
    return ali[minDistHapIndex]

def sortAliBySimilarityToHap(ali, targHap):
    sortedAli = list(reversed(sorted(ali, key=lambda hap: seqDist(hap, targHap))))
    return sortedAli

def sortAliAllTimePoints(alis):
    newAlis = []
    targHap = getMinimalEditDistanceHap(alis[-1])
    for i in range(len(alis)):
        newAli = sortAliBySimilarityToHap(alis[i], targHap)
        newAlis.append(newAli)
    return newAlis

def sortAliOneTimePoint(ali):
    targHap = getMinimalEditDistanceHap(ali)
    newAli = sortAliBySimilarityToHap(ali, targHap)
    return newAli

# old way when we were padding for some reason
#def getAliForTimePoint(currSample, maxSnps):
#    numChroms = len(currSample)
#    a = np.array(currSample, dtype='int8')
#    b = np.full((numChroms, maxSnps), -2, dtype='int8')
#    b[:,:a.shape[1]] = a
#    return b

def getAliForTimePoint(currHaps):
    return currHaps

def getTimeSeriesAlis(currHaps, sampleSizePerTimeStep):
    sampleSize = len(currHaps)

    if sampleSize == sampleSizePerTimeStep:
        return sortAliOneTimePoint(getAliForTimePoint(currHaps))
    else:
        alis = []
        for i in range(0, sampleSize, sampleSizePerTimeStep):
            currAli = getAliForTimePoint(currHaps[i:i+sampleSizePerTimeStep])
            alis.append(currAli)
        return sortAliAllTimePoints(alis)

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
                    if len(hapMats) % 100 == 0:
                        sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                    hapMats.append(getTimeSeriesAlis(currHaps, sampleSizePerTimeStep))
                    readMode = 0
                else:
                    assert line[0] in ["0","1"]
                    currHaps.append(list(line[start:end]))
        hapMats.append(getTimeSeriesAlis(currHaps, sampleSizePerTimeStep))
        sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
    return hapMats

classNameToLabel = {'hard': 1, 'neut': 0, 'soft': 2}
def readAndSplitMsData(inDir, maxSnps, sampleSizePerTimeStep, testProp=0.1, valProp=0.1):
    aliX = []
    y = []
    for inFileName in os.listdir(inDir):
        sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split(".")[0]
        classLabel = classNameToLabel[className]
        currAlis = readMsData(inDir + "/" + inFileName, maxSnps, sampleSizePerTimeStep)

        y += [classLabel]*len(currAlis)
        aliX += currAlis

    X = np.array(aliX, dtype='int8')
    y = np.array(y, dtype='int8')

    trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
    testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

    return trainX, trainy, testX, testy, valX, valy

# assume that each input file is a separate class
trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(inDir, maxSnps, sampleSizePerTimeStep)
print(trainX.shape, testX.shape, valX.shape)
print(len(trainy), len(testy), len(valy))

np.savez_compressed(outFileName, trainX=trainX, trainy=trainy, testX=testX, testy=testy, valX=valX, valy=valy)
