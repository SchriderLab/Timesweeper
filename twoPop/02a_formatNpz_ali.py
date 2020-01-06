import numpy as np
import sys, os, gzip, random
import sklearn.model_selection

inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, outFileName = sys.argv[1:]
maxSnps = int(maxSnps)
sampleSize1PerTimeStep=int(sampleSize1PerTimeStep)
sampleSize2PerTimeStep=int(sampleSize2PerTimeStep)

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

def sortAliAllTimePoints(alisTwoPop):
    newAlis = []

    alis1 = [aliTwoPop[0] for aliTwoPop in alisTwoPop]
    alis2 = [aliTwoPop[1] for aliTwoPop in alisTwoPop]
    targHap1 = getMinimalEditDistanceHap(alis1[-1])
    targHap2 = getMinimalEditDistanceHap(alis2[-1])

    for i in range(len(alis1)):
        newAli1 = sortAliBySimilarityToHap(alis1[i], targHap1)
        newAli2 = sortAliBySimilarityToHap(alis2[i], targHap2)
        newAlis.append(newAli1 + newAli2)

    alisForPop.append(newAlis)
    return newAlis

def sortAliOneTimePointTwoPop(aliTwoPop):
    ali1, ali2 = aliTwoPop

    targHap1 = getMinimalEditDistanceHap(ali1)
    newAli1 = sortAliBySimilarityToHap(ali1, targHap1)
    
    targHap2 = getMinimalEditDistanceHap(ali2)
    newAli2 = sortAliBySimilarityToHap(ali2, targHap2)

    return newAli1+newAli2

#def getAliForTimePoint(currSample, maxSnps):
#    numPops, numChroms = len(currSample), len(currSample[0])
#    a = np.array(currSample, dtype='int8')
#    b = np.full((numPops, numChroms, maxSnps), -2, dtype='int8')
#    b[:,:,:a.shape[2]] = a
#    return b

def getAliForTimePointTwoPop(currHapsPop1, currHapsPop2):
    return currHapsPop1, currHapsPop2

def getTimeSeriesAlis(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep):
    sampleSize = len(currHaps)

    if sampleSize == sampleSize1PerTimeStep + sampleSize2PerTimeStep:
        return sortAliOneTimePointTwoPop(getAliForTimePointTwoPop([currHaps[:sampleSize1PerTimeStep], currHaps[sampleSize1PerTimeStep:sampleSize1PerTimeStep+sampleSize2PerTimeStep]]))
    else:
        alis = []
        for i in range(0, sampleSize, sampleSize1PerTimeStep+sampleSize2PerTimeStep):
            currAli = getAliForTimePointTwoPop([currHaps[i:i+sampleSize1PerTimeStep], currHaps[i+sampleSize1PerTimeStep:i+sampleSize1PerTimeStep+sampleSize2PerTimeStep]])
            alis.append(currAli)
        return sortAliOneTimePointTwoPop(alis)

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
                    hapMats.append(getTimeSeriesAlis(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
                    readMode = 0
                else:
                    assert line[0] in ["0","1"]
                    currHaps.append(list(line[start:end]))
        hapMats.append(getTimeSeriesAlis(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
        sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
    return hapMats

classNameToLabel = {'hard': 1, 'neut': 0, 'soft': 2}
def readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, testProp=0.1, valProp=0.1):
    aliX = []
    y = []
    for inFileName in os.listdir(inDir):
        sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split(".")[0]
        classLabel = classNameToLabel[className]
        currAlis = readMsData(inDir + "/" + inFileName, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)

        y += [classLabel]*len(currAlis)
        aliX += currAlis

    X = np.array(aliX, dtype='int8')
    y = np.array(y, dtype='int8')

    trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
    testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

    return trainX, trainy, testX, testy, valX, valy

# assume that each input file is a separate class
trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)
print(trainX.shape, testX.shape, valX.shape)
print(len(trainy), len(testy), len(valy))

np.savez_compressed(outFileName, trainX=trainX, trainy=trainy, testX=testX, testy=testy, valX=valX, valy=valy)
