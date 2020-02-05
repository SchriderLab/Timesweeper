import numpy as np
import sys, os, gzip, random
import sklearn.model_selection

inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, outFileName = sys.argv[1:]
maxSnps = int(maxSnps)
sampleSize1PerTimeStep=int(sampleSize1PerTimeStep)
sampleSize2PerTimeStep=int(sampleSize2PerTimeStep)

def calcJointSFSForTimePoint(currSample, sampleSize1PerTimeStep, sampleSize2PerTimeStep):
    sfs = np.zeros((sampleSize1PerTimeStep+1, sampleSize2PerTimeStep+1))
    for site in range(len(currSample[0])):
        daCount1 = sum([int(currSample[x][site]) for x in range(0, sampleSize1PerTimeStep)])
        daCount2 = sum([int(currSample[x][site]) for x in range(sampleSize1PerTimeStep, len(currSample))])
        jointFreq = (daCount1, daCount2)
        if jointFreq != (0, 0) and jointFreq != (sampleSize1PerTimeStep, sampleSize2PerTimeStep):
            sfs[jointFreq] += 1
    denom=np.sum(sfs)
    if denom > 0:
        sfs = sfs/denom
    return sfs

def getTimeSeriesJointSFS(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep):
    sampleSize = len(currHaps)

    if sampleSize == sampleSize1PerTimeStep + sampleSize2PerTimeStep:
        sfsMat = calcJointSFSForTimePoint(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)
    else:
        sfsMat = []
        for i in range(0, sampleSize, sampleSize1PerTimeStep + sampleSize2PerTimeStep):
            currJointSFS = calcJointSFSForTimePoint(currHaps[i:i+sampleSize1PerTimeStep+sampleSize2PerTimeStep], sampleSize1PerTimeStep, sampleSize2PerTimeStep)
            sfsMat.append(currJointSFS)
    return sfsMat

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
                    hapMats.append(getTimeSeriesJointSFS(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
                    readMode = 0
                else:
                    assert line[0] in ["0","1"]
                    currHaps.append(list(line[start:end]))
        hapMats.append(getTimeSeriesJointSFS(currHaps, sampleSize1PerTimeStep, sampleSize2PerTimeStep))
        sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
    return hapMats

classNameToLabel = {'soft': 2, 'hard': 1, 'neut': 0}
def readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep, testProp=0.1, valProp=0.1):
    sfsX = []
    y = []
    for inFileName in os.listdir(inDir):
        sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split(".")[0]
        classLabel = classNameToLabel[className]
        currTimeSeriesSFS = readMsData(inDir + "/" + inFileName, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)

        y += [classLabel]*len(currTimeSeriesSFS)
        sfsX += currTimeSeriesSFS

    X = np.array(sfsX, dtype='float32')
    if len(X.shape) == 4:
        X = X.transpose(0, 2, 3, 1)
    y = np.array(y, dtype='int8')

    trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
    testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

    return trainX, trainy, testX, testy, valX, valy

# assume that each input file is a separate class
trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(inDir, maxSnps, sampleSize1PerTimeStep, sampleSize2PerTimeStep)
print(trainX.shape, testX.shape, valX.shape)
print(len(trainy), len(testy), len(valy))

np.savez_compressed(outFileName, trainX=trainX, trainy=trainy, testX=testX, testy=testy, valX=valX, valy=valy)
