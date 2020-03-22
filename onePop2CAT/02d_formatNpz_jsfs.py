import numpy as np
import sys, os, gzip, random
import sklearn.model_selection

inDir, maxSnps, sampleSizePerTimeStep, outFileName = sys.argv[1:]
maxSnps = int(maxSnps)
sampleSizePerTimeStep=int(sampleSizePerTimeStep)

def countFreqOfSegsite(currSample, segsite):
    return sum([int(x[segsite]) for x in currSample])

def getTimeSeriesJSFS(currHaps, sampleSizePerTimeStep):
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
                freq = countFreqOfSegsite(currHaps[i:i+sampleSizePerTimeStep], j)
                freqs.append(freq)
        tsJSFS[tuple(freqs)] += 1
    return tsJSFS

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
                    hapMats.append(getTimeSeriesJSFS(currHaps, sampleSizePerTimeStep))
                    readMode = 0
                else:
                    assert line[0] in ["0","1"]
                    currHaps.append(list(line[start:end]))
        hapMats.append(getTimeSeriesJSFS(currHaps, sampleSizePerTimeStep))
        sys.stderr.write("read {} hap matrices. done!\n".format(len(hapMats)))
    return hapMats

classNameToLabel = {'hard': 1, 'neut': 0, 'soft': 1}
def readAndSplitMsData(inDir, maxSnps, sampleSizePerTimeStep, testProp=0.1, valProp=0.1):
    sfsX = []
    y = []
    for inFileName in os.listdir(inDir):
        sys.stderr.write("reading {}\n".format(inFileName))
        className = inFileName.split(".")[0]
        classLabel = classNameToLabel[className]
        currTimeSeriesSFS = readMsData(inDir + "/" + inFileName, maxSnps, sampleSizePerTimeStep)

        y += [classLabel]*len(currTimeSeriesSFS)
        sfsX += currTimeSeriesSFS

    X = np.array(sfsX, dtype='float32')
    if len(X.shape) == 3:
        X = X.transpose(0, 2, 1)
    y = np.array(y, dtype='int8')

    trainX, tvX, trainy, tvy = sklearn.model_selection.train_test_split(X, y, stratify=y, test_size=testProp)
    testX, valX, testy, valy = sklearn.model_selection.train_test_split(tvX, tvy, stratify=tvy, test_size=valProp/(valProp+testProp))

    return trainX, trainy, testX, testy, valX, valy

# assume that each input file is a separate class
trainX, trainy, testX, testy, valX, valy = readAndSplitMsData(inDir, maxSnps, sampleSizePerTimeStep)
print(trainX.shape, testX.shape, valX.shape)
print(len(trainy), len(testy), len(valy))

np.savez_compressed(outFileName, trainX=trainX, trainy=trainy, testX=testX, testy=testy, valX=valX, valy=valy)
