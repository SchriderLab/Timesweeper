import sys, os, random, subprocess

#TODO: after making the required changes in here, check all downstream scripts for potential problems

scriptName, sampleSizePerStepTS, numSamplesTS, samplingIntervalTS, sampleSizePerStep1Samp, numSamples1Samp, samplingInterval1Samp, numReps, physLen, timeSeries, sweep, dumpFileName = sys.argv[1:]
if timeSeries.lower() in ["false", "none"]:
    timeSeries = False
else:
    timeSeries = True
if not sweep in ["hard", "soft", "neut"]:
    sys.exit("'sweep' argument must be 'hard', 'soft', or 'neut'")
sampleSizePerStepTS = int(sampleSizePerStepTS)
numSamplesTS = int(numSamplesTS)
samplingIntervalTS = int(samplingIntervalTS)
sampleSizePerStep1Samp = int(sampleSizePerStep1Samp)
numSamples1Samp = int(numSamples1Samp)
samplingIntervalTS = int(samplingIntervalTS)
numReps = int(numReps)
physLen = int(physLen)

def addMutationsAndGenomesFromSample(sampleText, locs, genomes):
    mode = 0
    idMapping = {}
    for line in sampleText:
        if mode == 0:
            if line.startswith("Mutations"):
                mode = 1
        elif mode == 1:
            if line.startswith("Genomes"):
                mode = 2
            else:
                tempId, permId, mutType, pos, selCoeff, domCoeff, subpop, gen, numCopies = line.strip().split()
                pos = int(pos)
                if not pos in locs:
                    locs[pos] = {}
                locs[pos][permId] = 1
                idMapping[tempId]=permId
        elif mode == 2:
            line = line.strip().split()
            gId, auto = line[:2]
            mutLs = line[2:]
            genomes.append(set([idMapping[x] for x in mutLs]))

def readSampleOutFromSlimRun(output, numSamples):
    totSampleCount = 0
    for line in output.decode("utf-8").split("\n"):
        if line.startswith("Sampling at generation"):
            totSampleCount += 1
    samplesToSkip = totSampleCount-numSamples
    #sys.stderr.write("found {} samples and need to skip {}\n".format(totSampleCount, samplesToSkip))

    mode = 0
    samplesSeen = 0
    locs = {}
    genomes = []
    for line in output.decode("utf-8").split("\n"):
        if mode == 0:
            #sys.stderr.write(line+"\n")
            if line.startswith("Sampling at generation"):
                samplesSeen += 1
                if samplesSeen >= samplesToSkip+1:
                    sampleText = []
                    mode = 1
        elif mode == 1:
            if line.startswith("Done emitting sample"):
                mode = 0
                addMutationsAndGenomesFromSample(sampleText, locs, genomes)
                #sys.stderr.write(line+"\n")
            else:
                sampleText.append(line)
        if "SEGREGATING" in line:
            sys.stderr.write(line+"\n")
    return locs, genomes

def buildMutationPosMapping(mutLocs, physLen):
    mutMapping = []
    mutLocs.sort()
    for i in range(len(mutLocs)):
        pos, mutId = mutLocs[i]
        contPos = pos/physLen
        mutMapping.append((i, pos, contPos, mutId))
    return mutMapping

def getFreq(mut, genomes):
    locI, loc, mutId = mut
    mutCount = 0
    for genome in genomes:
        if mutId in genome:
            mutCount += 1
    return mutCount

def removeMonomorphic(allMuts, genomes):
    newMuts = []
    newLocI = 0
    for locI, loc, contLoc, mutId in allMuts:
        freq = getFreq((locI, loc, mutId), genomes)
        if freq > 0 and freq < len(genomes):
            newMuts.append((newLocI, loc, contLoc, mutId))
            newLocI += 1
    return newMuts

def buildPositionsStr(muts):
    positionsStr = []
    for locationIndex, locationDiscrete, locationContinuous, mutId in muts:
        #positionsStr.append(f"{locationContinuous}")
        positionsStr.append(f"{locationDiscrete}.{mutId}")
    return "positions: " + " ".join(positionsStr)

def emitMsEntry(positionsStr, segsitesStr, haps, numReps, isFirst=True):
    if isFirst:
        print("slim {} {}".format(len(haps), numReps))
        print("blarg")
    print("\n//")
    print(segsitesStr)
    print(positionsStr)
    for line in haps:
        print("".join(line))

tol=0.5
for repIndex in range(numReps):
    sys.stderr.write("starting rep {}\n".format(repIndex))
    seed = random.randint(0, 2**32-1)
    if scriptName in ["sweep.slim", "sweep_twoPop.slim", "adaptiveIntrogressionTS.slim", "adaptiveIntrogressionTS_twoPop.slim"]:
        if timeSeries:
            numSamples=numSamplesTS
            if "twoPop" in scriptName:
                sampleSizeStr = "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(sampleSizePerStepTS, sampleSizePerStepTS)
            else:
                sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStepTS)
            slimCmd = "slim -seed {} {} -d samplingInterval={} -d numSamples={} -d sweep='{}' -d dumpFileName='{}' {}".format(seed, sampleSizeStr, samplingIntervalTS, numSamples, sweep, dumpFileName, scriptName)
        else:
            numSamples=numSamples1Samp
            if "twoPop" in scriptName:
                sampleSizeStr = "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(sampleSizePerStep1Samp, sampleSizePerStep1Samp)
            else:
                sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStep1Samp)
            slimCmd = "slim -seed {} {} -d samplingInterval={} -d numSamples={} -d sweep='{}' -d dumpFileName='{}' {}".format(seed, sampleSizeStr, samplingInterval1Samp ,numSamples, sweep, dumpFileName, scriptName)
    else:
        sys.exit("Unsupported slim script! ARRRGGHHHH!!!!!")

    procOut = subprocess.Popen(slimCmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output, err  = procOut.communicate()
    #print(output.decode("utf-8"))
    os.system("rm {}".format(dumpFileName))

    mutations, genomes = readSampleOutFromSlimRun(output, numSamples)
    newMutLocs = []
    for mutPos in mutations:
        if len(mutations[mutPos]) == 1:
            mutId = list(mutations[mutPos].keys())[0]
            newMutLocs.append((mutPos, mutId))
        else:
            firstPos = mutPos-tol
            lastPos = mutPos+tol
            interval = (lastPos-firstPos)/(len(mutations[mutPos])-1)
            currPos = firstPos
            for mutId in mutations[mutPos]:
                newMutLocs.append((currPos, mutId))
                currPos += interval

    allMuts = buildMutationPosMapping(newMutLocs, physLen)
    polyMuts = removeMonomorphic(allMuts, genomes)
    positionsStr = buildPositionsStr(polyMuts)
    segsitesStr = "segsites: {}".format(len(polyMuts))
    haps = []
    for i in range(len(genomes)):
        haps.append(["0"]*len(polyMuts))

    for i in range(len(genomes)):
        for locI, loc, contLoc, mutId in polyMuts:
            if mutId in genomes[i]:
                haps[i][locI] = "1"
    if repIndex == 0:
        emitMsEntry(positionsStr, segsitesStr, haps, numReps, isFirst=True)
    else:
        emitMsEntry(positionsStr, segsitesStr, haps, numReps, isFirst=False)
