import sys
from glob import glob


def addMutationsAndGenomesFromSample(sampleText, locs, genomes):
    mode = 0
    idMapping = {}
    for line in sampleText:
        if mode == 0:
            if "Mutations" in line:
                mode = 1
        elif mode == 1:
            if "Genomes" in line:
                mode = 2
            elif len(line.strip().split()) == 9:
                (
                    tempId,
                    permId,
                    mutType,
                    pos,
                    selCoeff,
                    domCoeff,
                    subpop,
                    gen,
                    numCopies,
                ) = line.strip().split()
                pos = int(pos)
                if not pos in locs:
                    locs[pos] = {}
                locs[pos][permId] = 1
                idMapping[tempId] = permId
        elif mode == 2:
            line = line.strip().split()
            gId, auto = line[:2]
            mutLs = line[2:]
            genomes.append(set([idMapping[x] for x in mutLs]))


def readSampleOutFromSlimRun(output, numSamples):
    with open(output, "r") as infile:
        lines = [i.strip() for i in infile.readlines()]

    totSampleCount = 0
    for lineidx in range(len(lines)):
        if "#OUT" in lines[lineidx]:
            totSampleCount += 1
    samplesToSkip = totSampleCount - numSamples

    mode = 0
    samplesSeen = 0
    locs = {}
    genomes = []
    for lineidx in range(len(lines)):
        if mode == 0:
            if "#OUT" in lines[lineidx]:
                sys.stderr.write(lines[lineidx] + "\n")
                samplesSeen += 1
                if samplesSeen >= samplesToSkip + 1:
                    sampleText = []
                    mode = 1
        elif mode == 1:
            if "#OUT" in lines[lineidx]:
                mode = 0
                addMutationsAndGenomesFromSample(sampleText, locs, genomes)
            else:
                sampleText.append(lines[lineidx])

    return locs, genomes


def buildMutationPosMapping(mutLocs, physLen):
    mutMapping = []
    mutLocs.sort()
    for i in range(len(mutLocs)):
        pos, mutId = mutLocs[i]
        contPos = pos / physLen
        mutMapping.append((i, pos, contPos, mutId))
    return mutMapping


def getFreq(mut, genomes):
    _, _, mutId = mut
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
    for _, locationDiscrete, _, mutId in muts:
        positionsStr.append(f"{locationDiscrete}.{mutId}")
    return "positions: " + " ".join(positionsStr)


def emitMsEntry(outFile, positionsStr, segsitesStr, haps, numReps, isFirst=True):
    with open(outFile, "w") as wtfile:
        if isFirst:
            wtfile.write("slim {} {}\n".format(len(haps), numReps))
            wtfile.write("blarg\n")
        wtfile.write("//\n")
        wtfile.write(segsitesStr + "\n")
        wtfile.write(positionsStr + "\n")
        for line in haps:
            wtfile.write("".join(line) + "\n")


def main():
    physLen = 100000
    tol = 0.5
    numSamples = 20
    numReps = 100
    mutdir = sys.argv[1]
    for mutfile in glob(mutdir + "/*.muts"):
        outFile = mutfile.split(".")[0] + ".msCombo"
        print(outFile)
        mutations, genomes = readSampleOutFromSlimRun(mutfile, numSamples)
        newMutLocs = []
        for mutPos in mutations:
            if len(mutations[mutPos]) == 1:
                mutId = list(mutations[mutPos].keys())[0]
                newMutLocs.append((mutPos, mutId))
            else:
                firstPos = mutPos - tol
                lastPos = mutPos + tol
                interval = (lastPos - firstPos) / (len(mutations[mutPos]) - 1)
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
            haps.append(["0"] * len(polyMuts))

        for i in range(len(genomes)):
            for locI, loc, contLoc, mutId in polyMuts:
                if mutId in genomes[i]:
                    haps[i][locI] = "1"

        emitMsEntry(outFile, positionsStr, segsitesStr, haps, numReps, isFirst=True)


if __name__ == "__main__":
    main()