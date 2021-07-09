import os, sys
from glob import glob
from tqdm import tqdm
import numpy as np
from collections import Counter


class MsHandler:
    """Handles haplotype-tracked MS-style formatting from a standard SLiM output file.
    Runner function is parse_MS for easy tracking."""

    def __init__(self, mutfile, numSamples, tol, physLen, numReps):
        self.mutfile = mutfile
        self.numSamples = numSamples
        self.tol = tol
        self.physLen = physLen
        self.numReps = numReps

    def parse_MS(self):
        """
        Runs all necessary steps to parse SLiM output and format it for hfs creation.

        Returns:
            list[str]: List of "lines" of a typical ms-format output. Used to be output as an intermediate file, but is now just passed as a list of str for parsing.
        """
        out_ms = []
        # Yield as many samples as we have
        for idx, mutations in self.readSampleOutFromSlimRun():
            newMutLocs = self.get_mutLocs(mutations)
            allMuts = self.buildMutationPosMapping(newMutLocs)
            polyMuts = self.removeMonomorphic(allMuts)
            positionsStr = self.buildPositionsStr(polyMuts)
            segsitesStr = "segsites: {}".format(len(polyMuts))
            haps = self.make_haps(polyMuts)

            if idx == 0:
                out_ms += self.emitMsEntry(
                    positionsStr, segsitesStr, haps, isFirst=True
                )
            else:
                out_ms += self.emitMsEntry(
                    positionsStr, segsitesStr, haps, isFirst=False
                )

        return out_ms

    def readSampleOutFromSlimRun(self):
        """
        Generator that reads SLiM output file and collates mutation information, yielding a mutation set for a sample every time a breakpoint is located.

        Returns:
            list[ints]: List of locations where mutations occur on the chromosome.
        """
        with open(self.mutfile, "r") as infile:
            lines = [i.strip() for i in infile.readlines()]

        totSampleCount = 0
        for lineidx in range(len(lines)):
            if "#OUT" in lines[lineidx]:
                # print(lines[lineidx])
                totSampleCount += 1
        samplesToSkip = totSampleCount - self.numSamples
        # sys.stderr.write("found {} samples and need to skip {}\n".format(totSampleCount, samplesToSkip))

        mode = 0
        samplesSeen = 0
        mutations = {}
        self.genomes = []
        for line in lines:
            if mode == 0:
                if "#OUT" in line:
                    samplesSeen += 1
                    if samplesSeen >= samplesToSkip + 1:
                        sampleText = []
                        mode = 1
            elif mode == 1:
                if line.startswith("Done emitting sample"):
                    mode = 0
                    mutations = self.addMutationsAndGenomesFromSample(
                        sampleText, mutations
                    )
                    yield mutations
                else:
                    sampleText.append(line)

        # Don't believe a return is needed

    def addMutationsAndGenomesFromSample(self, sampleText, mutations):
        """
        Maps mutation IDs to chromosomes that contain them, resulting in a genotype string that is added to the genomes list.

        Args:
            sampleText (list[str]): Lines of SLiM output relating to one timepoint sample from a series.
            mutations (dict[int]): Dict of mutation locations binned by their ID in the chromosome being sampled.

        Returns:
            dict[int]: Mutation IDs along the chromosome binned by location as key. Retains input values and is added on with new samples.
        """
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
                    if not pos in mutations:
                        mutations[pos] = {}
                    mutations[pos][permId] = 1
                    idMapping[tempId] = permId
            elif mode == 2:
                line = line.strip().split()
                gId, auto = line[:2]
                mutLs = line[2:]
                self.genomes.append(set([idMapping[x] for x in mutLs]))

        return mutations

    def get_mutLocs(self, mutations):
        """
        Build new mutation map based on windows of mutations.

        Args:
            mutations (dict[int]): Dict of mutations binned by location.

        Returns:
            list[tuple(int, int)]: List of paired mutation (positions, IDs) in new format.
        """
        newMutLocs = []
        for mutPos in mutations:
            if len(mutations[mutPos]) == 1:
                mutId = list(mutations[mutPos].keys())[0]
                newMutLocs.append((mutPos, mutId))
            else:
                firstPos = mutPos - self.tol
                lastPos = mutPos + self.tol
                interval = (lastPos - firstPos) / (len(mutations[mutPos]) - 1)
                currPos = firstPos
                for mutId in mutations[mutPos]:
                    newMutLocs.append((currPos, mutId))
                    currPos += interval

        return newMutLocs

    def buildMutationPosMapping(self, mutLocs):
        """
        Creates new mapping relative to length of chromosome, adds to information tuple for mutation.

        Args:
            mutLocs list[tuple(int, int)]: List of paired mutation (positions, IDs) in new format.

        Returns:
            list[tuple(int, int, float, int)]: Tuples of (newID, abs position, continuous position, permID).
        """
        mutMapping = []
        mutLocs.sort()
        for i in range(len(mutLocs)):
            pos, mutId = mutLocs[i]
            contPos = pos / self.physLen
            mutMapping.append((i, pos, contPos, mutId))

        return mutMapping

    def removeMonomorphic(self, allMuts):
        """
        Removes singletons by selecting only mutations that are polymorphic.

        Args:
            allMuts (list[tuple(int, int, float, int)]): Tuples of (newID, abs position, continuous position, permID)

        Returns:
            list[tuple(int, int, float, int)]: Tuples of (newID, abs position, continuous position, permID) for polymorphic mutations only.
        """
        newMuts = []
        newLocI = 0
        for locI, loc, contLoc, mutId in allMuts:
            freq = self.getFreq((locI, loc, mutId))
            if freq > 0 and freq < len(self.genomes):
                newMuts.append((newLocI, loc, contLoc, mutId))
                newLocI += 1

        return newMuts

    def getFreq(self, mut):
        """
        Calculate frequency of a mutation in each genome.

        Args:
            mut (tuple): Mutation information to query against the genome list.

        Returns:
            int: Number of times input mutation appears in all genomes.
        """
        _, _, mutId = mut
        mutCount = 0
        for genome in self.genomes:
            if mutId in genome:
                mutCount += 1

        return mutCount

    def buildPositionsStr(self, muts):
        """
        Uses new mutation locations to build an MS-style mutation positions string.

        Args:
            muts (list[tuple]): Tuples of (newID, abs position, continuous position, permID) for each mutation.

        Returns:
            str: ms-style chromosome mutation positions string for use in downstream parsing.
        """
        positionsStr = []
        for _, locationDiscrete, _, mutId in muts:
            positionsStr.append(f"{locationDiscrete}.{mutId}")

        return "positions: " + " ".join(positionsStr)

    def make_haps(self, polyMuts):
        """
        Creates genotype 0/1 strings for each haplotype in ms-style format.

        Args:
            polyMuts (list[tuple]): Polymorphic mutations with ID, location, and permID fields.

        Returns:
            list[str]: All haplotype genotype strings for a given sample.
        """
        haps = []
        for i in range(len(self.genomes)):
            haps.append(["0"] * len(polyMuts))

        for i in range(len(self.genomes)):
            for locI, loc, contLoc, mutId in polyMuts:
                if mutId in self.genomes[i]:
                    haps[i][locI] = "1"

        return haps

    def emitMsEntry(self, positionsStr, segsitesStr, haps, isFirst=False):
        """
        Writes a list of strings that is equivalent to the lines in an ms-formatted output.
        Can be edited to output to file instead easily.

        Args:
            positionsStr (str): Str of all positions with segsites
            segsitesStr (str): Str of number of segsites total in MS entry
            haps (list[str]]): All haplotype genotype strings for a given sample.
            isFirst (bool, optional): Flag for whether ms headers are added to the entry. Defaults to False.

        Returns:
            [type]: [description]
        """

        ms = []
        if isFirst:
            ms.append("slim {} {}\n".format(len(haps), self.numReps))
            ms.append("foo")
        ms.append("//")
        ms.append(segsitesStr)
        ms.append(positionsStr)
        for line in haps:
            ms.append("".join(line) + "\n")

        return ms


class HapHandler:
    """
    Handles haplotype frequency spectrum generation, sorting, and output.
    Structure is very nested, user-facing function is readAndSplitMsData.
    """

    def __init__(self, hap_ms, sampleSizePerTimeStep, maxSnps):
        self.hap_ms = hap_ms
        self.sampleSizePerTimeStep = sampleSizePerTimeStep
        self.maxSnps = maxSnps

    def readAndSplitMsData(self, inFileName):
        """Runner function that allows for broad exception catching from nested functions."""
        try:
            currTimeSeriesHFS = self.readMsData()
            X = np.array(currTimeSeriesHFS, dtype="float32")
            return X, inFileName.split("/")[-1].split(".")[0]

        except Exception as e:
            print(
                "couldn't make {} because of: {}".format(
                    inFileName.split("/")[-1].split(".")[0]
                ),
                e,
            )
            return None, None

    def readMsData(self):
        """
        Iterates through haplotype-tracked MS entry and creates haplotype matrices.


        Returns:
            list[list[float]]: Haplotype frequency spectrums for all timepoints; sorted by most common freq at any sampling point in series.
        """
        readMode = 0
        hapMats = []
        for line in self.hap_ms:
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
                    # if len(hapMats) % 100 == 0:
                    # sys.stderr.write("read {} hap matrices\r".format(len(hapMats)))
                    hapMats.append(self.getTimeSeriesHapFreqs(currHaps))
                    readMode = 0
                else:
                    if line.strip().startswith("blarg"):
                        pass
                    if line[0] in ["0", "1"]:
                        currHaps.append(line[start:end])

        hapMats.append(self.getTimeSeriesHapFreqs(currHaps))

        return hapMats

    def getTimeSeriesHapFreqs(self, currHaps):
        """
        Build haplotype frequency spectrum for entire time-series of samples.

        Args:
            currHaps (list[str]): List of haplotypes read from MS entry.

        Returns:
            list[float]: Haplotype frequency spectrum sorted by highest frequency at any point in the sampling process.
        """
        winningAllHap = self.getMostCommonHapInEntireSeries(currHaps)
        hapBag = list(set(currHaps))
        hapBag.pop(hapBag.index(winningAllHap))

        hapToIndex = {}
        index = 0
        hapToIndex[winningAllHap] = index

        while len(hapBag) > 0:
            index += 1
            mostSimilarHapIndex = self.getMostSimilarHapIndex(hapBag, winningAllHap)
            mostSimilarHap = hapBag.pop(mostSimilarHapIndex)
            hapToIndex[mostSimilarHap] = index

        if self.sampleSizePerTimeStep == len(currHaps):
            hapFreqMat = self.getHapFreqsForTimePoint(
                currHaps, hapToIndex, len(currHaps)
            )
        else:
            hapFreqMat = []
            for i in range(0, len(currHaps), self.sampleSizePerTimeStep):
                currHapFreqs = self.getHapFreqsForTimePoint(
                    currHaps[i : i + self.sampleSizePerTimeStep],
                    hapToIndex,
                    len(currHaps),
                )
                hapFreqMat.append(currHapFreqs)

        return hapFreqMat

    def getMostCommonHapInEntireSeries(self, haps):
        """
        Iterates through haplotype counts in entire series to find the most common hap.

        Args:
            haps (list[str]): List of haplotypes, each is a 1D genotype string

        Returns:
            str: Most frequent haplotype in entire sampling process
        """
        # Calculate haplotype frequency for each haplotype
        allFreqs = []
        for i in range(0, len(haps), self.sampleSizePerTimeStep):
            freqsInSamp = {}
            for hap in haps[i : i + self.sampleSizePerTimeStep]:
                if not hap in freqsInSamp:
                    freqsInSamp[hap] = 0
                freqsInSamp[hap] += 1 / self.sampleSizePerTimeStep
            allFreqs.append(freqsInSamp)

        # Calculate haplotype freq change from the start of sampling at each timestep and sort
        allHaps = []
        for timeStep in range(1, len(allFreqs)):
            for hap in allFreqs[timeStep]:
                if hap in allFreqs[0]:
                    freqChange = allFreqs[timeStep][hap] - allFreqs[0][hap]
                else:
                    freqChange = allFreqs[timeStep][hap]
                allHaps.append((allFreqs[timeStep][hap], freqChange, timeStep, hap))

        allHaps.sort()
        winningHapFreq, winningHapFreqChange, winningHapTime, winningHap = allHaps[-1]

        return winningHap

    def getMostSimilarHapIndex(self, haps, targHap):
        """
        Calculate distances between a current haplotype and all given haps in sample.

        Args:
            haps (list[str]): Haplotypes for a given sample point.
            targHap (str): Haplotype to calculate distance from.

        Returns:
            int: Index of the haplotype in the hapbag that has the min distance from targHap.
        """
        minDist = float("inf")
        for i in range(len(haps)):
            dist = self.seqDist(haps[i], targHap)
            if dist < minDist:
                minDist = dist
                minIndex = i

        return minIndex

    def getHapFreqsForTimePoint(self, currSample, hapToIndex, maxPossibleHaps):
        """
        Create haplotype freq spectrum for a given sample and haplotype.

        Args:
            currSample (list[str]): Set of haplotypes in current time-sample.
            hapToIndex (int): Index of hap from hap-bag to calculate with.
            maxPossibleHaps (int): Number of total possible haplotypes.

        Returns:
            list[float]: Haplotype frequency spectrum for a given set of haplotypes.
        """
        hfs = [0] * maxPossibleHaps
        for hap in currSample:
            hfs[hapToIndex[hap]] += 1

        hfs = [x / len(currSample) for x in hfs]

        return hfs

    def getMostCommonHapInLastBunch(self, haps):
        """
        Deprecated in favor of finding most common hap in entire series.
        Iterates through haplotype counts at the final timepoint to find the most common hap.

        Args:
            haps (list[str]): List of haplotypes, each is a 1D genotype string

        Returns:
            str: Most frequent haplotype in the last sample
        """
        counts = Counter(haps[-self.sampleSizePerTimeStep :])
        if len(counts) == 1:
            return counts.most_common(1)[0][0]

        winner, runnerUp = counts.most_common(2)

        if winner[1] == runnerUp[1]:
            initCounts = Counter(haps[: self.sampleSizePerTimeStep])
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

    def seqDist(self, hap1, hap2):
        """
        Calculates pairwise distance between two haplotypes

        Args:
            hap1 (list): Haplotype 1
            hap2 (list): Haplotype 2

        Returns:
            int: Number of pairwise differences (sequence distance) between haps
        """
        assert len(hap1) == len(hap2)
        numDiffs = 0
        for i in range(len(hap1)):
            if hap1[i] != hap2[i]:
                numDiffs += 1

        return numDiffs


def main():
    physLen = 100000
    tol = 0.5
    numReps = 1
    mutdir = sys.argv[1]
    maxSnps = 50
    sampleSizePerTimeStep = int(sys.argv[2])
    outDir = os.path.join("/".join(mutdir.split("/")[:-2]), "haps")

    npz_list = []
    for mutfile in tqdm(glob(mutdir + "/muts/*/*.pop"), desc="Creating MS files..."):
        outFile = mutfile.split(".")[0] + ".msCombo"
        print(mutfile)

        if "1Samp" in mutfile:
            numSamples = 1
        else:
            numSamples = int(mutfile.split("-")[2].split("Samp")[0])

        # Handles MS parsing
        msh = MsHandler(mutfile, numSamples, tol, physLen, numReps)
        hap_ms = msh.parse_MS()

        # Convert MS into haplotype freq spectrum and format output
        hh = HapHandler(hap_ms, sampleSizePerTimeStep, maxSnps)
        X, id = hh.readAndSplitMsData(mutfile)

        if X != None and id != None:
            npz_list.append((X, id))
        else:
            continue

    np.savez(os.path.join(outDir, "haps.npz"), {i: j for (i, j) in npz_list})


if __name__ == "__main__":
    main()