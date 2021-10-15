import os, sys
from glob import glob
from tqdm import tqdm
import numpy as np
from collections import Counter
import argparse
from math import ceil
import random as rand
import multiprocessing as mp
from itertools import cycle


class MsHandler:
    """Handles haplotype-tracked MS-style formatting from a standard SLiM output file.
    Runner function is parse_slim for easy tracking."""

    def __init__(self, mutfile, numSamples, tol, physLen, samp_size, samp_gens):
        # TODO docs
        self.mutfile = mutfile
        self.numSamples = numSamples
        self.tol = tol
        self.physLen = physLen
        self.samp_size = samp_size
        self.samp_gens = samp_gens

    def parse_slim(self):
        """
        Runs all necessary steps to parse SLiM output and format it for hfs creation.

        Returns:
            list[str]: List of "lines" of a typical ms-format output. Used to be output as an intermediate file, but is now just passed as a list of str for parsing.
        """
        # Yield as many timepoints as we have, then filter by however many we want
        mutations, genomes = self.readSampleOutFromSlimRun()
        newMutLocs = self.get_mutLocs(mutations)
        unfilteredMuts = self.buildMutationPosMapping(newMutLocs)
        polyMuts = self.removeMonomorphic(unfilteredMuts, genomes)
        positionsStr = self.buildPositionsStr(polyMuts)

        # Iterate through timepoints, mutations is just a length indicator at this point
        segsitesStr = f"segsites: {len(polyMuts)}"
        haps = self.make_haps(polyMuts, genomes)

        out_ms = self.emitMsEntry(positionsStr, segsitesStr, haps)

        return out_ms

    def readSampleOutFromSlimRun(self):
        """
        Pseudo-generator that adds genomes and mutations to object-wide dicts every time a new one is encountered.
        Scans through each line of SLiM output, at the end of a sample it will collate genomes and mutation info and store in dicts.
        """
        with open(self.mutfile, "r") as infile:
            lines = [i.strip() for i in infile.readlines()]

        totSampleCount = 0
        for lineidx in range(len(lines)):
            if "#OUT" in lines[lineidx]:
                totSampleCount += 1
        samplesToSkip = totSampleCount - self.numSamples

        mode = 0
        samplesSeen = 0
        mutations = {}
        genomes = []
        for line in lines:
            if mode == 0:
                if "#OUT" in line:
                    samplesSeen += 1
                    if samplesSeen > samplesToSkip:
                        sampleText = []
                        mode = 1
            elif mode == 1:
                if "Done emitting sample" in line:
                    mode = 0
                    # Only sample the data that's in the gens we're sampling from
                    if ((samplesSeen - samplesToSkip) in self.samp_gens) and (
                        (samplesSeen > samplesToSkip)
                    ):
                        all_samp_genomes = self.addMutationsAndGenomesFromSample(
                            sampleText, mutations,
                        )
                        sampled = self.subsample_genomes(all_samp_genomes)
                        genomes.extend(sampled)

                else:
                    sampleText.append(line)

        # Last one
        # sampled = self.subsample_genomes(all_samp_genomes)
        # genomes.extend(sampled)

        return mutations, genomes

    def addMutationsAndGenomesFromSample(self, sampleText, mutations):
        """
        Maps mutation IDs to chromosomes that contain them, resulting in a genotype string that is added to the genomes list.

        Args:
            sampleText (list[str]): Lines of SLiM output relating to one timepoint sample from a series.
            mutations (dict[int]): Dict of mutation locations binned by their ID in the chromosome being sampled.
            genomes (list[set(int)]): List of genome IDs for each sample that have muts

        Mutations and Genomes are added in-scope.
        """
        mode = 0
        idMapping = {}
        gens = []
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
                gens.append(set([idMapping[x] for x in mutLs]))

        return gens

    def subsample_genomes(self, genomes):
        """
        Take random subset of genomes at a given timepoint without replacement.
        For each timepoint in samp_gens:
            Sample (sampsize) genomes from the set

        Returns:
            List[List[int]]: IDs of sampled genomes for each timepoint
        """
        return rand.sample(genomes, self.samp_size)

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

    def removeMonomorphic(self, allMuts, genomes):
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
            freq = self.getFreq((locI, loc, mutId), genomes)
            if freq > 0 and freq < len(genomes):
                newMuts.append((newLocI, loc, contLoc, mutId))
                newLocI += 1

        return newMuts

    def getFreq(self, mut, genomes):
        """
        Calculate frequency of a mutation in each genome.

        Args:
            mut (tuple): Mutation information to query against the genome list.

        Returns:
            int: Number of times input mutation appears in all genomes.
        """
        _, _, mutId = mut
        mutCount = 0
        for genome in genomes:
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

    def make_haps(self, polyMuts, sampled_genomes):
        """
        Creates genotype 0/1 strings for each haplotype in ms-style format.

        Args:
            polyMuts (list[tuple]): Polymorphic mutations with ID, location, and permID fields.

        Returns:
            list[str]: All haplotype genotype strings for a given sample.
        """
        haps = []

        for i in range(len(sampled_genomes)):
            haps.append(["0"] * len(polyMuts))

        for i in range(len(sampled_genomes)):
            for locI, loc, contLoc, mutId in polyMuts:
                if mutId in sampled_genomes[i]:
                    haps[i][locI] = "1"

        return haps

    def emitMsEntry(self, positionsStr, segsitesStr, haps):
        """
        Writes a list of strings that is equivalent to the lines in an ms-formatted output.
        Can be edited to output to file instead easily.

        Args:
            positionsStr (str): Str of all positions with segsites
            segsitesStr (str): Str of number of segsites total in MS entry
            haps (list[str]]): All haplotype genotype strings for a given sample.

        Returns:
            List[str]: Expected MS output format for entire time series of sampled points and haps.
        """

        ms = []
        ms.append(f"slim {len(haps)} 1")
        ms.append("foo")
        ms.append("//")
        ms.append(segsitesStr)
        ms.append(positionsStr)
        for line in haps:
            ms.append("".join(line))

        return ms


class HapHandler:
    """
    Handles haplotype frequency spectrum generation, sorting, and output.
    Structure is very nested, user-facing function is readAndSplitMsData.
    """

    def __init__(self, hap_ms, sampleSizePerTimeStep, total_haps, maxSnps):
        self.hap_ms = hap_ms
        self.sampleSizePerTimeStep = sampleSizePerTimeStep
        self.total_haps = total_haps
        self.maxSnps = maxSnps

    def readAndSplitMsData(self, inFileName):
        """Runner function that allows for broad exception catching from nested functions."""
        try:
            currTimeSeriesHFS = self.readMsData()
            X = np.array(currTimeSeriesHFS, dtype="float32")
            return X, "/".join([inFileName.split("/")[-3], inFileName.split("/")[-1]])

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
                elif "//" in line:
                    # Should only happen for multi-series MS files, which is not normal
                    hapMats.append(self.getTimeSeriesHapFreqs(currHaps))
                    readMode = 0
                else:
                    if line.strip().startswith("foo"):
                        pass
                    if line[0] in ["0", "1"]:
                        currHaps.append(line[start:end])

        # Will generally happen once, for a single MS entry that covers an entire time-series
        hapMats.append(self.getTimeSeriesHapFreqs(currHaps))

        return hapMats

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

    def getTimeSeriesHapFreqs(self, currHaps):
        """
        Build haplotype frequency spectrum for a single timepoint.

        Args:
            currHaps (list[str]): List of haplotypes read from MS entry.

        Returns:
            list[float]: Haplotype frequency spectrum for a single timepoint sorted by the most common hap in entire set.
        """
        winningHap = self.getMostCommonHapInEntireSeries(currHaps)
        hapBag = list(set(currHaps))
        hapBag.pop(hapBag.index(winningHap))

        hapToIndex = {}
        index = 0
        hapToIndex[winningHap] = index

        while len(hapBag) > 0:
            index += 1
            mostSimilarHapIndex = self.getMostSimilarHapIndex(hapBag, winningHap)
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
                    self.total_haps,  #! Change this to be total number of haplotypes
                )
                hapFreqMat.append(currHapFreqs)

        return hapFreqMat

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
        # raise DeprecationWarning
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


def samps_to_gens(num_timepoints, total_samps):
    """
    Takes in the number of samples wanted from user and creates list of gens to sample from.
    This gives greater flexibility between the num_timepoints arg and a custom list of gens to sample.
    This is evenly-spaced distribution of gens, if you want something special feed it into the custom gens arg.

    Args:
        num_timepoints (int): Number of timepoints to subsample from the total pool of timepoints given.
        total_samps (int): Total number of timepoints in the given simulation output.

    Returns:
        list[int]: List of generations (in terms of list index, not abs number) to pull samples from.
    """
    return [
        int(ceil(i * float(total_samps) / num_timepoints))
        for i in range(num_timepoints)
    ]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Haplotype frequency spectrum feature vector preparation.\
            Each run will result in an npz file named {samp_frequency}_{samp_size}.npz"
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        metavar="INPUT_DIRECTORY",
        help="Base mutation type (hard/soft/etc) directory containing subdirs with *.pop files to create feature vectors from.",
        dest="in_dir",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--schema-name",
        metavar="SCHEMA-NAME",
        help="Name to use for output files. This is optional if running one instance, but necessary when doing multiple Snakemake runs at once.",
        dest="schema_name",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-n",
        "--num_timepoints",
        metavar="NUM_TIMEPOINTS",
        help="How many timepoints to sample from a population over a sweep window. Cannot be larger than the number of samples taken in the entire pool of timepoints.",
        dest="num_timepoints",
        type=int,
        required=False,
    )

    parser.add_argument(
        "-g",
        "--gens",
        metavar="GENS_CUSTOM",
        help="List of (relative) generations to sample from out of the pool of timepoints output. Must use either this or --num_timepoints flag to define timepoints to sample. \
            Please note that first generation must be 1, if it is 0 it will be turned to 1 for indexing purposes.",
        dest="gens_custom",
        type=int,
        required=False,
        nargs="+",
    )

    parser.add_argument(
        "-s",
        "--sample-size",
        metavar="SAMPLE_SIZE",
        help="How many individuals to sample at each point.",
        dest="samp_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max_timepoints",
        metavar="SAMPLE-POOL-SIZE",
        help="Total number of samples taken from the simulations directly. Mostly useful if you need to skip the first SLiM output entries consistently. Set to the number of #OUT statements in a SLiM output for standard usage. Defaults to 41. ",
        dest="max_timepoints",
        type=int,
        required=False,
        default=41,
    )

    parser.add_argument(
        "--nthreads",
        metavar="NUM-PROCESSES",
        help="Number of threads available to multiprocessing module, more threads reduces runtime drastically. Defaults to all available - 1.",
        dest="nthreads",
        type=int,
        required=False,
        default=mp.cpu_count() - 1 or 1,
    )
    args = parser.parse_args()

    # 0 index causes issues downstream since we count the first sample point as 1
    if 0 in args.gens_custom:
        args.gens_custom[args.gens_custom.index(0)] = 1
        args.gens_custom = sorted(list(set(args.gens_custom)))

    if args.num_timepoints is None and args.gens_custom is None:
        print(
            "Must supply either consistent sampling freq or custom list of generations."
        )
        sys.exit(1)

    if args.gens_custom is not None:
        if min(args.gens_custom) < 0:
            print(
                "Cannot sample negative timepoints. Range must be 0-max timepoints value."
            )
            sys.exit(1)
        elif max(args.gens_custom) > args.max_timepoints:
            print(
                "Cannot sample timepoints that past the maximum number of timepoints. Range must be 0-max timepoints value."
            )
            sys.exit(1)
        else:
            pass

    return args


def worker(args):
    mutfile, max_timepoints, tol, physLen, samp_size, sample_points, maxSnps = args

    total_haps = sample_points * samp_size
    try:
        # Handles MS parsing
        msh = MsHandler(
            mutfile, max_timepoints, tol, physLen, samp_size, sample_points,
        )
        hap_ms = msh.parse_slim()

        # Convert MS into haplotype freq spectrum and format output
        hh = HapHandler(hap_ms, samp_size, total_haps, maxSnps)
        X, id = hh.readAndSplitMsData(mutfile)
        #! (TPs * sampsize)
        X = np.squeeze(X)
        # print(X.shape)
        # print(len(sample_points))
        # sys.stdout.flush()

        # Gotta be the right number of haps
        if X.shape[0] == len(sample_points):
            return (id, X)
        elif X is not None and id is not None:
            pass
        else:
            pass
    except:
        pass


def main():
    argp = parse_arguments()

    if argp.num_timepoints is not None:
        sample_points = samps_to_gens(argp.num_timepoints, argp.max_timepoints)
    else:
        sample_points = argp.gens_custom

    print("\n")
    print(f"Using {argp.nthreads} threads.")
    print("Sampling generations:", *sample_points, "\n")
    print("Data dir:", argp.in_dir)

    filelist = glob(argp.in_dir + "/pops/*/*.pop")
    sweep_lab = argp.in_dir.split("/")[-1]
    physLen = 100000
    tol = 0.5
    maxSnps = 50

    id_arrs = []

    args = zip(
        filelist,
        cycle([argp.max_timepoints]),
        cycle([tol]),
        cycle([physLen]),
        cycle([argp.samp_size]),
        cycle([sample_points]),
        cycle([maxSnps]),
    )

    chunksize = 4
    pool = mp.Pool(processes=argp.nthreads)
    for proc_result in tqdm(
        pool.imap_unordered(worker, args, chunksize=chunksize),
        desc="Submitting processes...",
        total=len(filelist),
    ):
        id_arrs.append(proc_result)

    ids = []
    arrs = []
    # Have to do sanity check
    for i in id_arrs:
        if i:
            ids.append(i[0])
            arrs.append(i[1])

    # print(arrs)

    print("Number of samples processed:", len(ids))
    print("Shape of single sample:", arrs[0].shape)

    ids = [f"{sweep_lab}/{i}" for i in ids]
    np.savez(
        os.path.join(argp.in_dir, f"hfs_{argp.schema_name}.npz"),
        **dict(zip(ids, arrs)),
    )
    print(
        "HFS data saved to:", os.path.join(argp.in_dir, f"hfs_{argp.schema_name}.npz"),
    )


if __name__ == "__main__":
    main()
