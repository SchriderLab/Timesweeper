import sys
import numpy as np
import argparse



def getRepAndGen(headerEntry):
    genInfo, repInfo = headerEntry.split("_")[-2:]
    rep = int(repInfo)
    if genInfo == "Base":
        gen = 0
    else:
        gen = int(genInfo.lstrip("F"))
    return rep, gen


def getFreqs(baseCounts):
    aCount, tCount, cCount, gCount, nCount, delCount = [
        int(x) for x in baseCounts.split(":")
    ]
    counts = [aCount, tCount, cCount, gCount, delCount]
    denom = sum(counts)

    return [x / denom for x in counts]


def getWinningAlleleIndex(allFreqsForSnp, mode="final"):
    assert allFreqsForSnp.shape == (7, 5)  # 7 timepoints X freqs for 5 possible alleles

    freqScores = []
    for alleleIndex in range(allFreqsForSnp.shape[1]):
        if mode == "final":
            freqScore = allFreqsForSnp[-1][alleleIndex]
        elif mode == "velocity":
            freqScore = allFreqsForSnp[-1][alleleIndex] - allFreqsForSnp[0][alleleIndex]
        freqScores.append(freqScore)

    return np.argmax(freqScores)


def getMostCommonOtherAlleleIndex(allFreqsForSnp, winningAlleleIndex):
    assert allFreqsForSnp.shape == (7, 5)  # 7 timepoints X freqs for 5 possible alleles

    totalFreqs = []
    for alleleIndex in range(allFreqsForSnp.shape[1]):
        if alleleIndex == winningAlleleIndex:
            totalFreqs.append(-1)
        else:
            totalFreq = sum(allFreqsForSnp[:, alleleIndex])
            totalFreqs.append(totalFreq)

    return np.argmax(totalFreqs)


def encodeFreqsInPlace(freqArray):
    for snpIndex in range(len(freqArray[0])):
        allFreqsForSnp = []

        for gen in range(len(freqArray)):
            freqs = freqArray[gen][snpIndex]
            allFreqsForSnp.append(freqs)

        allFreqsForSnp = np.array(allFreqsForSnp)
        winningAlleleIndex = getWinningAlleleIndex(allFreqsForSnp, mode="velocity")
        otherAlleleIndex = getMostCommonOtherAlleleIndex(
            allFreqsForSnp, winningAlleleIndex
        )

        for gen in range(len(freqArray)):
            freqs = freqArray[gen][snpIndex]
            winningFreq = freqs[winningAlleleIndex]
            otherFreq = freqs[otherAlleleIndex]

            freq = winningFreq / (winningFreq + otherFreq)
            freqArray[gen][snpIndex] = freq
        # assert winningFreq > 0 and winningFreq >= otherFreq

def get_header(headerfile):
    with open(headerfile, "r") as hfile:
        header = hfile.readline().strip().split()
        
    return header

def main(ua):
    agp = argparse.ArgumentParser()
    agp.add_argument("-i", "--infile", help="Sync file to convert to timesweeper NPZ format.")
    agp.add_argument("-c", "--chrom", help="Chromosome to pull out of sync file.")
    agp.add_argument("-o", "--outdir", help="Output directory.")
    agp.add_argument("-w", "--window_size", help="Size of windows to extract.")
    agp.add_argument("-h", "--headerfile", help="File containing a single line - the header of the syncfile. Must be provided if header is not present in the syncfile.")
    ua = agp.parse_args()
    
    inFileName = ua.infile
    outDir = ua.outdir 
    winSize = ua.window_size 
    targetChrom = ua.chrom 
    
    if ua.headerfile:
       header = get_header(ua.headerfile)
    else:
        header = get_header(inFileName) 
    
    freqs = {}
    for i in range(len(header)):
        if "Dsim" in header[i]:
            rep, gen = getRepAndGen(header[i])
            if not rep in freqs:
                freqs[rep] = {}
            if not gen in freqs[rep]:
                freqs[rep][gen] = {}


    sys.stderr.write("reading snps and freqs\n")
    positions = {}
    with open(inFileName, "rt") as inFile:
        for line in inFile:
            line = line.strip().split()
            chrom = line[0]
            pos = int(line[1])
            if chrom == targetChrom:
                if not chrom in positions:
                    positions[chrom] = []
                    for rep in freqs:
                        for gen in freqs[rep]:
                            freqs[rep][gen][chrom] = []

                for i in range(len(line)):
                    if ":" in line[i]:
                        rep, gen = getRepAndGen(header[i])
                        currFreqs = getFreqs(line[i])
                        freqs[rep][gen][chrom].append(currFreqs)
                positions[chrom].append(pos)
    sys.stderr.write("got all snps and freqs\n")


    sys.stderr.write("formatting output\n")
    for rep in freqs:
        for chrom in positions:
            outFileName = f"{outDir}/dsim_chrom_{chrom}_rep_{rep}.npz"

            assert len(positions[chrom]) == len(freqs[rep][gen][chrom])

            freqArray = []
            for gen in sorted(freqs[rep]):
                freqArray.append([])
                for posIndex in range(len(positions[chrom])):
                    currFreqs = freqs[rep][gen][chrom][posIndex]
                    freqArray[-1].append(currFreqs)

            encodeFreqsInPlace(freqArray)
            freqArray = np.array(freqArray)
            numGens = len(freqs[rep])
            assert freqArray.shape == (numGens, len(positions[chrom]))

            allFreqWins = []
            allPosWins = []
            startingIndices = range(len(positions[chrom]) - winSize)
            for i in startingIndices:
                currWin = freqArray[:, i : i + winSize]
                allFreqWins.append(currWin)
                currPositions = positions[chrom][i : i + winSize]
                allPosWins.append(currPositions)
            assert i + winSize == len(positions[chrom]) - 1

            allFreqWins = np.array(allFreqWins)
            allPosWins = np.array(allPosWins)
            assert allFreqWins.shape == (len(startingIndices), numGens, winSize)
            assert allPosWins.shape == (len(startingIndices), winSize)

            np.savez(outFileName, aftIn=allFreqWins, aftInPosition=allPosWins)
            
    sys.stderr.write("all done!\n")

    
if __name__=="__main__":
    main()