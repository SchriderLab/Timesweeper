#!/usr/bin/env python
import os, sys, gzip, random

msFileDir, fixNum, shuffle = sys.argv[1:]
fixNum = [int(i) for i in fixNum.split(",")]

if not shuffle in ["shuffle", "no_shuffle"]:
    sys.exit("shuffle must be set to either 'shuffle' or 'no_shuffle'. AAAARRRRGGGGHHHHHHHHHH!!!!\n")

def readAllMSRepsFromFileNeut(msFileName):
    if msFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    msStream = fopen(msFileName, "rt")

    header = msStream.readline().strip().split()
    program,numSamples,numSims = header[:3]
    if len(header) > 3:
        otherParams = " " + " ".join(header[3:])
    else:
        otherParams = ""
    numSamples, numSims = int(numSamples),int(numSims)

    #advance to first simulation
    line = msStream.readline()
    while not line.strip().startswith("//"):
        line = msStream.readline()
    repLs = []
    while line:
        if not line.strip().startswith("//"):
            sys.exit("Malformed ms-style output file: read '%s' instead of '//'. AAAARRRRGGHHH!!!!!\n" %(line.strip()))
        repStr = ["\n//"]
        repStr.append(msStream.readline().strip()) #segsites line
        positionsLine = msStream.readline().strip()
        if not positionsLine.startswith("positions:"):
            sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
        repStr.append(positionsLine) #positions line

        for i in range(numSamples):
            currLine = msStream.readline()
            repStr.append(currLine.strip())
        line = msStream.readline()
        #advance to the next non-empty line or EOF
        while line and line.strip() == "":
            line = msStream.readline()
        repStr = "\n".join(repStr)
        repLs.append(repStr)
    msStream.close()

    return numSamples, repLs

def readAllMSRepsFromFile(msFileName, counter):
    if msFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    msStream = fopen(msFileName, "rt")

    header = msStream.readline().strip().split()
    program,numSamples,numSims = header[:3]
    if len(header) > 3:
        otherParams = " " + " ".join(header[3:])
    else:
        otherParams = ""
    numSamples, numSims = int(numSamples),int(numSims)

    #advance to first simulation
    line = msStream.readline()
    while not line.strip().startswith("//"):
        line = msStream.readline()
    repLs = []
    while line:
        counter+=1
        if not line.strip().startswith("//"):
            sys.exit("Malformed ms-style output file: read '%s' instead of '//'. AAAARRRRGGHHH!!!!!\n" %(line.strip()))
        repStr = ["\n//"]
        segsitesLine = msStream.readline().strip()
        if counter in fixNum:
            repStr.append(segsitesLine) #segsites line
        positionsLine = msStream.readline().strip()
        if not positionsLine.startswith("positions:"):
            sys.exit("Malformed ms-style output file. AAAARRRRGGHHH!!!!!\n")
        if counter in fixNum:
            repStr.append(positionsLine) #positions line

        for i in range(numSamples):
            currLine = msStream.readline()
            if counter in fixNum:
                repStr.append(currLine.strip())
        line = msStream.readline()
        #advance to the next non-empty line or EOF
        while line and line.strip() == "":
            line = msStream.readline()
        repStr = "\n".join(repStr)
        if counter in fixNum:
            repLs.append(repStr)
    msStream.close()

    return numSamples, repLs, counter 

repLs = []
allNumSamples = {}

counter = 0

for msFileName in os.listdir(msFileDir):
    sys.stderr.write("%s\n" %(msFileName))
    if "neut" in msFileName:
        currNumSamples, currRepLs = readAllMSRepsFromFileNeut(msFileDir + "/" + msFileName)
    else:
        currNumSamples, currRepLs, counter = readAllMSRepsFromFile(msFileDir + "/" + msFileName, counter)
    allNumSamples[currNumSamples] = 1
    repLs += currRepLs
assert len(allNumSamples) == 1
print("./msStyle %s %s\nblah\n" %(currNumSamples, len(repLs)))
if shuffle == "shuffle":
    random.shuffle(repLs)
print("\n".join(repLs))
