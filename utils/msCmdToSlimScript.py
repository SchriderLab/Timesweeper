import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-u', dest='u', help="mutation rate (used both for SLiM and for parsing out Ne_0 and r)", type=float, required=True, metavar='u')
parser.add_argument('-L', dest='L', help="size of the simulated chromosome in bp (used both for SLiM and for parsing out Ne_0 and r)", type=int, required=True)
parser.add_argument('-ms', dest='msCmd', help="quoted ms command to be converted into a SLiM script", required=True, metavar='msCmd')
parser.add_argument('-b', dest='burnTime', help="number of burn-in generations to add before simulating demographic model", type=int, required=True, metavar='burnTime')
parser.add_argument('-o', dest='outPath', help="path to file where the slim output should be written", required=True, metavar='outPath')
args = parser.parse_args()

simChrSize = args.L
burnTime = args.burnTime
msCmd = args.msCmd
mutRate = args.u
outFileName = args.outPath

msParser = argparse.ArgumentParser()
msParser.add_argument('ms')
msParser.add_argument('sampleSize', type=int)
msParser.add_argument('numReps', type=int)
msParser.add_argument('-en', action='append', dest='popSizeChanges', nargs='+')
msParser.add_argument('-eN', action='append', dest='popSizeChanges', nargs='+')
msParser.add_argument('-t', dest='theta', type=float)
msParser.add_argument('-r', dest='rho', nargs=2)
msArgs, unknown = msParser.parse_known_args(msCmd.split())

if unknown:
    sys.exit(f'Error: unrecognized args in ms command ({unknown}); currently only handling single-population dynamics (\'-en\' and \'-eN\')')

if msArgs.ms != 'ms':
    sys.exit("Error: malformed ms command (doesn't start with 'ms')")
sampleSize = msArgs.sampleSize

theta = msArgs.theta
rho, numRecSites = msArgs.rho
rho = float(rho)
numRecSites = int(numRecSites)
popSizeChanges = msArgs.popSizeChanges

effectivePopSize = theta/(4*mutRate)
r = rho / (4*effectivePopSize)

sizeChangesDiscrete = []
if popSizeChanges:
    nextSize = int(round(effectivePopSize))
    popSizeChanges.sort(key=lambda x: float(x[0]))

    for sizeChange in popSizeChanges:
        if len(sizeChange) == 3:
            time, popId, sizeRatio = float(sizeChange[0]), int(sizeChange[1]), float(sizeChange[2])
        else:
            time, sizeRatio = float(sizeChange[0]), float(sizeChange[1])
        numGen = int(round(time*4*effectivePopSize))
        prevSize = int(round(effectivePopSize * sizeRatio))
        #print(f"size changes from {prevSize} to {nextSize} individuals {numGen} generations before the simulation end")
        sizeChangesDiscrete.append([numGen, prevSize, nextSize])
        nextSize = prevSize

    oldestTime = sizeChangesDiscrete[-1][0]
    initialSize = sizeChangesDiscrete[-1][1]
    simDuration = oldestTime + burnTime
else:
    simDuration = burnTime
    initialSize = int(round(effectivePopSize))

sizeChangesDiscrete.reverse()
for i in range(len(sizeChangesDiscrete)):
    sizeChangesDiscrete[i][0] = simDuration - sizeChangesDiscrete[i][0]

demogStr = ""
rescheduleStr = ""

sizeChangeNum = 1
for numGen, prevSize, nextSize in sizeChangesDiscrete:
    demogStr += f"s{sizeChangeNum} {sizeChangeNum} late()\n{{\n\tp1.setSubpopulationSize({nextSize});\n}}\n\n"
    rescheduleStr += f"\tsim.rescheduleScriptBlock(s{sizeChangeNum}, start={numGen}, end={numGen});\n"
    sizeChangeNum += 1

demogStr += f"s{sizeChangeNum} {sizeChangeNum} late()\n{{\n\tp1.outputMSSample({sampleSize}, replace=F, filePath=\"{outFileName}\");\n\tsim.simulationFinished();\n}}\n\n"
rescheduleStr += f"\tsim.rescheduleScriptBlock(s{sizeChangeNum}, start=endTime, end=endTime);\n"

slimProg = f"""initialize() {{
	initializeMutationType("m1", 0.5, "f", 0.0);
	defineConstant("endTime", {simDuration});

	initializeGenomicElementType("g1", m1, 1.0);
	initializeGenomicElement(g1, 0, {simChrSize-1});
	initializeMutationRate({mutRate});
	initializeRecombinationRate({r});
}}

1 early() {{
	sim.addSubpop("p1", {initialSize});
}}

{demogStr}

1 {{
{rescheduleStr}}}
"""

print(slimProg)
