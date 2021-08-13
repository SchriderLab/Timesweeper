import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-N",
    dest="N",
    help="final effective population size (used both for SLiM and for parsing out u and r)",
    type=int,
    required=True,
)
parser.add_argument(
    "-L",
    dest="L",
    help="size of the simulated chromosome in bp (used both for SLiM and for parsing out u and r)",
    type=int,
    required=True,
)
parser.add_argument(
    "-ms",
    dest="msCmd",
    help="quoted ms command to be converted into a SLiM script",
    required=True,
    metavar="msCmd",
)
parser.add_argument(
    "-b",
    dest="burnTime",
    help="number of burn-in generations to add before simulating demographic model",
    type=int,
    required=True,
    metavar="burnTime",
)
args = parser.parse_args()

effectivePopSize = args.N
simChrSize = args.L
burnTime = args.burnTime
msCmd = args.msCmd

msParser = argparse.ArgumentParser()
msParser.add_argument("ms")
msParser.add_argument("sampleSize", type=int)
msParser.add_argument("numReps", type=int)
msParser.add_argument("-en", action="append", dest="popSizeChanges", nargs="+")
msParser.add_argument("-t", dest="theta", type=float)
msParser.add_argument("-r", dest="rho", nargs=2)
msArgs = msParser.parse_args(msCmd.split())

if msArgs.ms != "ms":
    sys.exit("Error: malformed ms command (doesn't start with 'ms')")
sampleSize = msArgs.sampleSize

theta = msArgs.theta
rho, numRecSites = msArgs.rho
rho = float(rho)
numRecSites = int(numRecSites)
popSizeChanges = msArgs.popSizeChanges

u = theta / (4 * effectivePopSize)
r = rho / (4 * effectivePopSize)

sizeChangesDiscrete = []
if popSizeChanges:
    nextSize = effectivePopSize
    popSizeChanges.sort(key=lambda x: x[0])

    for time, popId, sizeRatio in popSizeChanges:
        time, popId, sizeRatio = float(time), int(popId), float(sizeRatio)
        # print(time, popId, sizeRatio)
        numGen = int(round(time * 4 * effectivePopSize)) + burnTime
        prevSize = int(round(effectivePopSize * sizeRatio))
        # print(f"size changes from {prevSize} to {nextSize} individuals {numGen} generations into the simulation")
        sizeChangesDiscrete.append((numGen, prevSize, nextSize))
        nextSize = prevSize

    oldestTime = float(popSizeChanges[-1][0])
    simDuration = int(round(oldestTime * 4 * effectivePopSize)) + burnTime

    sizeChangesDiscrete.reverse()
    initialSize = sizeChangesDiscrete[0][1]
else:
    simDuration = burnTime
    initialSize = effectivePopSize

demogStr = ""
rescheduleStr = ""

sizeChangeNum = 1
for numGen, prevSize, nextSize in sizeChangesDiscrete:
    demogStr += f"s{sizeChangeNum} {sizeChangeNum} late()\n{{\n\tp1.setSubpopulationSize({nextSize});\n}}\n\n"
    rescheduleStr += f"\tsim.rescheduleScriptBlock(s{sizeChangeNum}, start={numGen}, end={numGen});\n"
    sizeChangeNum += 1

slimProg = f"""initialize() {{
	initializeMutationType("m1", 0.5, "f", 0.0);
	defineConstant("endTime", {simDuration});

	initializeGenomicElementType("g1", m1, 1.0);
	initializeGenomicElement(g1, 0, {simChrSize-1});
	initializeMutationRate({u});
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