import os
import random
import subprocess
import sys

# TODO add docs

(
    srcDir,
    scriptName,
    batch_start,
    sampleSizePerStepTS,
    numSamplesTS,
    samplingIntervalTS,
    sampleSizePerStep1Samp,
    numSamples1Samp,
    samplingInterval1Samp,
    numReps,
    physLen,
    timeSeries,
    sweep,
    dumpFileName,
    mutBaseName,
) = sys.argv[1:]

if timeSeries.lower() in ["false", "none"]:
    timeSeries = False
else:
    timeSeries = True

if "1Samp" in sweep:
    sweep = sweep.split("1Samp")[0]

if not sweep in ["hard", "soft", "neut"]:
    sys.exit("'sweep' argument must be 'hard', 'soft', or 'neut'")

batch_start = int(batch_start)
sampleSizePerStepTS = int(sampleSizePerStepTS)
numSamplesTS = int(numSamplesTS)
samplingIntervalTS = int(samplingIntervalTS)
sampleSizePerStep1Samp = int(sampleSizePerStep1Samp)
numSamples1Samp = int(numSamples1Samp)
numReps = int(numReps)
physLen = int(physLen)

# if "adaptiveIntrogression" in scriptName:
#    samplingInterval1Samp = (
#        200  #! Can this stay constant or does it need to be dynamically set?
#    )


for _batch in range(batch_start, batch_start + 20):
    for repIndex in range(numReps):
        sys.stderr.write(f"starting rep {repIndex}\n")
        seed = random.randint(0, 2 ** 32 - 1)

        if timeSeries:
            numSamples = numSamplesTS
            if "twoPop" in scriptName:
                sampleSizeStr = (
                    f"-d sampleSizePerStep1={sampleSizePerStepTS} -d sampleSizePerStep2={sampleSizePerStepTS}"
                )
            else:
                sampleSizeStr = f"-d sampleSizePerStep={sampleSizePerStepTS}"
            slimCmd = f"{srcDir}/SLiM/build/slim -seed {seed} {sampleSizeStr} \
                        -d samplingInterval={samplingIntervalTS} \
                        -d numSamples={numSamples} \
                        -d sweep='{sweep}' \
                        -d dumpFileName='{dumpFileName}' \
                        -d physLen={physLen} \
                        {scriptName}"
            print(slimCmd)

        else:
            numSamples = numSamples1Samp
            if "twoPop" in scriptName:
                sampleSizeStr = (
                    f"-d sampleSizePerStep1={sampleSizePerStep1Samp} -d sampleSizePerStep2={sampleSizePerStep1Samp}"
                )
            else:
                sampleSizeStr = f"-d sampleSizePerStep={sampleSizePerStep1Samp}"

            slimCmd = f"{srcDir}/SLiM/build/slim -seed {seed} {sampleSizeStr} \
                        -d samplingInterval={samplingInterval1Samp} \
                        -d numSamples={numSamples} \
                        -d sweep='{sweep}' \
                        -d dumpFileName='{dumpFileName}' \
                        -d physLen={physLen} \
                        {scriptName}"
            print(slimCmd)

        # sys.stderr.write(slimCmd)
        outstr = (
            subprocess.Popen(slimCmd.split(), stdout=subprocess.PIPE)
            .stdout.read()
            .decode()
            .splitlines()
        )

        if not os.path.exists(os.path.join(mutBaseName, "pops")):
            os.makedirs(os.path.join(mutBaseName, "pops"))

        with open(
            os.path.join(
                mutBaseName, "pops", "_".join([str(_batch), str(repIndex) + ".pop"])
            ),
            "w",
        ) as outfile:

            outfile.write(
                f"{srcDir}/SLiM/build/slim {sampleSizePerStepTS} {numSamples}\n"
            )
            for ol in outstr:
                outfile.write((ol + "\n"))

        os.system(f"rm {dumpFileName}")