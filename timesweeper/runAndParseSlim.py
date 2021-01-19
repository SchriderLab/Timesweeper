import os
import random
import subprocess
import sys

# TODO add docs

(
    scriptName,
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


sampleSizePerStepTS = int(sampleSizePerStepTS)
numSamplesTS = int(numSamplesTS)
samplingIntervalTS = int(samplingIntervalTS)
sampleSizePerStep1Samp = int(sampleSizePerStep1Samp)
numSamples1Samp = int(numSamples1Samp)
samplingInterval1Samp = int(samplingInterval1Samp)
numReps = int(numReps)
physLen = int(physLen)

tol = 0.5
for repIndex in range(numReps):
    sys.stderr.write("starting rep {}\n".format(repIndex))
    seed = random.randint(0, 2 ** 32 - 1)

    if timeSeries:
        numSamples = numSamplesTS
        if "twoPop" in scriptName:
            sampleSizeStr = "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(
                sampleSizePerStepTS, sampleSizePerStepTS
            )
        else:
            sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStepTS)
        slimCmd = "SLiM/build/slim -seed {} {} \
                    -d samplingInterval={} \
                    -d numSamples={} \
                    -d sweep='{}' \
                    -d dumpFileName='{}' \
                    -d physLen={} \
                    -d outFileName='{}' \
                    {}".format(
            seed,
            sampleSizeStr,
            samplingIntervalTS,
            numSamples,
            sweep,
            dumpFileName,
            physLen,
            mutBaseName + "/" + str(repIndex) + ".muts",
            scriptName,
        )

    else:
        numSamples = numSamples1Samp
        if "twoPop" in scriptName:
            sampleSizeStr = "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(
                sampleSizePerStep1Samp, sampleSizePerStep1Samp
            )
        else:
            sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStep1Samp)

        slimCmd = "SLiM/build/slim -seed {} {} \
                    -d samplingInterval={} \
                    -d numSamples={} \
                    -d sweep='{}' \
                    -d dumpFileName='{}' \
                    -d physLen={} \
                    -d outFileName='{}' \
                    {}".format(
            seed,
            sampleSizeStr,
            samplingInterval1Samp,
            numSamples,
            sweep,
            dumpFileName,
            physLen,
            mutBaseName + "/" + str(repIndex) + ".muts",
            scriptName,
        )

    if not os.path.exists(mutBaseName):
        os.makedirs(mutBaseName)

    sys.stderr.write(slimCmd)
    procOut = subprocess.Popen(
        slimCmd.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    output, err = procOut.communicate()
    print("SLiM/build/slim {} {}".format(sampleSizePerStepTS, numSamples))
    print(output.decode("utf-8"))
    os.system("rm {}".format(dumpFileName))