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
samplingInterval1Samp = int(samplingInterval1Samp)
numReps = int(numReps)
physLen = int(physLen)

tol = 0.5
for _batch in range(batch_start, batch_start + 20):
    for repIndex in range(numReps):
        sys.stderr.write("starting rep {}\n".format(repIndex))
        seed = random.randint(0, 2 ** 32 - 1)

        if timeSeries:
            numSamples = numSamplesTS
            if "twoPop" in scriptName:
                sampleSizeStr = (
                    "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(
                        sampleSizePerStepTS, sampleSizePerStepTS
                    )
                )
            else:
                sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStepTS)
            slimCmd = "{}/SLiM/build/slim -seed {} {} \
                        -d samplingInterval={} \
                        -d numSamples={} \
                        -d sweep='{}' \
                        -d dumpFileName='{}' \
                        -d physLen={} \
                        -d outFileName='{}' \
                        {}".format(
                srcDir,
                seed,
                sampleSizeStr,
                samplingIntervalTS,
                numSamples,
                sweep,
                dumpFileName,
                physLen,
                mutBaseName + "/" + str(_batch) + "_" + str(repIndex) + ".ms",
                scriptName,
            )
            print(slimCmd)

        else:
            numSamples = numSamples1Samp
            if "twoPop" in scriptName:
                sampleSizeStr = (
                    "-d sampleSizePerStep1={} -d sampleSizePerStep2={}".format(
                        sampleSizePerStep1Samp, sampleSizePerStep1Samp
                    )
                )
            else:
                sampleSizeStr = "-d sampleSizePerStep={}".format(sampleSizePerStep1Samp)

            slimCmd = "{}/SLiM/build/slim -seed {} {} \
                        -d samplingInterval={} \
                        -d numSamples={} \
                        -d sweep='{}' \
                        -d dumpFileName='{}' \
                        -d physLen={} \
                        -d outFileName='{}' \
                        {}".format(
                srcDir,
                seed,
                sampleSizeStr,
                samplingInterval1Samp,
                numSamples,
                sweep,
                dumpFileName,
                physLen,
                os.path.join(
                    mutBaseName, "_".join([str(_batch), str(repIndex) + ".ms"])
                ),  # Is this still needed?
                scriptName,
            )
            print(slimCmd)

        # sys.stderr.write(slimCmd)
        outstr = (
            subprocess.Popen(slimCmd.split(), stdout=subprocess.PIPE)
            .stdout.read()
            .decode()
            .splitlines()
        )

        if not os.path.exists(os.path.join(mutBaseName)):
            os.makedirs(os.path.join(mutBaseName))

        with open(
            os.path.join(mutBaseName, "_".join([str(_batch), str(repIndex) + ".pop"])),
            "w",
        ) as outfile:

            outfile.write(
                "{}/SLiM/build/slim {} {}\n".format(
                    srcDir, sampleSizePerStepTS, numSamples
                )
            )
            for ol in outstr:
                outfile.write((ol + "\n"))

        os.system("rm {}".format(dumpFileName))