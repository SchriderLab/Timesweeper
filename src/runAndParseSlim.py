import os
import random
import subprocess
import sys

# TODO add docs

(
    srcDir,
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
            mutBaseName + "/" + str(repIndex) + ".muts",
            scriptName,
        )

    # sys.stderr.write(slimCmd)
    outstr = (
        subprocess.Popen(slimCmd.split(), stdout=subprocess.PIPE)
        .stdout.read()
        .decode()
        .splitlines()
    )

    if timeSeries:
        # Clean up sims so we only get successful run
        gens = []
        gen_inds = []
        for (idx, genline) in zip(range(len(outstr)), outstr):

            if "#OUT:" in genline:
                gens.append(int(genline.split(" ")[1]))
                gen_inds.append(idx)

        # Find last instance of the first sampling timepoint
        min_gen_ind = gen_inds[len(gens) - 1 - gens[::-1].index(min(gens)) + 1]
        clean_output = outstr[min_gen_ind:]

    else:
        clean_output = outstr

    if not os.path.exists(mutBaseName):
        os.makedirs(mutBaseName)

    with open(os.path.join(mutBaseName, str(repIndex) + ".pop"), "w") as outfile:

        outfile.write(
            "{}/SLiM/build/slim {} {}\n".format(srcDir, sampleSizePerStepTS, numSamples)
        )
        for ol in clean_output:
            outfile.write((ol + "\n"))

    os.system("rm {}".format(dumpFileName))