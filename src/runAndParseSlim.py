import os
import random
import subprocess
import sys

# TODO add docs

(
    srcDir,
    scriptName,
    batch_start,
    numReps,
    physLen,
    sweep,
    dumpFileName,
    mutBaseName,
    total_samp_num,
    chroms_pool_size,
) = sys.argv[1:]

if "1Samp" in sweep:
    sweep = sweep.split("1Samp")[0]

if not sweep in ["hard", "soft", "neut"]:
    sys.exit("'sweep' argument must be 'hard', 'soft', or 'neut'")

batch_start = int(batch_start)
total_samp_num = int(total_samp_num)
chroms_pool_size = int(chroms_pool_size)
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

        if "twoPop" in scriptName:
            sampleSizeStr = f"-d sampleSizePerStep1={chroms_pool_size} -d sampleSizePerStep2={chroms_pool_size}"
        else:
            sampleSizeStr = f"-d sampleSizePerStep={chroms_pool_size}"

        # SamplingInterval = Num gens for sampling window (200 by default)/num samples (40 by default)
        slimCmd = f"{srcDir}SLiM/build/slim -seed {seed} {sampleSizeStr} \
                    -d samplingInterval={200/total_samp_num} \
                    -d numSamples={total_samp_num} \
                    -d sweep='{sweep}' \
                    -d dumpFileName='{dumpFileName}' \
                    -d physLen={physLen} \
                    {scriptName}"
        print(slimCmd)

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
                f"{srcDir}/SLiM/build/slim {chroms_pool_size} {total_samp_num}\n"
            )
            for ol in outstr:
                outfile.write((ol + "\n"))

        os.system(f"rm {dumpFileName}")