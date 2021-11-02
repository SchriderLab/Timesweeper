import os
import random
import subprocess
import sys

(
    scriptName,
    batch_start,
    numReps,
    physLen,
    sweep,
    dumpFileName,
    mutBaseName,
    total_samp_num,
) = sys.argv[1:]

if "1Samp" in sweep:
    sweep = sweep.split("1Samp")[0]

if not sweep in ["hard", "soft", "neut"]:
    sys.exit("'sweep' argument must be 'hard', 'soft', or 'neut'")

batch_start = int(batch_start)
total_samp_num = int(total_samp_num)
numReps = int(numReps)
physLen = int(physLen)

for _batch in range(batch_start, batch_start + 20):
    for repIndex in range(numReps):
        sys.stderr.write(f"starting rep {repIndex}\n")
        seed = random.randint(0, 2 ** 32 - 1)

        # SamplingInterval = Num gens for sampling window (200 by default)/num samples (40 by default)
        slimCmd = f"../SLiM/build/slim -seed {seed} \
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

        if not os.path.exists(os.path.join(mutBaseName, f"batch_{batch_start}")):
            os.makedirs(os.path.join(mutBaseName, f"batch_{batch_start}"))

        with open(
            os.path.join(
                mutBaseName,
                f"batch_{batch_start}",
                "_".join([str(_batch), str(repIndex) + ".pop"]),
            ),
            "w",
        ) as outfile:

            outfile.write(f"/SLiM/build/slim 1000 {total_samp_num}\n")
            for ol in outstr:
                outfile.write((ol + "\n"))

        os.system(f"rm {dumpFileName}")
