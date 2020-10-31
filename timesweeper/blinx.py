import glob
import os
from tqdm import tqdm
import utils as ut

# Testing vars #############################################################
# TODO set these as argparse args OR as file naming in slimfiles? JSON?
# Time Series
sampleSizePerStepTS = 20  # individuals in sample for each time interval in time series
numSamplesTS = 10  # number of time points sampled in time series
samplingIntervalTS = 20  # spacing between time points

# 1 Sample at 1 One Time Point
sampleSizePerStep1Samp = 200  # size of population sampled at one time point so that it is same size as time series data
numSamples1Samp = 1  # number of time points sampled
samplingInterval1Samp = 200  # spacing between time points
# TODO store these as a json made by user
maxSnps = 200

baseDir = "/proj/dschridelab/timeSeriesSweeps"
slimFile = "onePop-selectiveSweep-10Samp20Ind20Int-sweep"  # No ext

slimDir = baseDir + "/" + slimFile

baseSimDir = slimDir + "/sims"
baseDumpDir = slimDir + "/simDumps"
baseLogDir = slimDir + "/simLogs"

# End Testing vars #########################################################


def launch_sims():
    """TODO Add docs

    TODO Figure out if it's possible to ignore "Submitted job" message
    """
    for name in ["hard", "soft", "neut"]:  # , "hard1Samp", "soft1Samp", "neut1Samp"]:
        os.system(
            "mkdir -p {}/{} {}/{} {}/{} {}/{}/rawMS".format(
                baseSimDir, name, baseLogDir, name, baseDumpDir, name, baseSimDir, name
            )
        )

    if not os.path.exists(os.path.join(slimDir, "jobfiles")):
        os.mkdir(os.path.join(slimDir, "jobfiles"))

    physLen = 100000
    numBatches = 1000
    repsPerBatch = 100
    for timeSeries in [True, False]:
        for i in tqdm(range(numBatches), desc="\nSubmitting sim jobs...\n"):
            if timeSeries:
                suffix = ""
            else:
                suffix = "1Samp"
            for simType in ["hard", "soft", "neut"]:
                dumpDir = baseDumpDir + "/" + simType + suffix
                logDir = baseLogDir + "/" + simType + suffix
                outFileName = "{}/{}/rawMS/{}_{}.msOut".format(
                    baseSimDir, simType, simType, i
                )
                dumpFileName = "{}/{}_{}.trees.dump".format(dumpDir, simType, i)
                # Replace /test/ with slimdfile directory
                cmd = "python {}/timesweeper/runAndParseSlim.py {}/slimfiles/{}.slim {} {} {} {} {} {} {} {} {} {} {} > {}".format(
                    baseDir,
                    baseDir,
                    slimFile,
                    sampleSizePerStepTS,
                    numSamplesTS,
                    samplingIntervalTS,
                    sampleSizePerStep1Samp,
                    numSamples1Samp,
                    samplingInterval1Samp,
                    repsPerBatch,
                    physLen,
                    timeSeries,
                    simType,
                    dumpFileName,
                    outFileName,
                )

                ut.run_batch_job(
                    cmd,
                    simType + suffix,
                    "{}/jobfiles/{}{}.txt".format(slimDir, simType, suffix),
                    "10:00",
                    "general",
                    "1G",
                    "{}/{}_{}.log".format(logDir, simType, i),
                )


def clean_sims():
    """Finds and iterates through all raw msOut files recursively, \
        cleans them by stripping out unwanted lines.

    #TODO make this a submission?
    """
    i = 0
    for dirtyfile in tqdm(
        glob.glob("{}/*/rawMS/*.msOut".format(baseSimDir), recursive=True),
        desc="\nSubmitting cleaning jobs...\n",
    ):
        cmd = "source activate blinx;\
                python {}/timesweeper/clean_msOut.py {}".format(
            baseDir, dirtyfile
        )

        ut.run_batch_job(
            cmd,
            "wash",
            "{}/jobfiles/wash.txt".format(slimDir),
            "05:00",
            "general",
            "128Mb",
            "{}/{}_clean.log".format(baseLogDir, i),
        )

        i += 1


def create_shic_feats():
    """Finds all cleaned MS-format files recursively and runs diploSHIC fvecSim on them.
    Writes files to fvec subdirectory of sweep type.
    #TODO Make an arg pass to this to specify which folder you want
    """
    for cleanfile in tqdm(
        glob.glob("{}/**/cleaned/*/*point*.msOut".format(slimDir), recursive=True),
        desc="\nSubmitting SHIC generation jobs...\n",
    ):
        filepath = os.path.split(cleanfile)[0]
        filename = os.path.split(cleanfile)[1].split(".")[0]

        if not os.path.exists(os.path.join(filepath, "fvecs")):
            os.mkdir(os.path.join(filepath, "fvecs"))
            os.mkdir(os.path.join(filepath, "fvecs/logs"))

        cmd = "source activate blinx;\
            python {}/diploSHIC/diploSHIC.py fvecSim haploid {} {}".format(
            baseDir, cleanfile, os.path.join(filepath, "fvecs", filename + ".fvec")
        )

        ut.run_batch_job(
            cmd,
            "shic",
            "{}/jobfiles/shic.txt".format(slimDir),
            "1:00:00",
            "general",
            "2GB",
            "{}/{}_shic_fvec.log".format(
                os.path.join(filepath, "fvecs", "logs"), filename
            ),
        )


def train_nets():
    # This is out for now, need to make new model
    prefixLs = ["hard_v_neut_ttv_ali", "hard_v_neut_ttv_haps", "hard_v_neut_ttv_sfs"]
    simTypeToScript = {
        "": "../keras_CNN_loadNrun.py",
        "1Samp": "../keras_DNN_loadNrun.py",
    }
    for simType in ["", "1Samp"]:
        outDir = "{}/classifiers{}".format(baseDir, simType)
        os.system("mkdir -p {}".format(outDir))

        for prefix in prefixLs:
            cmd = "python {0} -i {1}/npzs{2}/{3}.npz -c {4}/{3}.mod".format(
                simTypeToScript[simType], baseDir, simType, prefix, outDir
            )
            ut.run_batch_job(
                cmd,
                "trainTS",
                "trainTS.txt",
                "12:00:00",
                "general",
                "32GB",
                outDir + "/{}.log".format(prefix),
            )


def main():
    ua = ut.parse_arguments()

    # TODO Gotta be a better way to do this
    if ua.run_func == "launch_sims":
        launch_sims()
    elif ua.run_func == "clean_sims":
        clean_sims()
    elif ua.run_func == "create_feat_vecs":
        create_shic_feats()
    elif ua.run_func == "train_nets":
        train_nets()


if __name__ == "__main__":
    main()
