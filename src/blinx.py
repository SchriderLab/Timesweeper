import glob
import os
import argparse
from tqdm import tqdm
import sys
import shutil
import sys
from glob import glob
from typing import Dict


def intitialize_vars(baseDir, slim_file, sweep_index):
    """Creates necessary dictionaries/lists/names for a set of simulations.

    Args:
        baseDir (str): Base directory to write intermediate files to. \
            Should contain a folder called "slimfiles" containing \
            *.slim scripts.

        slim_index (int): Index for slimfile in slimFileList to use for simulations.

        sweep_index (int): Index for lists inside spd, keeps the parameterizations \
            uniform.

    Returns:
        spd (dict): Slim Parameters Dictionary, \
            a dict with all relevant parameterizations for the slim simulations.

        slimfile (str): Slim script to use for simulations.

        slimDir (str): Directory containing all intermediate files, \
            subdirectories for each step will be created.

        baseSimDir (str):  Where to write simulations to.

        baseDumpDir (str): Where to write simulation dumps to.

        baseLogDir (str): Where to write logfiles from SLiM to.
    """
    # SLiM params dict
    spd = {
        # individuals in sample for each time interval in time series
        "sampleSizePerStepTS": 20,
        # number of time points sampled in time series
        "numSamplesTS": [10, 20, 2, 40, 5],
        # spacing between time points
        "samplingIntervalTS": [20, 10, 100, 5, 40],
        # size of population sampled at one time point so that it is same size as time series data
        "sampleSizePerStep1Samp": [200, 400, 40, 800, 100],
        # number of time points sampled
        "numSamples1Samp": 1,
        # spacing between time points
        "samplingInterval1Samp": 0,
    }

    outName = "{}-{}Samp-{}Int".format(
        slim_file.split("/")[-1].split(".")[0],
        spd["numSamplesTS"][sweep_index],
        spd["samplingIntervalTS"][sweep_index],
    )

    slimDir = baseDir + "/" + outName

    baseSimDir = slimDir + "/sims"
    baseDumpDir = slimDir + "/simDumps"
    baseLogDir = slimDir + "/simLogs"

    if not os.path.exists(baseSimDir):
        for dir in [baseSimDir, baseDumpDir, baseLogDir]:
            os.makedirs(dir)

    return (
        spd,
        slim_file,
        slimDir,
        baseSimDir,
        baseDumpDir,
        baseLogDir,
    )


def launch_sims(
    srcDir,
    spd,
    slimFile,
    slimDir,
    baseSimDir,
    baseDumpDir,
    baseLogDir,
    sweep_index,
):
    """Creates and runs SLURM jobs for generating simulation data using SLiM.

    Args:

        spd (dict): Slim Parameters Dictionary.
            A dict with all relevant parameterizations for the slim simulations.

        slimFile (str): Name of slimfile being used to simulate with.
            Should be located in  baseDir/slimfiles/<slimFile>.

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.

        slimDir (str): Directory containing all intermediate files,
            subdirectories for each step will be created.

        baseSimDir (str): Where to write simulations to.

        baseDumpDir (str): Where to write simulation dumps to.

        baseLogDir (str): Where to write logfiles from SLiM to.

        sweep_index (int): Index of lists inside of spd.
            Used for iterating through parameterizations in an index-matched manner.

    """
    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
        os.system(
            "mkdir -p {}/{} {}/{} {}/{} {}/{}/muts".format(
                baseSimDir,
                name,
                baseLogDir,
                name,
                baseDumpDir,
                name,
                baseSimDir,
                name,
            )
        )

    if not os.path.exists(os.path.join(slimDir, "jobfiles")):
        os.mkdir(os.path.join(slimDir, "jobfiles"))

    physLen = 100000
    numBatches = 500
    repsPerBatch = 100
    for timeSeries in [True, False]:
        for i in tqdm(range(0, numBatches, 20), desc="\nSubmitting sim jobs...\n"):
            if timeSeries:
                suffix = ""
            else:
                suffix = "1Samp"

            # Only looking at hard sweeps for AI
            if "adaptiveIntrogression" in slimFile:
                simTypeList = ["hard", "neut"]
            elif "selectiveSweep" in slimFile:
                simTypeList = ["hard", "soft", "neut"]
            else:
                print("Error in simTypeList definition. Exiting.")
                sys.exit(1)

            for simType in simTypeList:
                simType = simType + suffix
                dumpDir = baseDumpDir + "/" + simType
                logDir = baseLogDir + "/" + simType

                mutBaseName = "{}/{}/muts/{}_{}".format(baseSimDir, simType, simType, i)

                dumpFileName = "{}/{}_{}.trees.dump".format(dumpDir, simType, i)
                cmd = "python {}src/runAndParseSlim.py {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}".format(
                    srcDir,
                    srcDir,
                    slimFile,
                    i,
                    spd["sampleSizePerStepTS"],
                    spd["numSamplesTS"][sweep_index],
                    spd["samplingIntervalTS"][sweep_index],
                    spd["sampleSizePerStep1Samp"][sweep_index],
                    spd["numSamples1Samp"],
                    spd["samplingInterval1Samp"],
                    repsPerBatch,
                    physLen,
                    timeSeries,
                    simType,
                    dumpFileName,
                    mutBaseName,
                )

                run_batch_job(
                    cmd,
                    simType,
                    "{}/jobfiles/{}{}.txt".format(slimDir, simType, suffix),
                    "2-00:00:00",
                    "general",
                    "2G",
                    "{}/{}_{}.log".format(logDir, simType, i),
                )


def clean_sims(
    srcDir,
    slimDir,
    baseSimDir,
    baseLogDir,
):
    """Finds and iterates through all raw msOut files recursively, \
        cleans them by stripping out unwanted lines. Submits SLURM jobs to \
            run each cleaning task.

    Args:

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.

        slimDir (str): Directory containing all intermediate files.
            Subdirectories for each step will be created.

        baseSimDir (str): Where to write simulations to.     
          
        baseLogDir (str): Where to write logfiles from SLiM to.
    """
    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
        batchnums = set(
            [
                i.split("/")[-1].split("_")[0]
                for i in glob(f"{baseSimDir}/{name}/muts/*/*.ms")
            ]
        )
        for k in tqdm(batchnums):
            cmd = "source activate blinx &&\
                    python {}src/clean_msOut.py {} {}".format(
                srcDir, f"{baseSimDir}/{name}/muts/", k
            )

            run_batch_job(
                cmd,
                "wash",
                "{}/jobfiles/wash.txt".format(slimDir),
                "5-00:00:00",
                "general",
                "2G",
                "{}/{}_clean.log".format(
                    baseLogDir, "{}/{}".format(baseSimDir, name).split("/")[-1]
                ),
            )


def create_shic_feats(
    srcDir: str, baseDir: str, baseSimDir: str, slimDir: str, baseLogDir: str
) -> None:
    """Finds all cleaned MS-format files recursively and runs diploSHIC fvecSim on them.
    Writes files to fvec subdirectory of sweep type.

    Args:

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.

        slimDir (str): Directory containing all intermediate files,
            subdirectories for each step will be created.
    """
    if not os.path.exists(os.path.join(baseLogDir, "fvec")):
        os.makedirs(os.path.join(baseLogDir, "fvec"))

    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
        cmd = "source activate blinx; python {}/src/make_fvecs.py {}/{} {}".format(
            srcDir, baseSimDir, name, baseDir
        )

        run_batch_job(
            cmd,
            "shic",
            "{}/jobfiles/shic.txt".format(slimDir),
            "5-00:00:00",
            "general",
            "8GB",
            "{}/{}/{}_shic_fvec.log".format(
                baseLogDir, "fvec", baseSimDir.split("/")[-1]
            ),
        )


def calculate_FIt(srcDir: str, slimDir: str, baseLogDir: str) -> None:
    """
    Runs SLURM jobs of feder_method.py to calculate FIt values for all simulated mutations.

    Args:

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.
        slimDir (str): Directory containing all intermediate files.
            Subdirectories for each step will be created.
        baseLogDir (str): Where to write logfiles from SLURM jobs to.

    TODO UPDATE THIS
    """
    if not os.path.exists(os.path.join(baseLogDir, "fitlogs")):
        os.makedirs(os.path.join(baseLogDir, "fitlogs"))

    for mutdir in tqdm(
        glob("{}/sims/*".format(slimDir)),
        desc="\nSubmitting FIt calculation jobs...\n",
    ):
        if "1Samp" not in mutdir:
            cmd = "source activate blinx; python {}/src/feder_method.py {}".format(
                srcDir, mutdir
            )

            run_batch_job(
                cmd,
                "fit_" + mutdir.split("/")[-1],
                "{}/jobfiles/fit.txt".format(slimDir),
                "2-00:00:00",
                "general",
                "1GB",
                "{}/{}/{}_feder.log".format(
                    baseLogDir, "fitlogs", mutdir.split("/")[-1].split(".")[0]
                ),
            )


def create_hap_input(
    srcDir: str, slimDir: str, baseLogDir: str, spd, sweep_index
) -> None:
    """
    Runs SLURM jobs of create_haplotype_ms.py to generate custom MS output files where mutations are tracked across haplotypes.

    Args:

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.
        slimDir (str): Directory containing all intermediate files.
            Subdirectories for each step will be created.
        baseLogDir (str): Where to write logfiles from SLURM jobs to.
    """
    if not os.path.exists(os.path.join(baseLogDir, "hapgenlogs")):
        os.makedirs(os.path.join(baseLogDir, "hapgenlogs"))

    for mutdir in glob(slimDir + "/sims/*"):
        if "1Samp" in mutdir:
            sampsize = spd["sampleSizePerStep1Samp"][sweep_index]
        else:
            sampsize = spd["sampleSizePerStepTS"]

        cmd = (
            "source activate blinx; python {}/src/create_haplotype_ms.py {} {}".format(
                srcDir, mutdir, sampsize
            )
        )

        print(cmd)

        run_batch_job(
            cmd,
            "hapg_" + mutdir.split("/")[-1],
            "{}/jobfiles/hapgen.txt".format(slimDir),
            "02:00:00",
            "general",
            "1GB",
            "{}/{}/{}_hapgen.log".format(
                baseLogDir, "hapgenlogs", mutdir.split("/")[-1].split(".")[0]
            ),
        )


def remove_temp_files(slimDir):
    """
    Will remove ALL *.log, and *.msOut files and empty directories.
    WARNING: ONLY RUN IF YOU HAVE CLEANED ALL FILES AND JUST NEED FVECS NOW.
    ALL SIMULATIONS WILL HAVE TO BE RE-LAUNCHED, CLEANED, AND CONVERTED TO FVECS.

    Args:
        slimDir (str): Base slim-simulation directory.
    """

    shutil.rmtree(os.path.join(slimDir, "simLogs"), ignore_errors=True)
    print("Removed", os.path.join(slimDir, "simLogs"))

    for badfile in tqdm(
        glob.glob(
            os.path.join(slimDir, "sims", "*", "cleaned", "*", "*.msOut"),
        ),
        desc="Deleting files",
    ):
        os.remove(badfile)

    for baddir in glob.glob(os.path.join(slimDir, "sims", "*", "rawMS")):
        shutil.rmtree(baddir, ignore_errors=False)

    print(
        "Removed all msOut files in",
        os.path.join(slimDir, "sims"),
        "and all subdirectories.",
    )

    shutil.rmtree(os.path.join(slimDir, "simDumps"), ignore_errors=True)
    print("Removed", os.path.join(slimDir, "simDumps"))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="A set of functions that run slurm \
           jobs to create and parse SLiM \
           simulations for sweep detection."
    )

    parser.add_argument(
        metavar="SCRIPT_FUNCTION",
        help="Use one of the available \
                functions by specifying its name.",
        dest="run_func",
        type=str,
        choices=[
            "launch",
            "clean",
            "make_feat_vecs",
            "calc_fit",
            "make_haps",
            "nuke",
        ],
    )

    parser.add_argument(
        "-s",
        "--slim-paramfile",
        metavar="SLIM_SIMULATION_FILE",
        help="Filename of slimfile in /slimfiles/ dir.\
                New directory will be created with this as prefix \
                and will contain all the relevant files for this \
                set of parameters.",
        dest="slim_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--nuke-dir",
        metavar="DIR_TO_NUKE",
        help="Directory to clean of all log files, all msOut files, and all unnecessary empty directories.",
        dest="nuke_dir",
        type=str,
        required=False,
    )

    args = parser.parse_args()

    if args.run_func == "nuke" and args.nuke_dir == None:
        print("Must provide a directory to nuke if using this mode.")
        sys.exit(1)

    if (
        args.run_func in ["launch", "clean", "make_feat_vecs", "make_haps"]
        and args.slim_file == None
    ):
        print("No slim file supplied for a function that requires it. Exiting.")
        sys.exit(1)

    return args


def run_batch_job(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile):
    """
    Creates a SLURM batch job template and submits.

    Args:

        cmd (str): Full string of commands to run in bash shell.
        jobName (str): Name of job
        launchFile (str): Path to txt file to be written with this script.
        wallTime (str): SLURM time format for running, d-hh:mm:ss
        qName (str): Queue name, usually "general" or "volta-gpu"
        mbMem (int): MegaBytes of memory to use for job
        logFile (str): Name of logfile to write stdout/stderr to from SLURM
    """
    with open(launchFile, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" % (jobName))
        f.write("#SBATCH --time=%s\n" % (wallTime))
        f.write("#SBATCH --partition=%s\n" % (qName))
        f.write("#SBATCH --output=%s\n" % (logFile))
        f.write("#SBATCH --mem=%s\n" % (mbMem))
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write("\n%s\n" % (cmd))
    os.system("sbatch %s" % (launchFile))


def main():
    ua = parse_arguments()

    for sweep_index in range(0, 5):
        srcDir = "/overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/"
        baseDir = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps"  # Make this an arg
        print("Base directory:", baseDir)

        spd, slimFile, slimDir, baseSimDir, baseDumpDir, baseLogDir = intitialize_vars(
            baseDir, ua.slim_file, sweep_index
        )

        print("Working directory:", slimDir)

        if ua.run_func == "launch":
            launch_sims(
                srcDir,
                spd,
                ua.slim_file,
                slimDir,
                baseSimDir,
                baseDumpDir,
                baseLogDir,
                sweep_index,
            )

        elif ua.run_func == "clean":
            clean_sims(
                srcDir,
                slimDir,
                baseSimDir,
                baseLogDir,
            )

        elif ua.run_func == "make_feat_vecs":
            create_shic_feats(srcDir, baseDir, baseSimDir, slimDir, baseLogDir)

        elif ua.run_func == "calc_fit":
            calculate_FIt(srcDir, slimDir, baseLogDir)

        elif ua.run_func == "make_haps":
            create_hap_input(srcDir, slimDir, baseLogDir, spd, sweep_index)

        elif ua.run_func == "nuke":
            remove_temp_files(ua.nuke_dir)


if __name__ == "__main__":
    main()
