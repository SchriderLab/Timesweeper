import argparse
import glob
import os
import sys
from glob import glob

from tqdm import tqdm


def intitialize_vars(baseDir, slim_file):
    """Creates necessary dictionaries/lists/names for a set of simulations.

    Args:
        baseDir (str): Base directory to write intermediate files to. \
            Should contain a folder called "slimfiles" containing \
            *.slim scripts.

    Returns:
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

    outName = f"{slim_file.split('/')[-1].split('.')[0]}"

    slimDir = baseDir + "/" + outName

    baseSimDir = slimDir + "/sims"
    baseDumpDir = slimDir + "/simDumps"
    baseLogDir = slimDir + "/simLogs"

    if not os.path.exists(baseSimDir):
        for dir in [baseSimDir, baseDumpDir, baseLogDir]:
            os.makedirs(dir)

    return (
        slimDir,
        baseSimDir,
        baseDumpDir,
        baseLogDir,
    )


def launch_sims(
    srcDir,
    slimFile,
    slimDir,
    baseSimDir,
    baseDumpDir,
    baseLogDir,
    sample_pool_size=40,
    chroms_pool_size=800,
):
    """Creates and runs SLURM jobs for generating simulation data using SLiM.

    Args:
        slimFile (str): Name of slimfile being used to simulate with.
            Should be located in  baseDir/slimfiles/<slimFile>.

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.

        slimDir (str): Directory containing all intermediate files,
            subdirectories for each step will be created.

        baseSimDir (str): Where to write simulations to.

        baseDumpDir (str): Where to write simulation dumps to.

        baseLogDir (str): Where to write logfiles from SLiM to.
    """
    for name in ["hard", "soft", "neut"]:
        os.system(
            f"mkdir -p {baseSimDir}/{name} {baseLogDir}/{name} {baseDumpDir}/{name} {baseSimDir}/{name}"
        )

    if not os.path.exists(os.path.join(slimDir, "jobfiles")):
        os.mkdir(os.path.join(slimDir, "jobfiles"))

    physLen = 100000
    numBatches = 500
    repsPerBatch = 100
    for i in tqdm(range(0, numBatches, 20), desc="\nSubmitting sim jobs...\n"):
        # Only looking at hard sweeps for AI
        if "adaptiveIntrogression" in slimFile:
            simTypeList = ["hard", "neut"]
        elif "selectiveSweep" in slimFile:
            simTypeList = ["hard", "soft", "neut"]
        else:
            print("Error in simTypeList definition. Exiting.")
            sys.exit(1)

        for simType in simTypeList:
            dumpDir = baseDumpDir + "/" + simType
            logDir = baseLogDir + "/" + simType
            mutBaseName = f"{baseSimDir}/{simType}"
            dumpFileName = f"{dumpDir}/{simType}_{i}.trees.dump"
            cmd = f"python {srcDir}src/runAndParseSlim.py {srcDir} {slimFile} {i} {repsPerBatch} {physLen} {simType} {dumpFileName} {mutBaseName} {sample_pool_size} {chroms_pool_size}"

            run_batch_job(
                cmd,
                simType,
                f"{slimDir}/jobfiles/{simType}.txt",
                "6:00:00",
                "general",
                "2G",
                f"{logDir}/{simType}_{i}.log",
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

    #! Deprecated 

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
            cmd = f"source activate blinx && python {srcDir}src/clean_msOut.py {baseSimDir}/{name}/muts/ {k}"

            run_batch_job(
                cmd,
                "wash",
                f"{slimDir}/jobfiles/wash.txt",
                "5-00:00:00",
                "general",
                "2G",
                f"{baseLogDir}/{'{}/{}'.format(baseSimDir, name).split('/')[-1]}_clean.log",
            )


def create_shic_feats(
    srcDir: str, baseDir: str, baseSimDir: str, slimDir: str, baseLogDir: str
) -> None:
    """Finds all cleaned MS-format files recursively and runs diploSHIC fvecSim on them.
    Writes files to fvec subdirectory of sweep type.

    #! Deprecated

    Args:

        baseDir (str): Base directory of timesweeper intermediate files.
            Should contain a subdir called slimfiles.

        slimDir (str): Directory containing all intermediate files,
            subdirectories for each step will be created.
    """
    if not os.path.exists(os.path.join(baseLogDir, "fvec")):
        os.makedirs(os.path.join(baseLogDir, "fvec"))

    for name in ["hard", "soft", "neut", "hard1Samp", "soft1Samp", "neut1Samp"]:
        cmd = f"source activate blinx; python {srcDir}/src/make_fvecs.py {baseSimDir}/{name} {baseDir}"

        run_batch_job(
            cmd,
            "shic",
            f"{slimDir}/jobfiles/shic.txt",
            "5-00:00:00",
            "general",
            "8GB",
            f"{baseLogDir}/fvec/{baseSimDir.split('/')[-1]}_shic_fvec.log",
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
        glob(f"{slimDir}/sims/*"),
        desc="\nSubmitting FIt calculation jobs...\n",
    ):
        if "1Samp" not in mutdir:
            cmd = f"source activate blinx; python {srcDir}/src/feder_method.py {mutdir}"

            run_batch_job(
                cmd,
                "fit_" + mutdir.split("/")[-1],
                f"{slimDir}/jobfiles/fit.txt",
                "2-00:00:00",
                "general",
                "1GB",
                f"{baseLogDir}/fitlogs/{mutdir.split('/')[-1].split('.')[0]}_feder.log",
            )


def create_hap_input(
    srcDir: str, slimDir: str, baseLogDir: str, spd, sweep_index
) -> None:
    """
    #TODO Either convert this to work with updated haplotype script or remove
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

        cmd = f"source activate blinx; python {srcDir}/src/create_haplotype_ms.py {mutdir} {sampsize}"

        print(cmd)

        run_batch_job(
            cmd,
            "hapg_" + mutdir.split("/")[-1],
            f"{slimDir}/jobfiles/hapgen.txt",
            "02:00:00",
            "general",
            "1GB",
            f"{baseLogDir}/hapgenlogs/{mutdir.split('/')[-1].split('.')[0]}_hapgen.log",
        )


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
        f.write(f"#SBATCH --job-name={jobName}\n")
        f.write(f"#SBATCH --time={wallTime}\n")
        f.write(f"#SBATCH --partition={qName}\n")
        f.write(f"#SBATCH --output={logFile}\n")
        f.write(f"#SBATCH --mem={mbMem}\n")
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write(f"\n{cmd}\n")
    os.system(f"sbatch {launchFile}")


def main():
    ua = parse_arguments()

    srcDir = "/overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/"
    baseDir = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps"  # Make this an arg
    print("Base directory:", baseDir)

    slimDir, baseSimDir, baseDumpDir, baseLogDir = intitialize_vars(
        baseDir, ua.slim_file
    )

    print("Working directory:", slimDir)

    SAMPLE_POOL_SIZE = 40  # Total number of samples to write to output, will be filtered in haplotype featvec creation
    CHROMS_POOL_SIZE = (
        800  # Total number of chromosomes to sample from the total pool being output
    )

    if ua.run_func == "launch":
        launch_sims(
            srcDir,
            ua.slim_file,
            slimDir,
            baseSimDir,
            baseDumpDir,
            baseLogDir,
            SAMPLE_POOL_SIZE,
            CHROMS_POOL_SIZE,
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


if __name__ == "__main__":
    main()
