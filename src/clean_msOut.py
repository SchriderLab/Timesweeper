import os
import sys
from glob import glob
import csv
from tqdm import tqdm
import subprocess


def get_index_positions(list_of_elems, element):
    """Returns the indexes of all occurrences of give element in
    the list- listOfElements"""
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


def clean_msOut(msFile):
    """Reads in MS-style output from Slim, iterates line-by-line to find relevant portions of SLiM msOut file.
    Redundant (fixed and repeated) time samples are removed, and the final ms-style entry is written to its own timepoint file.

    Args:

        msFile (str): Full filepath of slim output file to clean.
    """

    cleandir = msFile.split(".")[-2]
    os.makedirs(cleandir, exist_ok=True)

    if "1Samp" in msFile:
        num_samps = 1
    else:
        num_samps = int(msFile.split("-")[2].split("Samp")[0])

    with open(msFile, "r") as rawfile:
        rawMS = [i.strip() for i in rawfile.readlines()]

    # Required for SHIC to run
    shic_header = "SLiM/build/slim 20 1"

    header_pos = get_index_positions(rawMS, "//")[-num_samps:]

    for i in range(len(header_pos)):
        if i < len(header_pos) - 1:
            _ms = rawMS[header_pos[i] : header_pos[i + 1]]
        else:
            _ms = rawMS[header_pos[i] :]

        _ms.insert(0, shic_header)

        # Write each timepoint to it's own file for each rep
        with open(
            os.path.join(
                cleandir,
                str(i) + ".ms",
            ),
            "w",
        ) as outFile:
            outFile.write("\n".join(_ms))

    return cleandir


def main():
    for i in tqdm(glob(os.path.join(sys.argv[1] + f"*/{sys.argv[2]}_*.ms"))):
        print(i)
        try:
            cleandir = clean_msOut(i)
            for k in glob(cleandir + "/*.ms"):
                if not os.path.exists(k.split(".")[0] + ".fvec"):
                    cmd = "python /overflow/dschridelab/users/lswhiteh/timeSeriesSweeps/diploSHIC/diploSHIC.py fvecSim haploid {} {} ".format(
                        k, k.split(".")[0] + ".fvec"
                    )
                    print(cmd)
                    subprocess.run(cmd, shell=True)
                os.remove(k)
        except:
            print("Couldn't wash {}".format(i))

        os.remove(i)


if __name__ == "__main__":
    main()