import os
import sys
from glob import glob
import csv


def clean_msOut(msFile):
    """Reads in MS-style output from Slim, iterates line-by-line to find relevant portions of SLiM msOut file.
    Redundant (fixed and repeated) time samples are removed, and the final ms-style entry is written to its own timepoint file.

    Args:

        msFile (str): Full filepath of slim output file to clean.
    """
    filepath = "/".join(os.path.split(msFile)[0].split("/")[:-1])
    filename = os.path.split(msFile)[1]
    series_label = filename.split(".")[0].split("_")[-1]

    # Separate dirs for each time series of cleaned and separated files
    if not os.path.exists(os.path.join(filepath, "cleaned", series_label)):
        os.makedirs(os.path.join(filepath, "cleaned", series_label))

    with open(msFile, "r") as rawfile:
        rawMS = [i.strip() for i in rawfile.readlines()]

    # Required for SHIC to run
    shic_header = [s for s in rawMS if "SLiM/build/slim" in s][0].split()
    shic_header[-1] = "1"
    shic_header = " ".join(shic_header)

    rep_counter = 0
    point_counter = 0

    reps = []
    gens = []
    ms_entries = []
    _ms = []
    for line_idx in range(len(rawMS)):
        if "#OUT:" in rawMS[line_idx]:
            _ms = []

            num_sampled = int(rawMS[line_idx].split()[-1])
            _ms.append(rawMS[line_idx])
            _ms.append(rawMS[line_idx + 1])

            gen = int(rawMS[line_idx].split()[1])
            if len(gens) > 1:
                if gen == gens[-1]:
                    # Restart!
                    ms_entries = []
                    point_counter = 0
                    gens = []

                else:
                    gens.append(gen)
                    point_counter += 1

            else:
                gens.append(gen)

        elif "segsites:" in rawMS[line_idx]:
            _ms.append(rawMS[line_idx])

        elif "positions:" in rawMS[line_idx]:
            _ms.append(rawMS[line_idx])
            _ms.extend(rawMS[line_idx + 1 : line_idx + num_sampled + 1])
            ms_entries.append(_ms)

        elif "SLiM/build/slim" in rawMS[line_idx] and line_idx != 0:
            reps.append(ms_entries)
            rep_counter += 1
            point_counter = 0
            ms_entries = []
            gens = []

    rep_lab = 0
    for rep in reps:
        point_lab = 0
        for entry in rep:
            # Make sure shic header is present
            entry[0] = shic_header

            # Write each timepoint to it's own file for each rep
            with open(
                os.path.join(
                    filepath,
                    "cleaned",
                    series_label,
                    "rep_" + str(rep_lab) + "_point_" + str(point_lab) + "_" + filename,
                ),
                "w",
            ) as outFile:
                outFile.write("\n".join(entry))

            point_lab += 1
        rep_lab += 1


def main():
    for i in glob(sys.argv[1] + "/*msOut"):
        print(i)
        try:
            clean_msOut(i)
        except:
            print("Couldn't wash {}".format(i))


if __name__ == "__main__":
    main()