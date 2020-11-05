import os
import sys
from glob import glob


def clean_msOut(msFile):
    """Reads in MS-style output from Slim, removes all extraneous information \
        so that the MS output is the only thing left. Writes to "cleaned" msOuts \
        for each timepoint sampled during the SLiM run.

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

    # First split by replicates, each item in list is a replicate of timepoints
    ms_list = split_ms_to_list(rawMS, "SLiM/build/slim")
    print("{} MS Replicates".format(len(ms_list)))

    # Required for SHIC to run
    shic_header = [s for s in rawMS if "SLiM/build/slim" in s][0].split()
    shic_header[-1] = "1"
    shic_header = " ".join(shic_header)
    print(shic_header)
    rep = 1
    for subMS in ms_list:
        # Skip any potential empty lists from the split
        if not subMS:
            continue

        # Split into timepoints, throw away restarts
        unfilt_timepoints = split_ms_to_list(subMS, "segsites")
        trimmed_tps = [s for s in unfilt_timepoints if "RESTARTING" not in "".join(s)]
        trimmed_tps = [s for s in trimmed_tps if s]

        # Clean up SLiM info
        cleaned_subMS = []
        for i in trimmed_tps:
            # Remove newlines and empty lists
            i = [s for s in i if len(s) > 0]
            cleaned_subMS.append(filter_unwanted_slim(i))

        cleaned_subMS = [s for s in cleaned_subMS if "segsites" in s[0]]

        print("{} timepoints in this rep".format(len(cleaned_subMS)))

        point = 1
        for single_ms in cleaned_subMS:
            # Make sure shic header is present
            single_ms_final = insert_shic_header(single_ms, shic_header)

            if single_ms_final[-1] == "//":
                single_ms_final.pop(-1)

            # Write each timepoint to it's own file for each rep
            with open(
                os.path.join(
                    filepath,
                    "cleaned",
                    series_label,
                    "rep_" + str(rep) + "_point_" + str(point) + "_" + filename,
                ),
                "w",
            ) as outFile:
                outFile.write("\n".join(single_ms_final))
            point += 1
        rep += 1


def split_ms_to_list(rawMS, splitter):
    """Splits the list of lines into multiple lists, separating by splitter str.
    https://www.geeksforgeeks.org/python-split-list-into-lists-by-particular-value/
    """
    size = len(rawMS)
    idx_list = [idx for idx, val in enumerate(rawMS) if splitter in val]
    try:
        ms_list = [
            rawMS[i:j]
            for i, j in zip(
                [0] + idx_list, idx_list + ([size] if idx_list[-1] != size else [])
            )
        ]

    except IndexError:
        ms_list = []

    return ms_list


def get_last_restart(subMS):
    """If mut gets thrown out too quickly sim will restart
    throw out anything from those failed runs

    Args:
        subMS (list[str]): Single list of MS entry separated by //

    Returns:
        last_restart (int): Index in subMS of last restart location in list
    """
    last_restart = 0
    for i in range(len(subMS)):
        try:
            if "RESTARTING" in subMS[i]:
                last_restart = i
        except IndexError:
            continue

    return last_restart


def filter_unwanted_slim(subMS):
    """Removes any SLiM-related strings from MS entry so that only MS is left.

    Args:

        subMS (list[str]): List of strings describing one MS entry split using \
            split_ms_to_list()

    Returns:
        cleaned_subMS: MS entry with no extraneous information from SLiM
    """
    # Clean up unwanted strings
    cleaned_subMS = []
    for i in range(len(subMS)):
        # Filter out lines where integer line immediately follows
        # Cleaner way to do this?
        if (
            (subMS[i] == "// Initial random seed:")
            or (subMS[i] == "// Starting run at generation <start>:")
            or (subMS[i - 1] == "// Initial random seed:")
            or (subMS[i - 1] == "// Starting run at generation <start>:")
            or ("#OUT" in subMS[i])
            or ("SEGREGATING" in subMS[i])
            or ("Done emitting sample" in subMS[i])
            or ("Sampling" in subMS[i])
            or (";" in subMS[i])
            or ("INTRODUCED" in subMS[i])
            or ("SAVING" in subMS[i])
        ):
            continue

        # Filter out commented lines that aren't ms related
        # e.g. '// RunInitializeCallbacks():'
        elif (subMS[i].split()[0] == "//") and (len(subMS[i].split()) > 1):
            continue

        else:
            cleaned_subMS.append(subMS[i])

    return cleaned_subMS


def insert_shic_header(cleaned_subMS, shic_header):
    """Checks for shic-required header, inserts into MS entry if not present.

    Args:

        cleaned_subMS (list[str]): MS entry cleaned using filter_unwanted_slim()
        shic_header (str): String in order of <tool> <samples> <timepoints>?

    Returns:
        cleaned_subMS: MS entry with shic header inserted into first value.
    """
    try:
        if "SLiM/build/slim" not in cleaned_subMS[0]:
            cleaned_subMS.insert(0, "//")
            cleaned_subMS.insert(0, shic_header)
    except IndexError:
        pass

    return cleaned_subMS


def main():
    clean_msOut(sys.argv[1])


if __name__ == "__main__":
    main()