from glob import glob
import os

import pandas as pd
from tqdm import tqdm

from timesweeper.utils.gen_utils import read_config


def parse_cmd(cmd):
    """Create dictionary with all information in the bash command running slim."""
    cmd = [i for i in cmd.split() if i not in ["slim", "-d"]]

    cmd_dict = {}
    for i in cmd:
        if "=" in i:
            clean_i = i.split("=")
            cmd_dict[clean_i[0].replace("'", "").replace('"', "")] = (
                clean_i[1].replace("'", "").replace('"', "")
            )

    return cmd_dict


def get_samp_gens(loglist):
    """Get sampling generations."""
    samp_entries = [i for i in loglist if "Sampling at generation" in i]
    samp_entries = set(samp_entries)
    samp_gens = [i.split()[-1] for i in samp_entries]

    return sorted([int(i) for i in samp_gens])


def count_restarts(loglist):
    restarts = 0
    for i in loglist:
        if "RESTARTING" in i:
            restarts += 1

    return restarts


def track_sel_freq(loglist, num_samps, offset):
    freq_entries = [i for i in loglist if "SEGREGATING" in i and "FIXED" not in i]
    freqs = [float(i.split()[-1]) for i in freq_entries]
    if offset < 1:
        freqs = [0.0] + freqs[len(freqs)-num_samps+1:]
    else:
        freqs = freqs[len(freqs)-num_samps:]

    return freqs


def get_rep_from_filename(filename):
    rep = filename.split("/")[-1].split(".")[0]
    return rep


def parse_logfile(logfile):
    with open(logfile, "r") as ifile:
        loglist = [i.strip() for i in ifile.readlines()]

    log_dict = parse_cmd(loglist[0])
    log_dict["sampGens"] = get_samp_gens(loglist)
    log_dict["sampOffset"] = int(log_dict["sampGens"][0])
    log_dict["selAlleleFreq"] = track_sel_freq(loglist, len(log_dict["sampGens"]), log_dict["sampOffset"])
    log_dict["numRestarts"] = count_restarts(loglist)
    log_dict["rep"] = get_rep_from_filename(log_dict["outFileVCF"])
    for i in ["outFileVCF", "outFileMS", "dumpFile"]:
        del log_dict[i]
        
        return log_dict





def main(ua):
    yaml_data = read_config(ua.yaml_file)
    work_dir, schema = (
        yaml_data["work dir"],
        yaml_data["experiment name"]
    )

    logfiles = glob(f"{work_dir}/logs/*/*.log", recursive=True)

    log_dict_list = []
    for l in tqdm(logfiles, desc="Parsing logs"):
        try:
            log_dict_list.append(parse_logfile(l))
        except:
            continue
    df = pd.DataFrame(log_dict_list)
    df = df[
        [
            "rep",
            "sweep",
            "selCoeff",
            "sampOffset",
            "numRestarts",
            "seed",
            "physLen",
            "sampGens",
            "selAlleleFreq",
        ]
    ]
    df.loc[df["sweep"] == "neut", "selCoeff"] = 0.0
    df.to_csv(f"{work_dir}/{schema}_params.tsv", index=False, header=True, sep="\t")

