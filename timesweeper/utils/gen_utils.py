import yaml
import pandas as pd
import os
import numpy as np
import logging


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


def add_file_label(filename, label):
    """Injects a model identifier to the outfile name."""
    splitfile = filename.split(".")
    newname = f"{''.join(splitfile[:-1])}_{label}.{splitfile[-1]}"
    return newname


def get_scenario_from_filename(filename, scenarios):
    for s in scenarios:
        if s in filename:
            return s


def get_rep_id(filepath):
    """Searches path for integer, uses as id for replicate."""
    for i in filepath.split("/"):
        try:
            int(i)
            return i
        except ValueError:
            continue


def write_preds(results_dict, outfile, benchmark):
    """
    Writes NN predictions to file.

    Args:
        results_dict (dict): SNP NN prediction scores and window edges.
        outfile (str): File to write results to.
    """
    lab_dict = {0: "Neut", 1: "SSV", 2: "SDN"}
    if benchmark:
        chroms, bps, mut_type, true_sel_coeff = zip(*results_dict.keys())
    else:
        chroms, bps = zip(*results_dict.keys())

    neut_scores = [i[0][0] for i in results_dict.values()]
    sdn_scores = [i[0][1] for i in results_dict.values()]
    ssv_scores = [i[0][2] for i in results_dict.values()]
    sel_preds = [i[1] for i in results_dict.values()]
    left_edges = [i[2] for i in results_dict.values()]
    right_edges = [i[3] for i in results_dict.values()]
    classes = [lab_dict[np.argmax(i[0])] for i in results_dict.values()]

    if benchmark:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Mut_Type": mut_type,
                "Class": classes,
                "Sweep_Score": [i + j for i, j in zip(sdn_scores, ssv_scores)],
                "True_Sel_Coeff": true_sel_coeff,
                "Sel_Coeff": sel_preds,
                "Neut_Score": neut_scores,
                "SDN_Score": sdn_scores,
                "SSV_Score": ssv_scores,
                "Win_Start": left_edges,
                "Win_End": right_edges,
            }
        )
    else:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Class": classes,
                "Sweep_Score": [i + j for i, j in zip(sdn_scores, ssv_scores)],
                "Sel_Coeff": sel_preds,
                "Neut_Score": neut_scores,
                "SDN_Score": sdn_scores,
                "SSV_Score": ssv_scores,
                "Win_Start": left_edges,
                "Win_End": right_edges,
            }
        )
        predictions = predictions[predictions["Neut_Score"] < 0.5]

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    if not os.path.exists(outfile):
        predictions.to_csv(outfile, header=True, index=False, sep="\t")
    else:
        predictions.to_csv(outfile, mode="a", header=False, index=False, sep="\t")

    bed_df = predictions[["Chrom", "Win_Start", "Win_End", "BP"]]
    bed_df.to_csv(
        outfile.replace(".csv", ".bed"), mode="a", header=False, index=False, sep="\t"
    )


def get_logger(module_name):
    logging.basicConfig()
    logger = logging.getLogger(module_name)
    logger.setLevel("INFO")

    return logger
