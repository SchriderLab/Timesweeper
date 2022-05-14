import yaml
import pandas as pd
import os


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


def get_sweep(filepath):
    """Grabs the sweep label from filepaths for easy saving."""
    for sweep in ["neut", "hard", "soft"]:
        if sweep in filepath:
            return sweep


def get_rep_id(filepath):
    """Searches path for integer, uses as id for replicate."""
    for i in filepath.split("/"):
        try:
            int(i)
            return i
        except ValueError:
            continue


def write_fit(fit_dict, outfile, benchmark):
    """
    Writes FIT predictions to file.

    Args:
        fit_dict (dict): FIT p values and SNP information.
        outfile (str): File to write results to.
    """
    inv_pval = [1 - i[1] for i in fit_dict.values()]
    if benchmark:
        chroms, bps, mut_type = zip(*fit_dict.keys())
        predictions = pd.DataFrame(
            {"Chrom": chroms, "BP": bps, "Mut Type": mut_type, "Inv pval": inv_pval}
        )
    else:
        chroms, bps = zip(*fit_dict.keys())
        predictions = pd.DataFrame({"Chrom": chroms, "BP": bps, "Inv pval": inv_pval})

    predictions.sort_values(["Chrom", "BP"], inplace=True)
    predictions.to_csv(
        os.path.join(outfile), header=True, index=False, sep="\t",
    )


def write_preds(results_dict, outfile, benchmark):
    """
    Writes NN predictions to file.

    Args:
        results_dict (dict): SNP NN prediction scores and window edges.
        outfile (str): File to write results to.
    """
    lab_dict = {0: "Neut", 1: "Hard", 2: "Soft"}
    if benchmark:
        chroms, bps, mut_type = zip(*results_dict.keys())
    else:
        chroms, bps = zip(*results_dict.keys())

    neut_scores = [i[0][0] for i in results_dict.values()]
    hard_scores = [i[0][1] for i in results_dict.values()]
    soft_scores = [i[0][2] for i in results_dict.values()]
    left_edges = [i[1] for i in results_dict.values()]
    right_edges = [i[2] for i in results_dict.values()]
    classes = [lab_dict[np.argmax(i[0])] for i in results_dict.values()]

    if benchmark:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Mut Type": mut_type,
                "Class": classes,
                "Neut Score": neut_scores,
                "Hard Score": hard_scores,
                "Soft Score": soft_scores,
                "Win Start": left_edges,
                "Win End": right_edges,
            }
        )
    else:
        predictions = pd.DataFrame(
            {
                "Chrom": chroms,
                "BP": bps,
                "Class": classes,
                "Neut_Score": neut_scores,
                "Hard_Score": hard_scores,
                "Soft_Score": soft_scores,
                "Win_Start": left_edges,
                "Win_End": right_edges,
            }
        )
        predictions = predictions[predictions["Neut Score"] < 0.5]

    predictions.sort_values(["Chrom", "BP"], inplace=True)

    if not os.path.exists(outfile):
        predictions.to_csv(outfile, header=True, index=False, sep="\t")
    else:
        predictions.to_csv(outfile, mode="a", header=False, index=False, sep="\t")

    bed_df = predictions[["Chrom", "Win_Start", "Win_End", "BP"]]
    bed_df.to_csv(
        outfile.replace(".csv", ".bed"), mode="a", header=False, index=False, sep="\t"
    )
