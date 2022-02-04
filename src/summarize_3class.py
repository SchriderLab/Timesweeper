import os
import random as rd
from glob import glob
from math import floor
from typing import Tuple
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

import plotting.plotting_utils as pu


def import_data(cnn_csv: str) -> pd.DataFrame:
    cnn_df = pd.read_csv(cnn_csv, header=0)
    # print(cnn_df.head())

    # CNN results
    cnn_df["combo_true"] = 0
    cnn_df.loc[(cnn_df["true"] == 1) | (cnn_df["true"] == 2), "combo_true"] = 1
    cnn_df["wombocombo"] = cnn_df["prob_hard"] + cnn_df["prob_soft"]
    cnn_df["combo_pred"] = 0
    cnn_df.loc[(cnn_df["pred"] == 1) | (cnn_df["pred"] == 2), "combo_pred"] = 1

    cnn_neut = cnn_df[cnn_df["combo_true"] == 0]
    num_soft = len(cnn_df[cnn_df["true"] == 2])
    num_hard = len(cnn_df[cnn_df["true"] == 1])
    num_neut = len(cnn_neut)

    # Exception handling for mismatched, why does this happen?
    if floor(num_neut / 2) > num_soft:
        soft_to_sample = num_soft
    else:
        soft_to_sample = floor(num_neut / 2)

    if floor(num_neut / 2) > num_hard:
        hard_to_sample = num_hard
    else:
        hard_to_sample = floor(num_neut / 2)

    cnn_soft_inds = rd.sample(
        cnn_df.index[cnn_df["true"] == 2].tolist(), k=soft_to_sample
    )
    cnn_hard_inds = rd.sample(
        cnn_df.index[cnn_df["true"] == 1].tolist(), k=hard_to_sample
    )

    # fmt: off
    cnn_balanced = pd.concat([cnn_neut, cnn_df.loc[cnn_soft_inds], cnn_df.loc[cnn_hard_inds]], axis=0)
    # fmt: on

    return cnn_balanced


def plot_conf_mats(cnn_df: pd.DataFrame, save_dir: str, samplab, schema) -> None:
    # Conf mats
    conf_mat = confusion_matrix(cnn_df["combo_true"], cnn_df["combo_pred"])
    pu.plot_confusion_matrix(
        save_dir,
        conf_mat,
        ["No Sweep", "Sweep"],
        title=f"summarized-{samplab}{schema}-confmat",
        normalize=True,
    )


def main():
    save_dir = sys.argv[1]
    plot_title = sys.argv[2]
    pred_files = sys.argv[3:]
    # pred_files = [i for i in sys.argv[3:] if "1Samp" not in i]
    # pred_files = [i for i in sys.argv[3:] if "1Samp" in i]
    # print(pred_files)
    # pred_files = [i for i in pred_files if "hap" in i]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fprs, tprs, threshs, aucs, samplabs, schema = [], [], [], [], [], []
    for i in pred_files:
        print(i)
        cnn_df = import_data(i)
        if "hap" in i:
            samplab = "Haps"
        elif "freq" in i:
            samplab = "Freqs"
        else:
            samplab = ""

        schema.append(i.split("/")[-1].split(".")[0])

        # Don't want to print out conf mats again if getting ROCs
        if len(pred_files) < 3:
            plot_conf_mats(cnn_df, f"{save_dir}/images", samplab, schema[-1])

        fpr, tpr, thresh = roc_curve(cnn_df["combo_true"], cnn_df["wombocombo"])
        auc = roc_auc_score(cnn_df["combo_true"], cnn_df["wombocombo"])

        fprs.append(fpr)
        tprs.append(tpr)
        threshs.append(thresh)
        aucs.append(auc)
        samplabs.append(samplab)

    aucs, fprs, tprs, threshs, samplabs, schema = zip(
        *sorted(zip(aucs, fprs, tprs, threshs, samplabs, schema), reverse=True)
    )

    plt.clf()
    # ROC
    plt.figure()
    plt.title(f"CNN ROC - {plot_title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postitive Rate")
    for i in range(len(fprs)):
        plt.plot(
            fprs[i],
            tprs[i],
            label=f"{samplabs[i]}-{schema[i]}, auc=".replace("_", "")
            + f"{aucs[i]:.2f}",
        )
        plt.title(f"ROC Curves - {plot_title}")

    plt.legend(loc="lower right", prop={"size": 6})
    plt.savefig(f"{save_dir}/{plot_title.replace(' ', '_')}-rocs.png")


if __name__ == "__main__":
    main()
