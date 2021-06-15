import os
import random as rd
from glob import glob
from math import floor
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import plotting_utils as pu


def import_data(cnn_csv: str) -> pd.DataFrame:
    cnn_df = pd.read_csv(cnn_csv, header=0)
    # print(cnn_df.head())

    # CNN results
    cnn_df["combo_true"] = 0
    cnn_df.loc[(cnn_df["true"] == 0) | (cnn_df["true"] == 2), "combo_true"] = 1
    cnn_df["wombocombo"] = cnn_df["prob_hard"] + cnn_df["prob_soft"]
    cnn_df["combo_pred"] = 0
    cnn_df.loc[(cnn_df["pred"] == 0) | (cnn_df["pred"] == 2), "combo_pred"] = 1

    cnn_neut = cnn_df[cnn_df["combo_true"] == 0]
    num_soft = len(cnn_df[cnn_df["true"] == 2])
    num_hard = len(cnn_df[cnn_df["true"] == 0])
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
        cnn_df.index[cnn_df["true"] == 0].tolist(), k=hard_to_sample
    )

    # fmt: off
    cnn_balanced = pd.concat([cnn_neut, cnn_df.loc[cnn_soft_inds], cnn_df.loc[cnn_hard_inds]], axis=0)
    # fmt: on

    return cnn_balanced


def plot_conf_mats(cnn_df: pd.DataFrame, save_dir: str, samplab, schema) -> None:
    # Conf mats
    conf_mat = pu.print_confusion_matrix(cnn_df["combo_true"], cnn_df["combo_pred"])

    pu.plot_confusion_matrix(
        save_dir,
        conf_mat,
        ["No Sweep", "Sweep"],
        title=f"{samplab}{schema}",
        normalize=True,
    )


def main():
    simType = "onePop-adaptiveIntrogression"
    pred_files = glob(
        f"/pine/scr/l/s/lswhiteh/timeSeriesSweeps/{simType}-*Samp-*Int/*predictions.csv"
    )
    # print(pred_files)

    # ROC
    plt.figure()
    plt.title(f"CNN ROC - {simType}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postitive Rate")

    for i in pred_files:
        print(i)
        cnn_df = import_data(i)
        # print(cnn_df.head())
        schema = "-".join(i.split("/")[-2].split("-")[-2:])
        if "1Samp" in i:
            samplab = "1Samp-"
        else:
            samplab = ""

        fpr, tpr, thresh = roc_curve(cnn_df["combo_true"], cnn_df["wombocombo"])
        auc = roc_auc_score(cnn_df["combo_true"], cnn_df["wombocombo"])
        plt.plot(fpr, tpr, label=f"{samplab}{schema}, auc=" + "%.2f" % auc)

    plt.legend(loc="lower right")
    plt.savefig(f"{simType}-fits_rocs.png")


if __name__ == "__main__":
    main()
