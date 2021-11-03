import os, sys
import random as rd
from math import floor
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


import plotting_utils as pu

df = pd.DataFrame


def import_data(fit_csv: str) -> df:
    fit_df = pd.read_csv(fit_csv, header=0)

    # FIt Results
    fit_df = fit_df[fit_df["window"] == 5]
    fit_df["combo_true"] = 0
    fit_df.loc[
        (fit_df["true_site_soft"] == 1) | (fit_df["true_site_hard"] == 1), "combo_true"
    ] = 1
    fit_df["inverted_p"] = 1 - fit_df["min_p_val"]

    fit_neut = fit_df[(fit_df["true_site_soft"] == 0) & (fit_df["true_site_hard"] == 0)]

    num_soft = len(fit_df[fit_df["true_site_soft"] == 1])
    num_hard = len(fit_df[fit_df["true_site_hard"] == 1])
    num_neut = len(fit_neut)

    if floor(num_neut / 2) > num_soft:
        soft_to_sample = num_soft
    else:
        soft_to_sample = floor(num_neut / 2)

    if floor(num_neut / 2) > num_hard:
        hard_to_sample = num_hard
    else:
        hard_to_sample = floor(num_neut / 2)

    print(floor(len(fit_neut) / 2))
    print(len(fit_df[fit_df["true_site_soft"] == 1]))

    fit_soft_inds = rd.sample(
        fit_df.index[fit_df["true_site_soft"] == 1].tolist(), k=soft_to_sample
    )
    fit_hard_inds = rd.sample(
        fit_df.index[fit_df["true_site_hard"] == 1].tolist(), k=hard_to_sample
    )

    # fmt: off
    fit_balanced = pd.concat([fit_neut, fit_df.loc[fit_soft_inds], fit_df.loc[fit_hard_inds]], axis=0)
    # fmt: on

    return fit_balanced


def plot_conf_mat(fit_df: df, save_dir: str) -> None:
    # Conf mats
    conf_mat = pu.get_confusion_matrix(fit_df["combo_true"], fit_df["min_p_detect"])
    pu.plot_confusion_matrix(
        save_dir,
        conf_mat,
        ["No Sweep", "Sweep"],
        title="FIt_Combined_Predictions",
        normalize=True,
    )


def plot_roc(fit_df: df, save_dir: str) -> None:
    # ROC
    plt.figure()
    plt.title("FIt ROC - All Types")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postitive Rate")

    fpr, tpr, thresh = roc_curve(fit_df["combo_true"], fit_df["inverted_p"])
    auc = roc_auc_score(fit_df["combo_true"], fit_df["inverted_p"])
    plt.plot(fpr, tpr, label="Min p val FIt, auc=" + f"{auc:.2f}")

    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_dir, "fits_rocs.png"))


def main():
    # Requires summarize_fit to run first
    fit_summary = sys.argv[1]
    fit_df = import_data(fit_summary)
    plot_conf_mat(fit_df, os.path.join(os.path.dirname(fit_summary)))
    plot_roc(fit_df, os.path.join(os.path.dirname(fit_summary)))


if __name__ == "__main__":
    main()
