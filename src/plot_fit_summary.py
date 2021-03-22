import os
import random as rd
from math import floor
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import plotting_utils as pu


def import_data(fit_csv: str, cnn_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fit_df = pd.read_csv(fit_csv, header=0)
    cnn_df = pd.read_csv(cnn_csv, header=0)

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
    cnn_balanced = pd.concat([cnn_neut, cnn_df.loc[cnn_soft_inds], cnn_df.loc[cnn_hard_inds]], axis=0)
    fit_balanced = pd.concat([fit_neut, fit_df.loc[fit_soft_inds], fit_df.loc[fit_hard_inds]], axis=0)
    # fmt: on

    return cnn_balanced, fit_balanced


def plot_conf_mats(cnn_df: pd.DataFrame, fit_df: pd.DataFrame, save_dir: str) -> None:
    # Conf mats
    conf_mat = pu.print_confusion_matrix(cnn_df["combo_true"], cnn_df["combo_pred"])
    pu.plot_confusion_matrix(
        save_dir,
        conf_mat,
        ["No Sweep", "Sweep"],
        title="CNN Combined Predictions",
        normalize=True,
    )

    conf_mat = pu.print_confusion_matrix(fit_df["combo_true"], fit_df["min_p_detect"])
    pu.plot_confusion_matrix(
        save_dir,
        conf_mat,
        ["No Sweep", "Sweep"],
        title="FIt Combined Predictions",
        normalize=True,
    )


def plot_roc(cnn_df: pd.DataFrame, fit_df: pd.DataFrame, save_dir: str) -> None:
    # ROC
    plt.figure()
    plt.title("FIt ROC - All Types")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Postitive Rate")

    fpr, tpr, thresh = roc_curve(cnn_df["combo_true"], cnn_df["wombocombo"])
    auc = roc_auc_score(cnn_df["combo_true"], cnn_df["wombocombo"])
    plt.plot(fpr, tpr, label="CNN Preds, auc=" + "%.2f" % auc)

    fpr, tpr, thresh = roc_curve(fit_df["combo_true"], fit_df["inverted_p"])
    auc = roc_auc_score(fit_df["combo_true"], fit_df["inverted_p"])
    plt.plot(fpr, tpr, label="Min p val FIt, auc=" + "%.2f" % auc)

    plt.legend(loc="lower right")

    plt.savefig(os.path.join(save_dir, "fits_rocs.png"))


def main():

    sample_dict_path = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/sample_dict.csv"
    cnn_preds_path = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/TimeSweeperHaps_predictions.csv"
    cnn_df, fit_df = import_data(sample_dict_path, cnn_preds_path)
    plot_conf_mats(
        cnn_df, fit_df, os.path.join(os.path.basename(sample_dict_path), "images")
    )
    plot_roc(cnn_df, fit_df, os.path.join(os.path.basename(sample_dict_path), "images"))


if __name__ == "__main__":
    main()
