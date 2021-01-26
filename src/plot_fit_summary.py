import matplotlib.pyplot as plt
import pandas as pd
import random as rd
from sklearn.metrics import roc_curve, roc_auc_score
import plotting_utils as pu
from math import floor
from typing import Tuple


def import_data(
    fit_csv: str = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/sample_dict.csv",
    cnn_csv: str = "/pine/scr/l/s/lswhiteh/timeSeriesSweeps/onePop-selectiveSweep-20Samp-10Int/TimeSweeperSHIC_predictions.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fit_df = pd.read_csv(fit_csv, header=0)

    fit_df = fit_df[fit_df["window"] == 5]
    fit_df["combo_true"] = 0
    fit_df.loc[
        (fit_df["true_site_soft"] == 1) | (fit_df["true_site_hard"] == 1), "combo_true"
    ] = 1
    fit_df["inverted_p"] = 1 - fit_df["min_p_val"]
    fit_df["combo_pred"] = 0
    fit_df.loc[
        (fit_df["min_p_soft"] == 1) | (fit_df["min_p_hard"] == 1), "combo_pred"
    ] = 1

    cnn_df = pd.read_csv(cnn_csv, header=0)

    cnn_df["combo_true"] = 0
    cnn_df.loc[(cnn_df["true"] == 0) | (cnn_df["true"] == 2), "combo_true"] = 1
    cnn_df["wombocombo"] = cnn_df["prob_hard"] + cnn_df["prob_soft"]
    cnn_df["combo_pred"] = 0
    cnn_df.loc[(cnn_df["pred"] == 0) | (cnn_df["pred"] == 2), "combo_pred"] = 1

    num_neut = len([i for i in fit_df["combo_true"] if i == 0])
    cnn_neut = cnn_df[cnn_df["true"] == 1]
    cnn_soft_inds = rd.sample(
        cnn_df.index[cnn_df["true"] == 2].tolist(), k=floor(num_neut / 2)
    )
    cnn_hard_inds = rd.sample(
        cnn_df.index[cnn_df["true"] == 0].tolist(), k=floor(num_neut / 2)
    )

    fit_neut = fit_df[(fit_df["true_site_soft"] == 0) & (fit_df["true_site_hard"] == 0)]
    fit_soft_inds = rd.sample(
        fit_df.index[fit_df["true_site_soft"] == 1].tolist(), k=floor(num_neut / 2)
    )
    fit_hard_inds = rd.sample(
        fit_df.index[fit_df["true_site_hard"] == 1].tolist(), k=floor(num_neut / 2)
    )

    # fmt: off
    cnn_balanced = pd.concat([cnn_neut, cnn_df.loc[cnn_soft_inds], cnn_df.loc[cnn_hard_inds]], axis=0)
    fit_balanced = pd.concat([fit_neut, fit_df.loc[fit_soft_inds], fit_df.loc[fit_hard_inds]], axis=0)
    # fmt: on

    return cnn_balanced, fit_balanced


def plot_conf_mats(cnn_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
    # Conf mats
    conf_mat = pu.print_confusion_matrix(cnn_df["combo_true"], cnn_df["combo_pred"])
    pu.plot_confusion_matrix(
        ".", conf_mat, ["No Sweep", "Sweep"], title="CNN Combined Predictions"
    )

    conf_mat = pu.print_confusion_matrix(fit_df["combo_true"], fit_df["combo_pred"])
    pu.plot_confusion_matrix(
        ".", conf_mat, ["No Sweep", "Sweep"], title="FIt Combined Predictions"
    )


def plot_roc(cnn_df: pd.DataFrame, fit_df: pd.DataFrame) -> None:
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

    plt.savefig("fits_rocs.png")


def main():
    cnn_df, fit_df = import_data()
    plot_conf_mats(cnn_df, fit_df)
    plot_roc(cnn_df, fit_df)


if __name__ == "__main__":
    main()