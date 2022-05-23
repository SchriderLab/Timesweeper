import os
from glob import glob
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from plotting_utils import plot_confusion_matrix

mpl.rcParams["agg.path.chunksize"] = 10000


def load_preds(csvfiles):
    """
    Reads in list of csvs as pd DataFrames, adds combined score to non-FIT preds.

    Args:
        csvfiles (list[str]): Paths of prediction csvs.

    Returns:
        pd.DataFrame: Dataframe containing predictions at each SNP from a series of VCFs.
    """
    all_results = [
        pd.read_csv(i, sep="\t", header=0) for i in tqdm(csvfiles, desc="Loading Files")
    ]
    print(all_results)
    if len(all_results) > 1:
        merged = pd.concat(all_results, ignore_index=True)
    else:
        merged = all_results[0]

    merged.groupby(["Chrom", "BP"]).mean()
    if "fit" not in csvfiles[0]:
        merged["Sweep Score"] = merged["Soft Score"]

    return merged


def bin_preds(merged_scores):
    """
    Bins predictions into 21 windows of SNPs for easier visualization.

    Args:
        merged_scores (pd.DataFrame): Prediction scores from all samples pooled.

    Returns:
        pd.DataFrame: Dataframe that now has a "Bin" column for easy groupby methods.
    """
    binned_dfs = []
    for chrom in merged_scores["Chrom"].unique():
        _df = merged_scores[merged_scores["Chrom"] == chrom]
        cut_bins = np.linspace(_df["BP"].min() - 1, _df["BP"].max(), 22)
        _df["Bin"] = pd.cut(_df["BP"], bins=cut_bins, labels=cut_bins[:-1]).astype(int)
        binned_dfs.append(_df)
        _df.loc[_df["Mut Type"] == 2, "Bin"] = 250000

    return pd.concat(binned_dfs, axis=0)


def plot_violinplots(preds_dict):
    """
    Plots violin plot in 3x3 subplot layout.

    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Simple Simulations Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "soft"]):
        aft = preds_dict[sweep]["aft"]
        fit = preds_dict[sweep]["fit"]

        axs[0, i].violinplot(
            [
                aft[aft["Bin"] == bp_bin]["Sweep Score"].dropna()
                for bp_bin in aft["Bin"].unique()
            ]
        )

        axs[1, i].violinplot(
            [
                fit[fit["Bin"] == bp_bin]["Inv pval"].dropna()
                for bp_bin in fit["Bin"].unique()
            ]
        )

        for j in range(3):
            axs[i, j].tick_params(axis="x", rotation=45)

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("aft Score")
    axs[2, 0].set_ylabel("FIT Inv pval")

    plt.savefig(f"violinplot.png")


def plot_boxplots(preds_dict):
    """
    Plots box and whisker plots in 3x3 subplot layout.

    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Simple Simulations Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "soft"]):
        aft = preds_dict[sweep]["aft"]
        fit = preds_dict[sweep]["fit"]

        aft.boxplot(column=["Sweep Score"], by="Bin", ax=axs[0, i])
        fit.boxplot(column=["Inv pval"], by="Bin", ax=axs[1, i])

        for j in range(3):
            axs[i, j].tick_params(axis="x", rotation=45)

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFT Score")
    axs[2, 0].set_ylabel("FIT Inv pval")

    plt.savefig(f"boxplot.png")


def plot_means(preds_dict):
    """
    Plots line plots of mean values in 1x3 subplot layout.

    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (15, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("Simple Simulations Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "hard", "soft"]):
        aft = preds_dict[sweep]["aft"]
        fit = preds_dict[sweep]["fit"]

        aft_mean = aft.groupby("Bin")["Sweep Score"].mean()
        fit_mean = fit.groupby("Bin")["Inv pval"].mean()

        axs[i].plot(aft_mean, label="aft")
        axs[i].plot(fit_mean, label="FIT")

        axs[i].set_xticks(aft["Bin"].unique().tolist())
        axs[i].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))
        axs[i].tick_params(axis="x", rotation=45)

    axs[0].set_title("Neut")
    axs[2].set_title("Soft")
    axs[0].legend()

    plt.savefig("meanline.png")


def plot_proportions(preds_dict):
    """
    Plots lines of proportions of each class over genomic window.

    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (30, 15)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Proportion of Class Predictions Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "soft"]):
        aft = preds_dict[sweep]["aft"]
        fit = preds_dict[sweep]["fit"].dropna()

        # Pandas is a nightmare so I'm doing it manually

        # Iterate through each prediction class, populate line
        # Separated out because Neut and Hard/Soft have 1 diff X tick (central locus)
        aft_bin_sizes = []
        for bin_lab in aft["Bin"].unique():
            subdf = aft[aft["Bin"] == bin_lab]
            aft_bin_sizes.append(len(subdf))

        aft_bin_sizes = np.array(aft_bin_sizes)

        for subsweep in ["Neut", "Soft"]:
            class_sizes = []
            for bin_lab in aft["Bin"].unique():
                subdf = aft[(aft["Bin"] == bin_lab) & (aft["Class"] == subsweep)]
                class_sizes.append(len(subdf))

            aft_prop = class_sizes / aft_bin_sizes
            axs[0, i].plot(aft["Bin"].unique(), aft_prop, label=f"AFT {subsweep}")

        fit_bin_sizes = []
        for bin_lab in fit["Bin"].unique():
            subdf = fit[fit["Bin"] == bin_lab]
            fit_bin_sizes.append(len(subdf))

        fit_bin_sizes = np.array(fit_bin_sizes)
        sweep_sizes = []
        neut_sizes = []
        for bin_lab in fit["Bin"].unique():
            sweepdf = fit[(fit["Bin"] == bin_lab) & (fit["Inv_pval"] > 0.95)]
            neutdf = fit[(fit["Bin"] == bin_lab) & (fit["Inv_pval"] < 0.95)]
            sweep_sizes.append(len(sweepdf))
            neut_sizes.append(len(neutdf))

        fit_sweep_prop = sweep_sizes / fit_bin_sizes
        fit_neut_prop = neut_sizes / fit_bin_sizes
        axs[1, i].plot(fit["Bin"].unique(), fit_sweep_prop, label=f"FIT Sweep")
        axs[1, i].plot(fit["Bin"].unique(), fit_neut_prop, label=f"FIT Neut")

        for j in range(3):
            axs[i, j].set_xticks(aft["Bin"].unique().tolist())
            axs[i, j].tick_params(axis="x", rotation=45)
            axs[i, j].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFT Score")
    axs[2, 0].set_ylabel("FIT Inv pval")

    axs[0, 0].legend()
    axs[1, 0].legend()
    axs[2, 0].legend()

    plt.savefig("propline.png")


def get_ys(pred_df, sweep):
    """
    Grabs true labels from all data and gives numerical label.

    Args:
        pred_df (pd.DataFrame): Predictions from samples.
        sweep (str): Whether sweep is present, hard, soft, to decide which numerical value the sample is labeled with.

    Returns:
        np.arr: Trues and predictions labels in [0,1,2].
    """
    trues = np.zeros(len(pred_df))
    if sweep in ["soft"]:
        # trues[
        #    np.array(
        #        pred_df.index[(pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Hard")]
        #    )
        # ] = 1
        trues[
            np.array(
                pred_df.index[(pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Soft")]
            )
        ] = 1

    preds = np.argmax(np.array(pred_df.loc[:, "Neut Score":"Soft Score"]), axis=1)

    print(f"{sweep}: {sum(trues)}")
    return trues, preds


def main():
    indir = sys.argv[1]
    datadict = {}

    aft_trues = []
    aft_preds = []

    for sweep in ["neut", "soft"]:
        datadict[sweep] = {}
        csvs = glob(os.path.join(indir, "*.csv"))
        print(f"{len(csvs)} files in {sweep}")

        for model in ["aft", "fit"]:
            rawfiles = [i for i in csvs if model in i]
            preds = load_preds(rawfiles)
            binned = bin_preds(preds)
            datadict[sweep][model] = binned

            if model == "aft":
                trues, preds = get_ys(preds, sweep)
                aft_trues.extend(trues)
                aft_preds.extend(preds)

    aft_cm = confusion_matrix(aft_trues, aft_preds)
    plot_confusion_matrix(
        ".", aft_cm, target_names=["Neut", "Soft"], title="aft_Confmat", normalize=True,
    )

    plot_boxplots(datadict)
    plot_violinplots(datadict)
    plot_means(datadict)
    plot_proportions(datadict)


if __name__ == "__main__":
    main()
