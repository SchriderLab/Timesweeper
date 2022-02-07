import argparse
import os
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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
    merged = pd.concat(all_results, ignore_index=True)
    merged.groupby(["Chrom", "BP"]).mean()
    if "fit" not in csvfiles[0]:
        merged["Sweep Score"] = merged["Hard Score"] + merged["Soft Score"]

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
    for i, sweep in enumerate(["neut", "hard", "soft"]):
        afs = preds_dict[sweep]["afs"]
        hfs = preds_dict[sweep]["hfs"]
        fit = preds_dict[sweep]["fit"]

        axs[0, i].violinplot(
            [
                afs[afs["Bin"] == bp_bin]["Sweep Score"].dropna()
                for bp_bin in afs["Bin"].unique()
            ]
        )

        axs[1, i].violinplot(
            [
                hfs[hfs["Bin"] == bp_bin]["Sweep Score"].dropna()
                for bp_bin in hfs["Bin"].unique()
            ]
        )

        axs[2, i].violinplot(
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
    axs[0, 1].set_title("Hard")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFS Score")
    axs[1, 0].set_ylabel("HFS Score")
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
    for i, sweep in enumerate(["neut", "hard", "soft"]):
        afs = preds_dict[sweep]["afs"]
        hfs = preds_dict[sweep]["hfs"]
        fit = preds_dict[sweep]["fit"]

        afs.boxplot(column=["Sweep Score"], by="Bin", ax=axs[0, i])
        hfs.boxplot(column=["Sweep Score"], by="Bin", ax=axs[1, i])
        fit.boxplot(column=["Inv pval"], by="Bin", ax=axs[2, i])

        for j in range(3):
            axs[i, j].tick_params(axis="x", rotation=45)

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 1].set_title("Hard")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFS Score")
    axs[1, 0].set_ylabel("HFS Score")
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
        afs = preds_dict[sweep]["afs"]
        hfs = preds_dict[sweep]["hfs"]
        fit = preds_dict[sweep]["fit"]

        afs_mean = afs.groupby("Bin")["Sweep Score"].mean()
        hfs_mean = hfs.groupby("Bin")["Sweep Score"].mean()
        fit_mean = fit.groupby("Bin")["Inv pval"].mean()

        axs[i].plot(afs_mean, label="AFS")
        axs[i].plot(hfs_mean, label="HFS")
        axs[i].plot(fit_mean, label="FIT")

        axs[i].set_xticks(afs["Bin"].unique().tolist())
        axs[i].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))
        axs[i].tick_params(axis="x", rotation=45)

    axs[0].set_title("Neut")
    axs[1].set_title("Hard")
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
    for i, sweep in enumerate(["neut", "hard", "soft"]):
        afs = preds_dict[sweep]["afs"]
        hfs = preds_dict[sweep]["hfs"]
        fit = preds_dict[sweep]["fit"].dropna()

        # Pandas is a nightmare so I'm doing it manually

        # Iterate through each prediction class, populate line
        # Separated out because Neut and Hard/Soft have 1 diff X tick (central locus)
        afs_bin_sizes = []
        for bin_lab in afs["Bin"].unique():
            subdf = afs[afs["Bin"] == bin_lab]
            afs_bin_sizes.append(len(subdf))

        afs_bin_sizes = np.array(afs_bin_sizes)

        for subsweep in ["Neut", "Hard", "Soft"]:
            class_sizes = []
            for bin_lab in afs["Bin"].unique():
                subdf = afs[(afs["Bin"] == bin_lab) & (afs["Class"] == subsweep)]
                class_sizes.append(len(subdf))

            afs_prop = class_sizes / afs_bin_sizes
            axs[0, i].plot(afs["Bin"].unique(), afs_prop, label=f"AFS {subsweep}")

        hfs_bin_sizes = []
        for bin_lab in hfs["Bin"].unique():
            subdf = hfs[hfs["Bin"] == bin_lab]
            hfs_bin_sizes.append(len(subdf))

        hfs_bin_sizes = np.array(hfs_bin_sizes)

        for subsweep in ["Neut", "Hard", "Soft"]:
            class_sizes = []
            for bin_lab in hfs["Bin"].unique():
                subdf = hfs[(hfs["Bin"] == bin_lab) & (hfs["Class"] == subsweep)]
                class_sizes.append(len(subdf))

            hfs_prop = class_sizes / hfs_bin_sizes
            axs[1, i].plot(hfs["Bin"].unique(), hfs_prop, label=f"HFS {subsweep}")

        fit_bin_sizes = []
        for bin_lab in fit["Bin"].unique():
            subdf = fit[fit["Bin"] == bin_lab]
            fit_bin_sizes.append(len(subdf))

        fit_bin_sizes = np.array(fit_bin_sizes)
        sweep_sizes = []
        neut_sizes = []
        for bin_lab in fit["Bin"].unique():
            sweepdf = fit[(fit["Bin"] == bin_lab) & (fit["Inv pval"] > 0.95)]
            neutdf = fit[(fit["Bin"] == bin_lab) & (fit["Inv pval"] < 0.95)]
            sweep_sizes.append(len(sweepdf))
            neut_sizes.append(len(neutdf))

        fit_sweep_prop = sweep_sizes / fit_bin_sizes
        fit_neut_prop = neut_sizes / fit_bin_sizes
        axs[2, i].plot(fit["Bin"].unique(), fit_sweep_prop, label=f"FIT Sweep")
        axs[2, i].plot(fit["Bin"].unique(), fit_neut_prop, label=f"FIT Neut")

        for j in range(3):
            axs[i, j].set_xticks(afs["Bin"].unique().tolist())
            axs[i, j].tick_params(axis="x", rotation=45)
            axs[i, j].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 1].set_title("Hard")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFS Score")
    axs[1, 0].set_ylabel("HFS Score")
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
    if sweep in ["hard", "soft"]:
        trues[
            np.array(
                pred_df.index[(pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Hard")]
            )
        ] = 1
        trues[
            np.array(
                pred_df.index[(pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Soft")]
            )
        ] = 2

    preds = np.argmax(np.array(pred_df.loc[:, "Neut Score":"Soft Score"]), axis=1)

    print(f"{sweep}: {sum(trues)}")
    return trues, preds


def main():
    indir = "/proj/dschridelab/lswhiteh/timesweeper-experiments/simple_sims/vcf_sims/onePop-selectiveSweep-vcf.slim/"
    datadict = {}

    afs_trues = []
    afs_preds = []
    hfs_trues = []
    hfs_preds = []
    for sweep in ["neut", "hard", "soft"]:
        datadict[sweep] = {}
        csvs = glob(os.path.join(indir, sweep, "pops", "*", "*.csv"))
        print(f"{len(csvs)} files in {sweep}")

        for model in ["afs", "hfs", "fit"]:
            rawfiles = [i for i in csvs if model in i]
            preds = load_preds(rawfiles)
            binned = bin_preds(preds)
            datadict[sweep][model] = binned

            if model == "afs":
                trues, preds = get_ys(preds, sweep)
                afs_trues.extend(trues)
                afs_preds.extend(preds)

            elif model == "hfs":
                trues, preds = get_ys(preds, sweep)
                hfs_trues.extend(trues)
                hfs_preds.extend(preds)

    hfs_cm = confusion_matrix(hfs_trues, hfs_preds)
    plot_confusion_matrix(
        ".",
        hfs_cm,
        target_names=["Neut", "Hard", "Soft"],
        title="HFS_Confmat",
        normalize=True,
    )

    afs_cm = confusion_matrix(afs_trues, afs_preds)
    plot_confusion_matrix(
        ".",
        afs_cm,
        target_names=["Neut", "Hard", "Soft"],
        title="AFS_Confmat",
        normalize=True,
    )

    plot_boxplots(datadict)
    plot_violinplots(datadict)
    plot_means(datadict)
    plot_proportions(datadict)


if __name__ == "__main__":
    main()
