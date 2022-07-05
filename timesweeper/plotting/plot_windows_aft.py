import os
from glob import glob
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from plotting_utils import plot_confusion_matrix, plot_roc, plot_prec_recall

mpl.rcParams["agg.path.chunksize"] = 10000


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
    for i, sweep in enumerate(list(preds_dict.keys())):
        aft = preds_dict[sweep]["aft"]
        # hft = preds_dict[sweep]["hft"]
        # fit = preds_dict[sweep]["fit"]

        axs[0, 0].set_title(f"{sweep.upper()}")
        axs[0, 1].set_title(f"{sweep.upper()}")
        axs[0, 2].set_title(f"{sweep.upper()}")

        axs[0, i].violinplot(
            [
                aft[aft["Bin"] == bp_bin]["Sweep_Score"].dropna()
                for bp_bin in aft["Bin"].unique()
            ]
        )
        """
        axs[1, i].violinplot(
            [
                hft[hft["Bin"] == bp_bin]["Sweep_Score"].dropna()
                for bp_bin in hft["Bin"].unique()
            ]
        )

        axs[2, i].violinplot(
            [
                fit[fit["Bin"] == bp_bin]["Inv_pval"].dropna()
                for bp_bin in fit["Bin"].unique()
            ]
        )
        """

        for j in range(3):
            axs[i, j].tick_params(axis="x", rotation=45)

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")
            axs[i, j].set_xticks(aft["Bin"].unique().tolist())

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 1].set_title("Soft")
    axs[0, 2].set_title("Hard")

    axs[0, 0].set_ylabel("AFT Score")
    # axs[1, 0].set_ylabel("HFT Score")
    # axs[2, 0].set_ylabel("FIT Inv pval")

    plt.savefig(f"violinplot.pdf")


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
    for i, sweep in enumerate(list(preds_dict.keys())):
        aft = preds_dict[sweep]["aft"]
        # hft = preds_dict[sweep]["hft"]
        # fit = preds_dict[sweep]["fit"]

        aft.boxplot(column=["Sweep_Score"], by="Bin", ax=axs[0, i])
        # hft.boxplot(column=["Sweep_Score"], by="Bin", ax=axs[1, i])
        # fit.boxplot(column=["Inv_pval"], by="Bin", ax=axs[2, i])

        axs[0, i].tick_params(axis="x", rotation=90)
        axs[0, i].set_title(f"{sweep.upper()}")

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")

        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_ylabel("AFT Score")
    # axs[1, 0].set_ylabel("HFT Score")
    # axs[2, 0].set_ylabel("FIT Inv pval")

    plt.savefig(f"boxplot.pdf")


def plot_means(preds_dict):
    """
    Plots line plots of mean values in 1x3 subplot layout.
    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (20, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("Mean Sweep Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(list(preds_dict.keys())):
        aft = preds_dict[sweep]["aft"]
        # hft = preds_dict[sweep]["hft"]
        # fit = preds_dict[sweep]["fit"]

        aft_mean = aft.groupby("Bin")["Sweep_Score"].mean()
        # hft_mean = hft.groupby("Bin")["Sweep_Score"].mean()
        # fit_mean = fit.groupby("Bin")["Inv_pval"].mean()

        axs[i].plot(aft_mean, label=f"AFT {sweep.upper()}")
        # axs[i].plot(hft_mean, label="HFT")
        # axs[i].plot(fit_mean, label="FIT")

        loclist = aft["Bin"].unique().tolist()
        axs[i].set_xticks([loclist[0], 250000, loclist[-1]])

        # axs[i].set_xticklabels([loclist[0], 250000, loclist[-1]])
        axs[i].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))
        axs[i].tick_params(axis="x", rotation=45)

    axs[0].set_title("Mean Score per Bin")
    axs[1].set_title("foo")
    axs[2].set_title("bar")
    axs[0].legend()

    plt.savefig("meanline.pdf")


def plot_maxes(preds_dict):
    """
    Plots line plots of max values in 1x3 subplot layout.
    Args:
        preds_dict (pd.DataFrame): Predictions from all samples.
    """
    plt.clf()
    plt.rcParams["figure.figsize"] = (15, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("Simple Simulations Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "soft", "hard"]):
        aft = preds_dict[sweep]["aft"]
        # hft = preds_dict[sweep]["hft"]
        # fit = preds_dict[sweep]["fit"]

        aft_max = aft.groupby("Bin")["Sweep_Score"].max()
        # hft_max = hft.groupby("Bin")["Sweep_Score"].max()
        # fit_max = fit.groupby("Bin")["Inv_pval"].max()

        axs[i].plot(aft_max, label=f"AFT {sweep.upper()}")
        # axs[i].plot(hft_max, label="HFT")
        # axs[i].plot(fit_max, label="FIT")

        loclist = aft["Bin"].unique().tolist()
        axs[i].set_xticks([loclist[0], 250000, loclist[-1]])

        axs[i].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))
        axs[i].tick_params(axis="x", rotation=45)

    axs[0].set_title("Neut")
    axs[1].set_title("SSV")
    axs[2].set_title("SDN")
    axs[0].legend()

    plt.savefig("maxline.pdf")


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
        aft = preds_dict[sweep]["aft"]
        # hft = preds_dict[sweep]["hft"]
        # fit = preds_dict[sweep]["fit"].dropna()

        # Pandas is a nightmare so I'm doing it manually

        # Iterate through each prediction class, populate line
        # Separated out because Neut and Hard/Soft have 1 diff X tick (central locus)
        aft_bin_sizes = []
        for bin_lab in aft["Bin"].unique():
            subdf = aft[aft["Bin"] == bin_lab]
            aft_bin_sizes.append(len(subdf))

        aft_bin_sizes = np.array(aft_bin_sizes)

        for subsweep in ["Neut", "Soft", "Hard"]:
            class_sizes = []
            for bin_lab in aft["Bin"].unique():
                subdf = aft[(aft["Bin"] == bin_lab) & (aft["Class"] == subsweep)]
                class_sizes.append(len(subdf))

            aft_prop = class_sizes / aft_bin_sizes
            axs[0, i].plot(aft["Bin"].unique(), aft_prop, label=f"AFT {subsweep}")

        """
        hft_bin_sizes = []
        for bin_lab in hft["Bin"].unique():
            subdf = hft[hft["Bin"] == bin_lab]
            hft_bin_sizes.append(len(subdf))

        hft_bin_sizes = np.array(hft_bin_sizes)

        for subsweep in ["Neut", "Hard", "Soft"]:
            class_sizes = []
            for bin_lab in hft["Bin"].unique():
                subdf = hft[(hft["Bin"] == bin_lab) & (hft["Class"] == subsweep)]
                class_sizes.append(len(subdf))

            hft_prop = class_sizes / hft_bin_sizes
            axs[1, i].plot(hft["Bin"].unique(), hft_prop, label=f"HFT {subsweep}")

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
        axs[2, i].plot(fit["Bin"].unique(), fit_sweep_prop, label=f"FIT Sweep")
        axs[2, i].plot(fit["Bin"].unique(), fit_neut_prop, label=f"FIT Neut")
        """

        for j in range(3):
            loclist = aft["Bin"].unique().tolist()
            axs[i, j].set_xticks([loclist[0], 250000, loclist[-1]])
            axs[i, j].tick_params(axis="x", rotation=45)
            axs[i, j].set_yticks(np.linspace(0, 1.1, 11, endpoint=False))

    for i in range(3):
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in range(3):
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neutral")
    axs[0, 1].set_title("SSV")
    axs[0, 2].set_title("SDN")

    axs[0, 0].set_ylabel("AFT Score")
    # axs[1, 0].set_ylabel("HFT Score")
    # axs[2, 0].set_ylabel("FIT Inv pval")

    axs[0, 0].legend()
    # axs[1, 0].legend()
    # axs[2, 0].legend()

    plt.savefig("propline.pdf")


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
        merged["Sweep_Score"] = merged["Hard_Score"] + merged["Soft_Score"]

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
        cut_bins = list(range(500, 250000, 500)) + (list(range(250500, 501000, 500))) #Want that central window to be exactly in the center
        _df["Bin"] = pd.cut(_df["BP"], bins=cut_bins, labels=cut_bins[:-1]).astype(int)
        binned_dfs.append(_df)
        _df.loc[_df["Mut_Type"] == 2, "Bin"] = 250000 #Ensure that soft sweeps are in the central bin

    return pd.concat(binned_dfs, axis=0)


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
    if sweep in ["soft", "hard"]:
        trues[
            np.array(pred_df.index[(pred_df["Mut_Type"] == 2) & (sweep == "soft")])
        ] = 1
        trues[
            np.array(pred_df.index[(pred_df["Mut_Type"] == 2) & (sweep == "hard")])
        ] = 2
    probs = np.array(pred_df.loc[:, ["Neut_Score", "Soft_Score", "Hard_Score"]])
    preds = np.argmax(probs, axis=1)

    print(f"{sweep}: {sum(trues)}")
    return trues, preds, probs


def main():
    indir = sys.argv[1]
    datadict = {}

    aft_trues = []
    aft_preds = []
    aft_probs = []
    hft_trues = []
    hft_preds = []
    hft_probs = []

    for sweep in ["neut", "soft", "hard"]:
        datadict[sweep] = {}
        csvs = glob(os.path.join(indir, f"{sweep}/*/*.csv"))
        print(f"{len(csvs)} files in {sweep}")

        for model in ["aft"]:  # , "hft", "fit"]:
            rawfiles = [i for i in csvs if model in i]
            preds = load_preds(rawfiles)
            binned = bin_preds(preds)
            print(binned)
            datadict[sweep][model] = binned.sort_values(by="Bin")

            if model == "aft":
                trues, preds, probs = get_ys(preds, sweep)
                aft_trues.extend(trues)
                aft_preds.extend(preds)
                aft_probs.extend(probs)

            elif model == "hft":
                trues, preds, probs = get_ys(preds, sweep)
                hft_trues.extend(trues)
                hft_preds.extend(preds)
                hft_probs.extend(probs)

    plot_roc(np.array(aft_trues), np.array(aft_probs), "aft_ROC", "aft_ROC.pdf", True)
    plot_prec_recall(
        np.array(aft_trues), np.array(aft_probs), "aft_PR", "aft_PR.pdf", True
    )

    aft_cm = confusion_matrix(aft_trues, aft_preds)
    plot_confusion_matrix(
        ".",
        aft_cm,
        target_names=["Neutral", "SSV", "SDN"],
        title="aft_Confmat_normed",
        normalize=True,
    )
    plot_confusion_matrix(
        ".",
        aft_cm,
        target_names=["Neutral", "SSV", "SDN"],
        title="aft_Confmat_unnormed",
        normalize=False,
    )
    # hft_cm = confusion_matrix(hft_trues, hft_preds)
    # plot_confusion_matrix(
    #    ".",
    #    hft_cm,
    #    target_names=["Neut", "Hard", "Soft"],
    #    title="hft_Confmat",
    #    normalize=True,
    # )

    #plot_boxplots(datadict)
    # plot_violinplots(datadict)
    plot_means(datadict)
    plot_proportions(datadict)
    plot_maxes(datadict)


if __name__ == "__main__":
    main()
