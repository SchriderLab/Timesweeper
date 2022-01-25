import argparse
import os
from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from plotting_utils import plot_confusion_matrix

mpl.rcParams["agg.path.chunksize"] = 10000


def load_preds(csvfiles):
    all_results = [pd.read_csv(i, sep="\t", header=0) for i in csvfiles]
    merged = pd.concat(all_results, ignore_index=True)
    merged.groupby(["Chrom", "BP"]).mean()
    if "fit" not in csvfiles[0]:
        merged["Sweep Score"] = merged["Hard Score"] + merged["Soft Score"]

    return merged


def bin_preds(merged_scores):
    binned_dfs = []
    for chrom in merged_scores["Chrom"].unique():
        _df = merged_scores[merged_scores["Chrom"] == chrom]
        cut_bins = np.linspace(_df["BP"].min() - 1, _df["BP"].max(), 22)
        _df["Bin"] = pd.cut(_df["BP"], bins=cut_bins, labels=cut_bins[:-1]).astype(int)
        # print(_df[_df["Bin"].isnull().values])
        binned_dfs.append(_df)
        _df.loc[_df["Mut Type"] == 2, "Bin"] = 250000

    return pd.concat(binned_dfs, axis=0)


def squash_dist(binned_dfs, model):
    if model in ["afs", "hfs"]:
        binned_dfs["Log Score"] = -np.log10(binned_dfs["Sweep Score"])
    else:
        binned_dfs["Log Score"] = -np.log10(binned_dfs["Inv pval"])

    return binned_dfs


def plot_nn_sweep(preds_dict):
    plt.clf()
    plt.rcParams["figure.figsize"] = (30, 20)
    plt.rcParams["axes.titlesize"] = "large"
    plt.rcParams["axes.labelsize"] = "large"

    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Simple Simulations Scores Over Chromosome", fontsize=22)
    for i, sweep in enumerate(["neut", "hard", "soft"]):
        # _df = preds_dict[sweep]["afs"]
        # axs[i, 0].violinplot(
        #    [_df[_df["Bin"] == bp_bin]["Log Score"] for bp_bin in _df["Bin"].unique()]
        # )
        preds_dict[sweep]["afs"].boxplot(column=["Sweep Score"], by="Bin", ax=axs[0, i])
        # axs[i, 0].plot(
        #    preds_dict[sweep]["afs"]["BP"],
        #    preds_dict[sweep]["afs"][f"{sweep.capitalize()} Score"],
        # )
        axs[i, 0].tick_params(axis="x", rotation=45)
        # axs[i, 0].set_xticklabels(preds_dict[sweep]["afs"]["Bin"].astype(int))

        # _df = preds_dict[sweep]["hfs"]
        # axs[i, 1].violinplot(
        #    [_df[_df["Bin"] == bp_bin]["Log Score"] for bp_bin in _df["Bin"].unique()]
        # )
        preds_dict[sweep]["hfs"].boxplot(column=["Sweep Score"], by="Bin", ax=axs[1, i])
        # axs[i, 1].plot(
        #    preds_dict[sweep]["hfs"]["BP"],
        #    preds_dict[sweep]["hfs"][f"{sweep.capitalize()} Score"],
        # )
        axs[i, 1].tick_params(axis="x", rotation=45)
        # axs[i, 1].set_xticklabels(preds_dict[sweep]["hfs"]["Bin"].astype(int))

        preds_dict[sweep]["fit"].boxplot(column=["Inv pval"], by="Bin", ax=axs[2, i])

        # _df = preds_dict[sweep]["fit"]
        # axs[i, 2].violinplot(
        #    [_df[_df["Bin"] == bp_bin]["Log Score"] for bp_bin in _df["Bin"].unique()]
        # )
        # axs[i, 2].plot(
        #    preds_dict[sweep]["fit"]["BP"],
        #    preds_dict[sweep]["fit"]["Inv pval"],
        # )
        axs[i, 2].tick_params(axis="x", rotation=45)
        # axs[i, 2].set_xticklabels(preds_dict[sweep]["fit"]["Bin"].astype(int))

    for i in [0, 1, 2]:
        axs[1, i].set_title("")
        axs[2, i].set_title("")
        for j in [0, 1, 2]:
            axs[i, j].set_xlabel("")

    axs[2, 1].set_xlabel("SNP Location")

    axs[0, 0].set_title("Neut")
    axs[0, 1].set_title("Hard")
    axs[0, 2].set_title("Soft")

    axs[0, 0].set_ylabel("AFS Score")
    axs[1, 0].set_ylabel("HFS Score")
    axs[2, 0].set_ylabel("FIT Inv pval")

    plt.savefig(f"test.png")


def plot_fit(preds):
    preds.plot(x="BP", y="Inv pval")
    plt.savefig(f"test_fit.png")


def get_ys(pred_df, sweep):
    trues = [0] * len(pred_df)
    if sweep in ["hard", "soft"]:
        for i in pred_df.index[
            (pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Hard")
        ].tolist():
            trues[i] = 1
        for i in pred_df.index[
            (pred_df["Mut Type"] == 2) & (pred_df["Class"] == "Soft")
        ].tolist():
            trues[i] = 2

    preds = np.argmax(np.array(pred_df.loc[:, "Neut Score":"Soft Score"]), axis=1)

    print(f"{sweep}: {sum(trues)}")
    return trues, preds


def main():
    indir = "/proj/dschridelab/lswhiteh/timesweeper/simple_sims/vcf_sims/onePop-selectiveSweep-vcf.slim/"  # hard/pops/997/Timesweeper_predictions_afs.csv"
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
            # squashed = squash_dist(binned, model)
            datadict[sweep][model] = binned

            if model == "afs":
                trues, preds = get_ys(preds, sweep)
                afs_trues.extend(trues)
                afs_preds.extend(preds)
            elif model == "hfs":
                trues, preds = get_ys(preds, sweep)
                hfs_trues.extend(trues)
                hfs_preds.extend(preds)

    # print(hfs_trues.count(1))
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

    plot_nn_sweep(datadict)


if __name__ == "__main__":
    main()
