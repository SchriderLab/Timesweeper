import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main(ua):
    if not os.path.exists(ua.output_dir):
        os.makedirs(ua.output_dir)

    log_df = pd.read_csv(ua.summary_tsv, header=0, sep="\t")
    acc_df = pd.read_csv(ua.test_file, header=0, sep=",")
    merged = pd.merge(
        right=acc_df,
        left=log_df,
        how="right",
        left_on=["rep", "sweep"],
        right_on=["rep", "class"],
    )
    merged["abs_error"] = np.abs(merged["pred_sel_coeff"] - merged["true_sel_coeff"])
    merged.to_csv(
        f"{ua.output_dir}/merged_table.tsv", index=False, header=True, sep="\t"
    )

    merged.plot(
        x="selCoeff",
        y="abs_error",
        kind="scatter",
        title="Absolute S Prediction Error vs Selection Coefficient",
        xlabel="Selection Coefficient s",
        ylabel="Absolute Error | abs(predicted s - true s)",
        xlim=(0.0, 0.25),
        ylim=(0.0, 0.25),
    )
    plt.savefig(f"{ua.output_dir}/error_vs_selcoeff.png")
    plt.clf()

    merged.plot(
        x="sampOffset",
        y="abs_error",
        kind="scatter",
        title="Absolute S Prediction Error vs Selection Coefficient",
        xlabel="Sampling Offset from Selection",
        ylabel="Absolute Error | abs(predicted s - true s)",
        xlim=(-50, 50),
        ylim=(0.0, 0.25),
        xticks=range(-50, 60, 10),
    )
    plt.savefig(f"{ua.output_dir}/error_vs_sampletime.png")
    plt.clf()

    merged.plot(
        x="sampOffset",
        y="selCoeff",
        kind="scatter",
        title="Sampling Offset from Selection vs Selection Coefficient",
        xlabel="Sampling Offset from Selection",
        ylabel="Selection Coefficient",
        xlim=(-50, 50),
        ylim=(0.0, 0.25),
        xticks=range(-50, 60, 10),
    )
    plt.savefig(f"{ua.output_dir}/selcoeff_vs_sampletime.png")
    plt.clf()
