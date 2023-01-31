import argparse
import multiprocessing as mp
import os
import warnings
from itertools import cycle

import allel
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from timesweeper.utils import snp_utils as su
from timesweeper.utils.gen_utils import read_config

warnings.filterwarnings("ignore")


def makeHeatmap(data, plotTitle, plotFileName):
    """
    Plot matrices on bluescale heatmap to visualize aft or HFS in a given region.

    Args:
        mat_type (str): aft or HFS, adjusts direction of plots and scaling for easier viewing.
        data (list[np.arr, np.arr, np.arr]): List of 3 numpy arrays to visualize
        plotTitle (str): Title to write at head of plot.
        axTitles (list[str, str, str]): Labels for the three subplots.
        plotFileName (str): Name to use for image file written.

    Writes to file, no return.
    """
    data[data == -1] = np.nan

    plt.figure()
    minMin = np.amin(data) + 1e-6
    maxMax = np.amax(data)

    fig, axes = plt.subplots(1, 1)
    normscheme = matplotlib.colors.Normalize(vmin=minMin, vmax=maxMax)

    heatmap = (
        axes.pcolor(
            data,
            cmap=plt.cm.Blues,
            norm=normscheme,
        ),
    )[0]

    plt.colorbar(heatmap, ax=axes)

    axes.set_yticks([], minor=False)
    axes.set_yticks([25.5], minor=True)
    axes.set_yticklabels(["25"], minor=True)

    axes.set_xlabel("Timepoint")
    axes.set_ylabel("Polymorphism")

    fig.set_size_inches(4, 2.5)
    plt.suptitle(plotTitle, fontsize=10, y=1.08)
    plt.tight_layout()
    plt.savefig(plotFileName, bbox_inches="tight")
    plt.clf()


def parse_ua():
    """Read in user arguments and sanitize inputs."""
    uap = argparse.ArgumentParser(
        description="Plots windows from a BED file of output to visualize data used for prediction."
    )

    uap.add_argument(
        "-i",
        "--input-vcf",
        dest="input_vcf",
        help="Merged VCF to scan for sweeps. Must be merged VCF where files are merged in order from earliest to latest sampling time, -0 flag must be used.",
        required=True,
    )
    uap.add_argument(
        "-b",
        "--input-bedfile",
        dest="input_bedfile",
        metavar="INPUT_BEDFILE",
        type=str,
        help="BED file of regions to visualize.",
    )

    uap.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT DIR",
        dest="output_dir",
        required=False,
        default=".",
        type=str,
        help="Directory to write images to.",
    )

    subparsers = uap.add_subparsers(dest="config_format")
    subparsers.required = True
    yaml_parser = subparsers.add_parser("yaml")
    yaml_parser.add_argument(
        metavar="YAML CONFIG",
        dest="yaml_file",
        help="YAML config file with all cli options defined.",
    )

    cli_parser = subparsers.add_parser("cli")

    cli_parser.add_argument(
        "-s",
        "--sample-sizes",
        dest="samp_sizes",
        help="Number of individuals from each timepoint sampled. Used to index VCF data from earliest to latest sampling points.",
        required=True,
        nargs="+",
        type=int,
    )

    ua = uap.parse_args()

    return ua


def worker(_reg, vcf, samp_sizes, plotDir):
    reg = f"{_reg[0]}:{_reg[1]}-{_reg[2]}"
    fields = ["variants/CHROM", "variants/POS", "calldata/GT"]
    vcf_reg = allel.read_vcf(vcf, fields=fields, region=reg)
    genos, snps = su.vcf_to_genos(vcf_reg, benchmark=False)
    aft = prep_ts_aft(genos, samp_sizes)
    aft = np.swapaxes(aft, 0, 1)

    makeHeatmap(
        aft,
        reg,
        os.path.join(plotDir, reg + ".aft.pdf"),
    )


def main(ua):
    yaml_data = read_config(ua.yaml_file)
    samp_sizes = yaml_data["sample sizes"]

    plotDir = ua.output_dir + "/freq_plots_deduped"
    os.makedirs(plotDir, exist_ok=True)

    with open(ua.input_bedfile, "r") as ifile:
        regions = [i.strip().split()[:3] for i in ifile.readlines()]

    args = zip(regions, cycle([ua.input_vcf]), cycle([samp_sizes]), cycle([plotDir]))

    with mp.Pool(mp.cpu_count()) as p:
        p.starmap(worker, args)


if __name__ == "__main__":
    ua = parse_ua()
    main(ua)
