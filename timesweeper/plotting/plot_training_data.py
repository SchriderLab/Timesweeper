import os
import pickle

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")


def makeHeatmap(mat_type, data, plotTitle, axTitles, plotFileName):
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
    plt.figure()
    minMin = np.amin(data) + 1e-6
    maxMax = np.amax(data)
    normscheme = matplotlib.colors.Normalize(vmin=minMin, vmax=maxMax)

    if mat_type == "aft":
        fig, axes = plt.subplots(len(data), 1)
        for i in range(len(data)):
            heatmap = (
                axes[i].pcolor(
                    data[i],
                    cmap=plt.cm.Blues,
                    norm=normscheme,
                ),
            )[0]

            plt.colorbar(heatmap, ax=axes[i])
            axes[i].set_title(axTitles[i], fontsize=14)
            axes[i].set_yticks([], minor=False)
            if "zoomed" in plotFileName:
                axes[i].set_yticks([10.5], minor=True)
                axes[i].set_yticklabels(["25"], minor=True)
            else:
                axes[i].set_yticks([25.5], minor=True)
                axes[i].set_yticklabels(["25"], minor=True)

            axes[i].set_xlabel("Timepoint")
            axes[i].set_ylabel("Polymorphism")

        fig.set_size_inches(5, 5)

    elif mat_type == "hft":
        fig, axes = plt.subplots(1, len(data))
        for i in range(len(data)):
            heatmap = (
                axes[i].pcolor(
                    data[i],
                    cmap=plt.cm.Blues,
                    norm=normscheme,
                ),
            )[0]

            plt.colorbar(heatmap, ax=axes[i])
            axes[i].set_title(axTitles[i], fontsize=14)
            axes[i].set_xlabel("Timepoint")
            axes[i].set_ylabel("Haplotype")

        fig.set_size_inches(5, 5)

    plt.suptitle(plotTitle, fontsize=20, y=1.08)
    plt.tight_layout()
    plt.savefig(plotFileName, bbox_inches="tight")
    plt.clf()


def readData(picklefile, data_type):
    """
    Loads in npy files and sorts based on sweep label.

    Args:
        infiles (list[filenames]): List of filenames found with glob.

    Returns:
        Tuple[list[np.arr], list[np.arr], list[np.arr]]: Neut, hard, soft lists of arrays for processing.
    """

    pikl_dict = pickle.load(open(picklefile, "rb"))
    sweep_types = pikl_dict.keys()
    data_dict = {}
    for key in sweep_types:
        data_dict[key] = [
            pikl_dict[key][rep][data_type.lower()] for rep in pikl_dict[key].keys()
        ]

    return data_dict


def getMeanMatrix(data):
    """Returns cell-wise mean of all matrices in a stack."""
    return np.nanmean(data, axis=0)


def main(ua):
    plotDir = ua.output_dir
    schema_name = ua.schema_name

    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    for mat_type in ["aft", "hft"]:
        base_filename = f"{plotDir}/{schema_name}_{mat_type}"

        raw_data = {}
        data_dict = readData(ua.input_pickle, mat_type)
        for lab in data_dict:
            raw_data[lab] = np.stack(data_dict[lab]).transpose(0, 2, 1)

        if mat_type == "aft":
            print(
                "Shape of AFT samples before mean (samples, snps, timepoints):",
                raw_data[lab].shape,
            )
        elif mat_type == "hft":
            print(
                "Shape of HFT samples before mean (samples, haps, timepoints):",
                raw_data[lab].shape,
            )

        # Remove missingness for plotting's sake
        mean_data = {}
        if len(raw_data.keys()) == 3:
            labs = ["neut", "hard", "soft"]
        elif len(raw_data.keys()) == 2:
            labs = raw_data.keys()

        for lab in labs:
            raw_data[lab][raw_data[lab] == -1] = np.nan
            mean_data[lab] = getMeanMatrix(raw_data[lab])

        print("Shape after mean:", mean_data[lab].shape)

        if mat_type == "aft":
            makeHeatmap(
                mat_type,
                [mean_data[i][15:36, :] for i in mean_data],
                schema_name,
                [i.upper() for i in labs],
                base_filename + f".zoomed.png",
            )
            for j in range(3):
                makeHeatmap(
                    mat_type,
                    [raw_data[i][j] for i in raw_data],
                    schema_name,
                    [i.upper() for i in labs],
                    base_filename + f".{j}.single.png",
                )

        elif mat_type == "hft":
            makeHeatmap(
                mat_type,
                [mean_data[i][:40, :] for i in mean_data],
                schema_name,
                [i.upper() for i in labs],
                base_filename + f".zoomed.png",
            )
            for j in range(3):
                makeHeatmap(
                    mat_type,
                    [raw_data[i][j][:40, :] for i in raw_data],
                    schema_name,
                    [i.upper() for i in labs],
                    base_filename + f".{j}.single.png",
                )

        makeHeatmap(
            mat_type,
            [mean_data[i] for i in mean_data],
            schema_name,
            [i.upper() for i in labs],
            base_filename + f".all.png",
        )

        if ua.save_example:
            for label in raw_data:
                np.savetxt(
                    f"{mat_type}_{label}.csv",
                    raw_data[label][0],
                    delimiter="\t",
                    fmt="%1.2f",
                )
