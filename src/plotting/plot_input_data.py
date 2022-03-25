import pickle
from argparse import ArgumentParser
from glob import glob

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

    minMin = 1e-6  # np.amin(data) + 1e-6
    maxMax = 1  # np.amax(data)

    if mat_type == "aft":
        fig, axes = plt.subplots(3, 1)
        normscheme = matplotlib.colors.Normalize(vmin=minMin, vmax=maxMax)

    elif mat_type == "hfs":
        fig, axes = plt.subplots(1, 3)
        normscheme = matplotlib.colors.Normalize(vmin=minMin, vmax=maxMax)

    for i in range(len(data)):
        heatmap = (axes[i].pcolor(data[i], cmap=plt.cm.Blues, norm=normscheme,),)[0]

        plt.colorbar(heatmap, ax=axes[i])
        axes[i].set_title(axTitles[i], fontsize=14)
        # cbar.set_label("foo", rotation=270, labelpad=20, fontsize=10.5)

        axes[i].set_yticks(np.arange(data[i].shape[1]) + 0.5, minor=False)
        axes[i].set_yticklabels(np.arange(data[i].shape[1]))
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].set_xlabel("Timepoint")
        axes[i].set_ylabel("Frequency")

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
    neut_data = [
        pikl_dict["neut"][rep][data_type.lower()] for rep in pikl_dict["neut"].keys()
    ]
    hard_data = [
        pikl_dict["hard"][rep][data_type.lower()] for rep in pikl_dict["hard"].keys()
    ]
    soft_data = [
        pikl_dict["soft"][rep][data_type.lower()] for rep in pikl_dict["soft"].keys()
    ]

    return neut_data, hard_data, soft_data


def getMeanMatrix(data):
    """Returns cell-wise mean of all matrices in a stack."""
    return np.mean(data, axis=0)


def parse_ua():
    """Read in user arguments and sanitize inputs."""
    argparser = ArgumentParser(
        description="Aggregates and plots central SNPs from simulations to visually inspect mean trends over replicates."
    )

    argparser.add_argument(
        "-i",
        "--input-pickle",
        dest="input_pickle",
        metavar="INPUT DIR",
        type=str,
        help="Base directory containing <subdirs>/<aft/hfs>_center.npy. Will search through all sublevels of directories while globbing.",
    )

    argparser.add_argument(
        "-s",
        "--schema-name",
        metavar="SCHEMA NAME",
        dest="schema_name",
        required=False,
        default="simulation_center_means",
        type=str,
        help="Experiment label to use for output file naming.",
    )

    argparser.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT DIR",
        dest="output_dir",
        required=False,
        default=".",
        type=str,
        help="Directory to write images to.",
    )
    user_args = argparser.parse_args()

    return user_args


def main():
    ua = parse_ua()
    plotDir = ua.output_dir
    schema_name = ua.schema_name

    for mat_type in ["aft", "hfs"]:
        base_filename = f"{plotDir}/{schema_name}_{mat_type}"

        neut_list, hard_list, soft_list = readData(ua.input_pickle, mat_type)

        neut_arr = np.stack(neut_list).transpose(0, 2, 1)
        hard_arr = np.stack(hard_list).transpose(0, 2, 1)
        soft_arr = np.stack(soft_list).transpose(0, 2, 1)

        if mat_type == "aft":
            print(
                "Shape of hard samples before mean (samples, snps, timepoints):",
                hard_arr.shape,
            )
        elif mat_type == "hfs":
            print(
                "Shape of hard samples before mean (samples, haps, timepoints):",
                hard_arr.shape,
            )

        mean_neut = getMeanMatrix(neut_arr)
        mean_hard = getMeanMatrix(hard_arr)
        mean_soft = getMeanMatrix(soft_arr)

        print("Shape after mean:", mean_neut.shape)
        print("Biggest value in hard (should be 1):", np.max(hard_arr))

        makeHeatmap(
            mat_type,
            [mean_neut, mean_hard, mean_soft],
            schema_name,
            ["neut", "hard", "soft"],
            base_filename + ".all.png",
        )

        if mat_type == "aft":
            makeHeatmap(
                mat_type,
                [mean_neut[10:40, :], mean_hard[10:40, :], mean_soft[10:40, :]],
                schema_name,
                ["neut", "hard", "soft"],
                base_filename + ".zoomed.png",
            )

            makeHeatmap(
                mat_type,
                [neut_arr[0][10:40, :], hard_arr[0][10:40, :], soft_arr[0][10:40, :]],
                schema_name,
                ["neut", "hard", "soft"],
                base_filename + ".single.zoomed.png",
            )

            np.savetxt("neut.csv", neut_arr[0][10:40, :], delimiter="\t", fmt="%1.2f")
            np.savetxt("hard.csv", hard_arr[0][10:40, :], delimiter="\t", fmt="%1.2f")
            np.savetxt("soft.csv", soft_arr[0][10:40, :], delimiter="\t", fmt="%1.2f")

        elif mat_type == "hfs":
            makeHeatmap(
                mat_type,
                [mean_neut[:20, :], mean_hard[:20, :], mean_soft[:20, :]],
                schema_name,
                ["neut", "hard", "soft"],
                base_filename + ".zoomed.png",
            )


if __name__ == "__main__":
    main()
