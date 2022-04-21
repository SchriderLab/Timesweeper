import pickle
from argparse import ArgumentParser

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

    if mat_type == "aft":
        fig, axes = plt.subplots(3, 1)
        normscheme = matplotlib.colors.Normalize(vmin=minMin, vmax=maxMax)

    for i in range(len(data)):
        heatmap = (axes[i].pcolor(data[i], cmap=plt.cm.Blues, norm=normscheme,),)[0]

        plt.colorbar(heatmap, ax=axes[i])
        axes[i].set_title(axTitles[i], fontsize=14)
        # cbar.set_label("foo", rotation=270, labelpad=20, fontsize=10.5)

        axes[i].set_yticks([], minor=False)
        axes[i].set_yticks([25.5], minor=True)
        axes[i].set_yticklabels(["25"], minor=True)
        axes[i].set_xlabel("Timepoint")
        axes[i].set_ylabel("Polymorphism")

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
    return np.nanmean(data, axis=0)


def parse_ua():
    """Read in user arguments and sanitize inputs."""
    argparser = ArgumentParser(
        description="Aggregates and plots central SNPs from simulations to visually inspect mean trends over replicates."
    )

    argparser.add_argument(
        "-i",
        "--input-pickle",
        dest="input_pickle",
        metavar="INPUT PICKLE",
        type=str,
        help="Pickle file containing dictionary of structure dict[sweep][rep]['aft'] created by make_training_features.py.",
    )

    argparser.add_argument(
        "-n",
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
    argparser.add_argument(
        "--save-example",
        dest="save_example",
        required=False,
        action="store_true",
        help="Will create a directory with example input matrices.",
    )
    user_args = argparser.parse_args()

    return user_args


def main(ua):
    plotDir = ua.output_dir
    schema_name = ua.schema_name

    for mat_type in ["aft"]:
        base_filename = f"{plotDir}/{schema_name}_{mat_type}"

        raw_data = {}
        for lab, data_list in zip(
            ["neut", "hard", "soft"], readData(ua.input_pickle, mat_type)
        ):
            raw_data[lab] = np.stack(data_list).transpose(0, 2, 1)

        if mat_type == "aft":
            print(
                "Shape of hard samples before mean (samples, snps, timepoints):",
                raw_data["hard"].shape,
            )

        # Remove missingness for plotting's sake
        mean_data = {}
        for lab in raw_data.keys():
            raw_data[lab][raw_data[lab] == -1] = np.nan
            mean_data[lab] = getMeanMatrix(raw_data[lab])

        print("Shape after mean:", mean_data["hard"].shape)
        # print("Biggest value in hard:", np.max(mean_data["hard"]))

        makeHeatmap(
            mat_type,
            [mean_data["neut"], mean_data["hard"], mean_data["soft"]],
            schema_name,
            ["Neutral", "SDN", "SSV"],
            base_filename + ".all.png",
        )

        if mat_type == "aft":
            #    makeHeatmap(
            #        mat_type,
            #        [
            #            mean_data["neut"][10:40, :],
            #            mean_data["hard"][10:40, :],
            #            mean_data["soft"][10:40, :],
            #        ],
            #        schema_name,
            #        ["Neutral", "SDN", "SSV"],
            #        base_filename + ".zoomed.png",
            #    )

            makeHeatmap(
                mat_type,
                [raw_data["neut"][1], raw_data["hard"][1], raw_data["soft"][1],],
                schema_name,
                ["Neutral", "SDN", "SSV"],
                base_filename + ".single.png",
            )

            # makeHeatmap(
            #    mat_type,
            #    [
            #        raw_data["neut"][0][10:40, :],
            #        raw_data["hard"][0][10:40, :],
            #        raw_data["soft"][0][10:40, :],
            #    ],
            #    schema_name,
            #    ["Neutral", "SDN", "SSV"],
            #    base_filename + ".single.zoomed.png",
            # )

            if ua.save_example:
                for label in ["neut", "hard", "soft"]:
                    np.savetxt(
                        f"{label}.csv", raw_data[label][0], delimiter="\t", fmt="%1.2f",
                    )


if __name__ == "__main__":
    ua = parse_ua()
    main(ua)
