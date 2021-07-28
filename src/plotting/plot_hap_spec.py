import os, sys
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
import random as rd

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors


def makeHeatmap(data, plotTitle, axTitles, plotFileName):
    plt.figure()
    fig, axes = plt.subplots(1, 3)

    minMin = np.amin(data) + 0.000001
    maxMax = np.amax(data)
    # print("min", minMin)
    # print("max", maxMax)

    for i in range(len(data)):
        heatmap = (
            axes[i].pcolor(
                data[i],
                cmap=plt.cm.Blues,
                norm=matplotlib.colors.LogNorm(
                    vmin=minMin,
                    vmax=maxMax,
                ),
            ),
        )[0]

        plt.colorbar(heatmap, ax=axes[i])
        axes[i].set_title(axTitles[i], fontsize=14)
        # cbar.set_label("foo", rotation=270, labelpad=20, fontsize=10.5)

        axes[i].set_yticks(np.arange(data[i].shape[1]) + 0.5, minor=False)
        axes[i].set_yticklabels(np.arange(data[i].shape[1]))
        axes[i].set_yticks([])
        axes[i].set_xticks([])
        axes[i].set_xlabel("Timepoint")
        axes[i].set_ylabel("Frequency")

    fig.set_size_inches(5, 3)
    plt.suptitle(plotTitle, fontsize=20, y=1.08)
    plt.tight_layout()
    plt.savefig(plotFileName, bbox_inches="tight")
    plt.clf()


def readNpzData(inFileName):
    u = np.load(inFileName)
    hard = []
    neut = []
    soft = []

    hardfiles = [aname for aname in u.files if "hard" in aname]
    neutfiles = [aname for aname in u.files if "neut" in aname]
    softfiles = [aname for aname in u.files if "soft" in aname]

    hard = [u[aname] for aname in hardfiles]
    print("Loaded hard sweeps")

    neut = [u[aname] for aname in neutfiles]
    print("Loaded neut data")

    soft = [u[aname] for aname in softfiles]
    print("Loaded soft sweeps")

    # Transpose for vertical figures, shape is now (samples, haps, timepoints)
    hard_arr = np.stack(hard).transpose(0, 2, 1)
    neut_arr = np.stack(neut).transpose(0, 2, 1)
    soft_arr = np.stack(soft).transpose(0, 2, 1)

    return hard_arr, neut_arr, soft_arr


def getMeanMatrix(data):
    return np.mean(data, axis=0)


input_npz = sys.argv[1]
schema_name = os.path.basename(input_npz).split(".")[0]
plotDir = os.path.join(os.path.dirname(input_npz), "images")

plotFileName = f"{plotDir}/{schema_name}.mean"

hard_samp, neut_samp, soft_samp = readNpzData(input_npz)

print("Shape before mean (samples, timepoints, haps):", hard_samp.shape)

data = []
data.append(getMeanMatrix(hard_samp))
data.append(getMeanMatrix(neut_samp))
data.append(getMeanMatrix(soft_samp))

print("Shape after mean:", data[0].shape)

makeHeatmap(
    data,
    schema_name,
    ["hard", "neut", "soft"],
    plotFileName + ".all.png",
)

makeHeatmap(
    [data[0][:17, :], data[1][:17, :], data[2][:17, :]],
    schema_name,
    ["hard", "neut", "soft"],
    plotFileName + ".zoomed.png",
)