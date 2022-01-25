import numpy as np
import matplotlib as mpl
from glob import glob

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors


def pad_data(all_data, largest_dim_0, largest_dim_1):
    for idx in range(len(all_data)):
        pad_arr = np.zeros((largest_dim_0, largest_dim_1))
        pad_arr[
            largest_dim_0 - all_data[idx].shape[0] :, : all_data[idx].shape[1],
        ] = all_data[idx]
        all_data[idx] = pad_arr
    return all_data


def makeHeatmap(data, plotTitle, axTitles, plotFileName):
    plt.figure()
    fig, axes = plt.subplots(3, 1)

    minMin = np.amin(data) + 1e-6
    maxMax = np.amax(data)
    # if "freq" in plotFileName:
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

    fig.set_size_inches(3, 5)
    plt.suptitle(plotTitle, fontsize=20, y=1.08)
    plt.tight_layout()
    plt.savefig(plotFileName, bbox_inches="tight")
    plt.clf()


def readData(infiles):
    hard = []
    neut = []
    soft = []

    hardfiles = [aname for aname in infiles if "hard" in aname]
    neutfiles = [aname for aname in infiles if "neut" in aname]
    softfiles = [aname for aname in infiles if "soft" in aname]

    hard = [np.load(npy) for npy in hardfiles]
    print("Loaded hard sweeps")

    neut = [np.load(npy) for npy in neutfiles]
    print("Loaded neut data")

    soft = [np.load(npy) for npy in softfiles]
    print("Loaded soft sweeps")

    # all_data = hard + neut + soft
    # largest_dim_0 = max([i.shape[0] for i in all_data])
    # largest_dim_1 = max([i.shape[1] for i in all_data])
    # print(largest_dim_0, largest_dim_1)

    # hard = pad_data(hard, largest_dim_0, largest_dim_1)
    # soft = pad_data(soft, largest_dim_0, largest_dim_1)
    # neut = pad_data(neut, largest_dim_0, largest_dim_1)

    # Transpose for vertical figures, shape is now (samples, haps, timepoints)
    hard_arr = np.stack(hard).transpose(0, 2, 1)
    neut_arr = np.stack(neut).transpose(0, 2, 1)
    soft_arr = np.stack(soft).transpose(0, 2, 1)

    return hard_arr, neut_arr, soft_arr


def getMeanMatrix(data):
    return np.mean(data, axis=0)


def main():
    base_dir = "/proj/dschridelab/lswhiteh/timesweeper/simple_sims/vcf_sims/onePop-selectiveSweep-vcf.slim"
    input_npys = glob(f"{base_dir}/*/pops/*/afs_centers.npy")
    schema_name = "simple_onepop_selSweep_afs"
    plotDir = f"{base_dir}/images"

    plotFileName = f"{plotDir}/{schema_name}"

    hard_samp, neut_samp, soft_samp = readData(input_npys)

    print("Shape before mean (samples, haps, timepoints):", hard_samp.shape)

    data = []
    data.append(getMeanMatrix(neut_samp))
    data.append(getMeanMatrix(hard_samp))
    data.append(getMeanMatrix(soft_samp))

    print("Shape after mean:", data[0].shape)
    print("Biggest value in hard:", np.max(hard_samp))

    makeHeatmap(
        data, schema_name, ["neut", "hard", "soft"], plotFileName + ".all.png",
    )

    makeHeatmap(
        [data[0][10:40, :], data[1][10:40, :], data[2][10:40, :]],
        schema_name,
        ["neut", "hard", "soft"],
        plotFileName + ".zoomed.png",
    )

    # for i in rd.sample(range(len(soft_samp)), 3):
    #    makeHeatmap(
    #        [neut_samp[i], hard_samp[i], soft_samp[i]],
    #        schema_name + "singles",
    #        ["Neut", "Hard", "Soft"],
    #        f"{plotFileName}_singles_{i}.zoomed.png",
    #    )


if __name__ == "__main__":
    main()
