import numpy as np
import pandas as pd
import sys


def load_data(
    infile="../empirical_model/Online Table 1 Newly reported ancient individuals.csv",
):
    """Gets and prints sample numbers from the mongolian paper."""
    raw_data = pd.read_csv(infile, header=0)
    bp_dates = raw_data.iloc[:, [5, 6, 10, 13, 18]]
    bp_dates.columns = [
        "mean_dates",
        "stddev_dates",
        "study_pass",
        "location",
        "avg_cov",
    ]

    # Filtering to only get passing samples or those that failed from contamination
    bp_dates = bp_dates[
        (bp_dates["location"] == "Mongolia")
        & (bp_dates["avg_cov"] > 1)
        & (
            (bp_dates["study_pass"] == "Yes")
            | (bp_dates["study_pass"] == "Yes and plotted in Figure 3")
            | (bp_dates["study_pass"].str.contains("contamination"))
        )
    ]

    # print(bp_dates)
    # plt.hist(pd.to_numeric(bp_dates["avg_cov"]), 20)
    # plt.title("Average Coverage for Passing Samples")
    # plt.savefig("coverage.png")

    actual_dates = []
    for i in bp_dates.itertuples():
        # Correct for BP standardization
        actual_dates.append(int(abs(1950 - i[1])))
        # sampled_dates.append(abs(1950 - int(np.random.normal(i[1], i[2]))))

    return actual_dates


def bin_times(years, bin_window):
    """
    Creates sampling bins to get sample sizes for each generation.
    
    Arguments:
        years (List[int]): Years of the sampling that took place.
        max_time (int): Last year of the simulation, used for binning.
        bin_window (int): Number of years to span for a bin.
        size_threshold (int): Smallest number of 

    Returns:

        counts - only values where there are at least one sample present
        
        bin_edges (left inclusive) - time to sample for each count
    """
    counts, edges = np.histogram(years, range(0, max(years), bin_window))
    trimmed_edges = np.delete(edges, 0)

    sample_counts = counts[counts > 0]
    binned_years = trimmed_edges[counts > 0]

    return sample_counts, binned_years


def bin_samps(samp_sizes, gens_sampled, gen_threshold, size_threshold):
    """
    Bins a list of ints into condensed bins where the minimum value is equal to <size_threshold>.
    Each bin must also not be larger than <gen_threshold>.

    Args:
        samp_sizes (list[int]): List of ints to bin.
        gens_sampled (list[int]): List of generations sampled.
        gen_threshold (int, optional): Minimum size of generation window any given bin can be. Defaults to 25.
        size_threshold (int, optional): Minimum number of samples in any given bin. Defaults to 3.

    Returns (prints):
        list[int]: Binned values.
    """
    bin_inds = []  # End of each bin

    i = 0
    while i < len(samp_sizes):
        if samp_sizes[i] >= size_threshold:
            i += 1
            bin_inds.append(i)

        else:
            j = 0
            while sum(samp_sizes[i : i + j]) < size_threshold:
                if i + j == len(gens_sampled):
                    # Hit the end before it's good, just take whatever's left
                    break
                elif gens_sampled[i + j] - gens_sampled[i] > gen_threshold:
                    # Good to go, append sample
                    break
                else:
                    # Need more samples, add the next timepoint
                    j += 1

            bin_inds.append(i + j)
            i += j

    binned_gens = []
    binned_sizes = []
    i = 0
    for j in bin_inds:
        binned_sizes.append(sum(samp_sizes[i:j]))
        binned_gens.append(int(np.mean(gens_sampled[i:j])))
        i = j

    return binned_sizes, binned_gens


def main():
    """
    For binning the samples in the Mongolian dataset to boost numbers per sampling point.
    This script is specific to the analysis done for the data in this paper, not for general use at the moment.
    """
    gen_time = 25  # Should be consistent with the SLIM file
    size_threshold = 3  # Minimum number of individuals for a bin
    gen_threshold = 10
    years_sampled = load_data()

    # Convert to bins based on generation time and get samples per gen
    sample_counts, binned_years = bin_times(years_sampled, gen_time)
    # Bin into reasonably-sized samps
    binned_sizes, binned_gens = bin_samps(
        sample_counts,
        [int(i / gen_time) for i in binned_years],
        gen_threshold,
        size_threshold,
    )
    print("Sizes:", " ".join([str(i) for i in binned_sizes[::-1]]))
    print(
        "Years before present (1950):",
        " ".join([str(int(i * gen_time)) for i in binned_gens[::-1]]),
    )


if __name__ == "__main__":
    main()
