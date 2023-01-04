import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp
from tqdm import tqdm
import allel
from ..utils import snp_utils as su
from ..make_training_features import prep_ts_aft

warnings.filterwarnings("ignore", category=RuntimeWarning)

"""
Time series:
-	FIT
-	FET
-	spectralHMM
Single point: final timepoint
-	S/HIC
-	Fay and Wu’s H (windowed, from S/HIC) 
-	SweepFinder
"""


def fet(maj_counts, min_counts):
    """
    Run Fisher's exact test using minor allele frequency vector given by prep_ts_aft.add()

    Args:
        maf_vector (list[float]): Time-series of allele frequencies for highest velocity allele in a given SNP.add()

    Returns:
        float: p-value, result of FET on first and last minor/major allele frequency contingency table.
    """
    return stats.fisher_exact(
        [
            [maj_counts[0], maj_counts[-1]],
            [min_counts[0], min_counts[-1]],
        ]
    )[1]


def fit(freqs, gens):
    """
    Calculate FIT by performing 1-Sided Student t-test on frequency increments.

    Args:
        freqs (List[int]): Frequencies at all given generations for targeted alleles.
        gens (List[int]): Generations sampled.

    Returns:
        List[int]: t-test results.
    """
    incs = []
    i = 0
    # advance to first non-zero freq
    while i < len(freqs) and (freqs[i] == 0 or freqs[i] == 1):
        i += 1
    if i < len(freqs):
        prevFreq = freqs[i]
        prevGen = gens[i]
        i += 1
        while i < len(freqs):
            if freqs[i] != 0 and freqs[i] != 1:
                num = freqs[i] - prevFreq
                denom = ((2 * prevFreq) * (1 - prevFreq) * (gens[i] - prevGen)) ** 0.5
                incs.append(num / denom)
                prevFreq = freqs[i]
                prevGen = gens[i]
            i += 1

    return ttest_1samp(incs, 0)


def write_fit(fit_dict, outfile, benchmark):
    """
    Writes FIT predictions to file.

    Args:
        fit_dict (dict): FIT p values and SNP information.
        outfile (str): File to write results to.
    """
    inv_pval = [1 - i[1] for i in fit_dict.values()]
    if benchmark:
        chroms, bps, mut_type = zip(*fit_dict.keys())
        predictions = pd.DataFrame(
            {"Chrom": chroms, "BP": bps, "Mut_Type": mut_type, "Inv_pval": inv_pval}
        )
    else:
        chroms, bps = zip(*fit_dict.keys())
        predictions = pd.DataFrame({"Chrom": chroms, "BP": bps, "Inv_pval": inv_pval})

    predictions.dropna(inplace=True)
    predictions.sort_values(["Chrom", "BP"], inplace=True)
    predictions.to_csv(
        os.path.join(outfile),
        header=True,
        index=False,
        sep="\t",
    )


def get_ua():
    parser = argparse.ArgumentParser(
        description="Module to run alternative selective sweep detection methods for both time series and single-timepoint data. Methods in this script include: Fisher's Exact Test (FET), Frequency Increment Test (FIT), SweepFinder2, spectralHMM, diploSHIC, Fay and Wu's H."
    )
    parser.add_argument(
        "-v",
        "--vcf",
        action="store",
        required=True,
        help="Multi-timepoint input VCF file formatted as required for Timesweeper.",
    )
    parser.add_argument(
        "-o", "--out", action="store", required=True, help="Output CSV."
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        action="store_true",
        help="Use this flag to denote the input_vcf is a simulated sample and the SLIM mutation type can be parsed.",
    )
    parser.add_argument(
        "-m",
        "--methods",
        action="store",
        nargs="+",
        choices=["FIT-FET", "SweepFinder2", "diploSHIC", "HMM"],
        required=False,
        help="Which methods to use.",
    )
    parser.add_argument(
        "-dm",
        "--diploshic-model",
        action="store",
        required=False,
        help="If using diploSHIC must provide a trained model.",
    )
    parser.add_argument(
        "-s",
        "--sample-sizes",
        action="store",
        nargs="+",
        dtype=int,
        required=True,
        help="Number of sampled individuals at each timepoint separated by a space.",
    )
    parser.add_argument(
        "-g",
        "--gens-sampled",
        action="store",
        nargs="+",
        dtype=int,
        required=True,
        help="Generation of each sampling point. Relative spacing between values is only relevant aspect, so can be generations from present or from 0.",
    )
    return parser.parse_args()


def main():
    """
    Load VCF
    Iterate through SNPs
    Calculate frequencies
    Do tests

    Convert VCF to SF input
    Run SF2
    """

    args = get_ua()
    in_vcf = args.vcf
    outfile = args.out
    benchmark = args.benchmark
    methods = args.methods
    shic_model = args.dm
    samp_sizes = args.samp_sizes
    gens_sampled = args.gens_sampled

    if "FIT-FET" in methods:
        print("Running Frequency Increment Test (FIT) and Fisher's Exact Test (FET)")

        vcf_iter = su.get_vcf_iter(in_vcf, benchmark)
        for chunk_idx, chunk in enumerate(vcf_iter):
            chunk = chunk[0]  # Why you gotta do me like that, skallel?

            genos, snps = su.vcf_to_genos(chunk, benchmark)
            ts_aft = prep_ts_aft(genos, samp_sizes)

            # FIT
            results_dict = {}
            for idx in tqdm(
                range(len(snps)), desc=f"Calculating FIT for chunk {chunk_idx}"
            ):
                results_dict[snps[idx]] = fit(
                    list(ts_aft[:, idx]), gens_sampled
                )  # tval, pval

            ts_genos = su.split_arr(genos, samp_sizes)
            min_alleles = su.get_vel_minor_alleles(ts_genos, np.max(genos))

            # FET
            for timepoint in ts_genos:
                _genotypes = allel.GenotypeArray(timepoint).count_alleles(
                    max_allele=min_alleles.max()
                )

                major_geno_counts = []
                minor_geno_counts = []
                for snp, min_allele_idx in zip(_genotypes, min_alleles):
                    ## Get major & minor allele counts
                    cur_maj_counts, cur_min_counts = su.get_allele_counts(
                        snp, min_allele_idx
                    )

                    ## Append major & minor counts
                    major_geno_counts.append(cur_maj_counts)
                    minor_geno_counts.append(cur_min_counts)

                fet_pvals = []
                ## Perform Fisher's exact test on each snp
                for i in range(0, len(major_geno_counts[0])):
                    fet_pvals.append(fet(major_geno_counts[i], minor_geno_counts[i]))

    return results_dict


def run_fet_windows(genos, samp_sizes):
    """
    Iterates through timepoints and creates fisher's exact test feature matrices.

    Args:
        genos (allel.GenotypeArray): Genotype array containing all timepoints.
        samp_sizes (list[int]): Number of chromosomes sampled at each timepoint.

    Returns:
        list: P values array from fisher's exact test. Shape is (timepoints, fet).
    """

    ts_genos = su.split_arr(genos, samp_sizes)
    min_alleles = su.get_vel_minor_alleles(ts_genos, np.max(genos))

    ## Get all major & minor allele counts
    for timepoint in ts_genos:
        _genotypes = allel.GenotypeArray(timepoint).count_alleles(
            max_allele=min_alleles.max()
        )

        major_geno_counts = []
        minor_geno_counts = []
        for snp, min_allele_idx in zip(_genotypes, min_alleles):
            ## Get major & minor allele counts
            cur_maj_counts, cur_min_counts = su.get_allele_counts(snp, min_allele_idx)

            ## Append major & minor counts
            major_geno_counts.append(cur_maj_counts)
            minor_geno_counts.append(cur_min_counts)

    fet_pvals = []
    ## Perform Fisher's exact test on each snp
    for i in range(0, len(major_geno_counts[0])):

        cur_snp_pval = fet(major_geno_counts[i], minor_geno_counts[i])

        fet_pvals.append(fet_geno_pvals)

    return fet_pvals
