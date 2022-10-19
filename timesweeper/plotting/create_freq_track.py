from ..utils import snp_utils as su
from ..utils.gen_utils import get_logger, read_config
import numpy as np
import allel
import os
from tqdm import tqdm

def calc_freq_diffs(genos, snps, samp_sizes):
    """
    Iterates through SNPs in a genotype/snp set and finds calculates the allele frequency change from last to first timepoints.
    Minor allele here is denoted as the allele with the largest change from beginning to end of the time-series, similarly to the standard Timesweeper algorithm.

    Args:
        genos (allel.GenotypeArray): Genotype array of all timepoints.
        snps (list[tuple[chrom, pos]]): Location of each SNP in the VCF iterable. Index-matched with genos.
        samp_sizes (list[int]): Sample sizes of each timepoint, used for indexing the genotype arrays by timepoint.

    Returns:
        tuple[tuple[tuple[chrom, pos], minor_freq_changes, major_freq_changes]]: Tuple of all details needed to create bedfile.
    """
    ts_genos = su.split_arr(genos, samp_sizes)
    min_alleles = su.get_vel_minor_alleles(ts_genos, np.max(genos))
    
    #[(snp, min_change, maj_change)]
    freq_diffs = []

    first_counts = allel.GenotypeArray(ts_genos[0]).count_alleles(
        max_allele=min_alleles.max()
    )
    last_counts = allel.GenotypeArray(ts_genos[-1]).count_alleles(
        max_allele=min_alleles.max()
    )

    for snp_idx, min_allele_idx in tqdm(zip(range(len(first_counts)), min_alleles), total=len(first_counts), desc="Calculating frequency diffs:"):
        first_min_af = su.calc_maft(first_counts[snp_idx], min_allele_idx)
        first_maj_af = 1-first_min_af

        last_min_af = su.calc_maft(last_counts[snp_idx], min_allele_idx)
        last_maj_af = 1-last_min_af

        minor_freq_changes = last_min_af - first_min_af
        major_freq_changes = last_maj_af - first_maj_af
        
        freq_diffs.append((snps[snp_idx], minor_freq_changes, major_freq_changes))
    
    return freq_diffs

def write_bedfiles(outfile, freq_data):
    if len(os.path.dirname(outfile)) > 0:
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

    for filetype in ["major", "minor"]:
        with open(outfile + "." + filetype + ".bedGraph", "w+") as ofile:
            ofile.write("""track type=bedGraph visibility=full color=200,100,0 altColor=0,100,200\n""")
            for i in freq_data:
                (chrom, loc), min_freq_change, maj_freq_change = i
                if filetype == "major":
                    ofile.write("\t".join([str(chrom), str(int(loc)), str(int(loc+1)), str(maj_freq_change)]) + "\n")
                else:
                    ofile.write("\t".join([str(chrom), str(int(loc)), str(int(loc+1)), str(min_freq_change)]) + "\n")

def main(ua):
    yaml_data = read_config(ua.yaml_file)

    logger = get_logger("freq_viz")
    vcf_iter = su.get_vcf_iter(ua.input, benchmark=False)
    for chunk_idx, chunk in enumerate(vcf_iter):
        chunk = chunk[0]  # Why you gotta do me like that, skallel?
        logger.info(f"Processing VCF chunk {chunk_idx}")

        genos, snps = su.vcf_to_genos(chunk, benchmark=False)
        freq_data = calc_freq_diffs(genos, snps, yaml_data["sample sizes"])
        write_bedfiles(ua.output, freq_data)

        