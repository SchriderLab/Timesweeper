import allel
import numpy as np

# General util functions
def read_vcf(vcf_file):
    """
    Loads VCF file and grabs relevant fields.

    Args:
        vcf_file (str): Path to vcf file.

    Returns:
        allel.vcf object: VCF data in the form of dictionary type object.
    """
    vcf = allel.read_vcf(
        vcf_file,
        fields=["variants/CHROM", "variants/POS", "calldata/GT", "variants/MT"],
    )

    return vcf


def get_geno_arr(vcf):
    """
    Returns Genotype array with calldata.

    Args:
        vcf (allel.vcf object): VCF dict-like object.

    Returns:
        allel.GenotypeArray: Genotype data in the form of np-like array.
    """
    return allel.GenotypeArray(vcf["calldata/GT"])


def make_loc_tups(vcf):
    """
    Zips up chrom, position, and mutations for easy dict-filling.

    Args:
        vcf (allel.vcf object): VCF dict-like object.

    Returns:
        list[tuple]: List of tuples with (chrom, pos, mut).
    """
    return list(zip(vcf["variants/CHROM"], vcf["variants/POS"], vcf["variants/MT"]))


### Get afs from vcf
def vcf_to_genos(vcf_file):
    """
    Takes in vcf file, accesses and collates data for easy genotype dict-filling.

    Args:
        vcf_file (str): Path to vcf file.

    Returns:
        tuple[allel.GenotypeArray, list[tup(chrom, pos,  mut)]]: Genotype arrays and associated id information.
    """
    vcf = read_vcf(vcf_file)
    geno_arr = get_geno_arr(vcf)
    locs = make_loc_tups(vcf)

    return geno_arr, locs


### Haplotypes from VCF
def vcf_to_haps(vcf_file):
    """
    Takes in vcf file, accesses and collates data for easy haplotype dict-filling.

    Args:
        vcf_file (str): Path to vcf file.

    Returns:
        tuple[allel.HaplotypeArray, list[tup(chrom, pos,  mut)]]: Haplotype arrays and associated id information.
    """
    vcf = read_vcf(vcf_file)
    hap_arr = get_geno_arr(vcf).to_haplotypes()
    locs = make_loc_tups(vcf)

    return hap_arr, locs


def split_arr(arr, samp_sizes):
    """
    Restacks array to be in shape (time bins, snps, inds, alleles).
    This method allows for restarts that are reported in the vcf to be handled in the off chance it doesn't get filtered out during sims.

    Args:
        arr (np.arr): SNP or Haplotype array with all timepoints in flat structure.
        samp_sizes (list(int)): List of chromosomes (not individuals) to index from the array.

    Returns:
        np.arr: Time-serialized array of SNP or haplotype data.
    """
    i = arr.shape[1] - sum(samp_sizes)  # Skip restarts for sims
    arr_list = []
    for j in samp_sizes:
        arr_list.append(arr[:, i : i + j])
        i += j

    return np.stack(arr_list)


def get_minor_alleles(ts_genos):
    """
    Gets index of minor allele to use for MAF.
    Based on the highest-frequency minor allele at last timepoint.

    Args:
        ts_genos (np.arr): Time-series array of SNPs organized with split_arr().

    Returns:
        np.arr: Array of indices of minor alleles.
    """
    # Shape is (snps, counts)
    # Use allele that is highest freq at final timepoint
    last_genos = allel.GenotypeArray(ts_genos[-1, :, :, :]).count_alleles()
    return np.argmax(last_genos[:, 1:], axis=1) + 1


def calc_mafs(snp, min_allele_idx):
    """
    Calculates minor allele frequency for given snp.

    Args:
        snp (np.arr): Frequencies of all minor alleles for a given SNP.
        min_allele_idx (int): Index of minor allele at highest frequency at final timepoint.

    Returns:
        float: Minor allele frequency (MAF) at a given timepoint.
    """
    return snp[min_allele_idx] / snp.sum()
