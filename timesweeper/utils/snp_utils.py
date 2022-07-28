import allel
import numpy as np

# General util functions
def read_vcf(vcf_file, benchmark):
    """
    Loads VCF file and grabs relevant fields.
    For generating training data from simulated VCFs, which are typically small.

    Args:
        vcf_file (str): Path to vcf file.
        benchmark (bool): Whether to look for Mut_Type or not.
    Returns:
        allel.vcf object: VCF data in the form of dictionary type object.
    """
    if benchmark:
        fields = [
            "variants/CHROM",
            "variants/POS",
            "calldata/GT",
            "variants/MT",
            "variants/S",
        ]
    else:
        fields = ["variants/CHROM", "variants/POS", "calldata/GT"]

    vcf = allel.read_vcf(vcf_file, fields=fields)

    return vcf


def get_vcf_iter(vcf_file, benchmark):
    """
    Loads VCF file into allel generator and grabs relevant fields.
    For real data VCFs that are too large to load into memory.

    Args:
        vcf_file (str): Path to vcf file.
        benchmark (bool): Whether to look for Mut_Type or not.

    Returns:
        allel.vcf_iterator object: Generator for VCF data in the form of dictionary type object.
    """
    if benchmark:
        fields = [
            "variants/CHROM",
            "variants/POS",
            "calldata/GT",
            "variants/MT",
            "variants/S",
        ]
    else:
        fields = ["variants/CHROM", "variants/POS", "calldata/GT"]

    _fields, _samples, _headers, vcf_iter = allel.iter_vcf_chunks(
        vcf_file, fields=fields, chunk_length=100000
    )

    return vcf_iter


def get_geno_arr(vcf):
    """
    Returns Genotype array with calldata.

    Args:
        vcf (allel.vcf object): VCF dict-like object.

    Returns:
        allel.GenotypeArray: Genotype data in the form of np-like array.
    """
    return allel.GenotypeArray(vcf["calldata/GT"])


def make_loc_tups(vcf, benchmark):
    """
    Zips up chrom, position, and mutations for easy dict-filling.

    Args:
        vcf (allel.vcf object): VCF dict-like object.
        benchmark (bool): Whether to look for Mut_Type or not.

    Returns:
        list[tuple]: List of tuples with (chrom, pos, mut).
    """
    if benchmark:
        return list(
            zip(
                vcf["variants/CHROM"],
                vcf["variants/POS"],
                vcf["variants/MT"],
                vcf["variants/S"],
            )
        )
    else:
        return list(zip(vcf["variants/CHROM"], vcf["variants/POS"]))


### Get aft from vcf
def vcf_to_genos(vcf, benchmark):
    """
    Takes in vcf file, accesses and collates data for easy genotype dict-filling.

    Args:
        vcf_obj (allel.vcf): Loaded VCF object (whole or chunked).
        benchmark (bool): Whether to look for Mut_Type or not.

    Returns:
        tuple[allel.GenotypeArray, list[tup(chrom, pos,  mut)]]: Genotype arrays and associated id information.
    """
    # Shape (snps, samps, ploidy)
    geno_arr = get_geno_arr(vcf)
    locs = make_loc_tups(vcf, benchmark)

    return geno_arr, locs


### Haplotypes from VCF
def vcf_to_haps(vcf, benchmark):
    """
    Takes in vcf file, accesses and collates data for easy haplotype dict-filling.

    Args:
        vcf_obj (allel.vcf): Loaded VCF object (whole or chunked).
        benchmark (bool): Whether to look for Mut_Type or not.

    Returns:
        tuple[allel.HaplotypeArray, list[tup(chrom, pos,  mut)]]: Haplotype arrays and associated id information.
    """
    hap_arr = get_geno_arr(vcf).to_haplotypes()
    locs = make_loc_tups(vcf, benchmark)

    return hap_arr, locs


def split_arr(arr, samp_sizes):
    """
    Restacks array to be in list of shape (timepoint_bins[snps, inds, alleles]).

    Args:
        arr (np.arr): SNP or Haplotype array with all timepoints in flat structure.
        samp_sizes (list(int)): List of chromosomes (not individuals) to index from the array.

    Returns:
        list[np.arr]: Time-serialized list of arrays of SNP or haplotype data.
    """
    i = 0
    arr_list = []
    for j in samp_sizes:
        arr_list.append(arr[:, i : i + j, :])
        i += j

    return arr_list


def get_vel_minor_alleles(ts_genos, max_allele):
    """
    Gets index of minor allele to use for MAF.
    Based on the highest-velocity minor allele at last timepoint.

    Args:
        ts_genos (np.arr): Time-series array of SNPs organized with split_arr().

    Returns:
        np.arr: Array of indices of minor alleles.
    """
    # Shape is (snps, counts)
    # Highest velocity allele wins
    last_genos = allel.GenotypeArray(ts_genos[-1]).count_alleles(max_allele=max_allele)
    first_genos = allel.GenotypeArray(ts_genos[0]).count_alleles(max_allele=max_allele)

    return np.argmax(last_genos - first_genos, axis=1)


def get_last_minor_alleles(ts_genos, max_allele):
    """
    Gets index of minor allele to use for MAF.
    Based on the highest-frequency minor allele at last timepoint.

    Args:
        ts_genos (np.arr): Time-series array of SNPs organized with split_arr().

    Returns:
        np.arr: Array of indices of minor alleles.
    """
    # Shape is (snps, counts)
    last_genos = allel.GenotypeArray(ts_genos[-1]).count_alleles(max_allele=max_allele)

    return np.argmax(last_genos, axis=1)


def calc_maft(snp, min_allele_idx):
    """
    Calculates minor allele frequency for given snp.

    Args:
        snp (np.arr): Frequencies of all minor alleles for a given SNP.
        min_allele_idx (int): Index of minor allele.

    Returns:
        float: Minor allele frequency (MAF) at a given timepoint.
    """
    return snp[min_allele_idx] / snp.sum()


def get_allele_counts(snp, min_allele_idx):
    min_counts = snp[min_allele_idx]
    maj_counts = np.sum(np.delete(snp, min_allele_idx, axis=1))

    return maj_counts, min_counts
