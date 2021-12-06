import allel
import numpy as np

### Get afs from vcf
def vcf_to_afs(vcf_file):
    vcf = read_vcf(vcf_file)
    geno_arr = get_geno_arr(vcf)
    mafs = calc_mafs(geno_arr)
    locs = make_loc_tups(vcf)
    freqs_dict = annotate_counts(mafs, locs)

    return freqs_dict


def read_vcf(vcf_file):
    vcf = allel.read_vcf(
        vcf_file, fields=["variants/CHROM", "variants/POS", "calldata/GT"]
    )

    return vcf


def get_geno_arr(vcf):
    return allel.GenotypeArray(vcf["calldata/GT"])


def calc_mafs(geno_arr):
    return geno_arr.count_alleles()[:, 1] / geno_arr.count_alleles().sum(axis=1)


def make_loc_tups(vcf):
    return zip(vcf["variants/CHROM"], vcf["variants/POS"])


def annotate_counts(minor_allele_counts, loc_keys):
    """Generates dict in the form of {(chrom, bp): freq}"""
    return dict(zip(loc_keys, minor_allele_counts))


### Merge timepoints
def dicts_to_keyset(freq_dicts):
    keys = []
    for i in freq_dicts:
        keys.extend([*i])

    return set(keys)


def create_afm(freq_dicts):
    snps = dicts_to_keyset(freq_dicts)
    afs = []
    for snp in snps:
        _freqs = []
        for sample in freq_dicts:
            if snp in sample:
                _freqs.append(sample[snp])
            else:
                _freqs.append(0.0)
        afs.append(_freqs)

    return snps, np.array(afs)
