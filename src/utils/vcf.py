import allel
import numpy as np

# General util functions
def read_vcf(vcf_file):
    vcf = allel.read_vcf(
        vcf_file, fields=["variants/CHROM", "variants/POS", "calldata/GT"]
    )

    return vcf


def get_geno_arr(vcf):
    return allel.GenotypeArray(vcf["calldata/GT"])


def make_loc_tups(vcf):
    return list(zip(vcf["variants/CHROM"], vcf["variants/POS"]))


### Get afs from vcf
def vcf_to_genos(vcf_file):
    vcf = read_vcf(vcf_file)
    geno_arr = get_geno_arr(vcf)
    locs = make_loc_tups(vcf)

    return geno_arr, locs


### Haplotypes from VCF
def vcf_to_haps(vcf_file):
    vcf = read_vcf(vcf_file)
    hap_arr = get_geno_arr(vcf).to_haplotypes()
    locs = make_loc_tups(vcf)

    return hap_arr, locs

