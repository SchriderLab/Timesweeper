from scipy import stats
import numpy as np
import snp_utils as su


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


def test_fet():
    maj_counts = np.random.randint(0, 20, (20))
    min_counts = np.random.randint(0, 5, (20))
    fish_res = fet(maj_counts, min_counts)
    print(fish_res)


def main():
    test_fet()


if __name__ == "__main__":
    main()
