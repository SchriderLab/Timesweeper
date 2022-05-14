from timesweeper.utils import snp_utils as su
import numpy as np
import allel

# fmt: off
g = allel.GenotypeArray([[[0, 0], [0, 1]],
                         [[0, 1], [1, 1]],
                         [[0, 2], [-1, -1]]], dtype='i1')
                         
geno_list = [
    allel.GenotypeArray([[[0, 0]], [[0, 1]], [[0, 2]]],),
    allel.GenotypeArray([[[0, 1]], [[1, 1]], [[-1, -1]]],),
]

# fmt: on
def test_split_arr():
    assert np.array_equal(su.split_arr(g, [1, 1]), np.array(geno_list),)


def test_get_vel_minor_alleles():
    assert np.array_equal(su.get_vel_minor_alleles(geno_list, 1), np.array([1, 1, 1]))


def test_get_last_minor_alleles():
    assert np.array_equal(su.get_last_minor_alleles(geno_list, 1), np.array([0, 1, 0]))


def test_calc_maft():
    assert su.calc_maft(geno_list[0].count_alleles(2)[0], 1) == 0.0
    assert su.calc_maft(geno_list[0].count_alleles(2)[1], 1) == 0.5
    assert su.calc_maft(geno_list[0].count_alleles(2)[2], 2) == 0.5
