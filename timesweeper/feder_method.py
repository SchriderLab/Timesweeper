from math import sqrt
from typing import List, Tuple
import random as rd
import sys
import numpy as np

# https://www.genetics.org/content/196/2/509

"""
L: Number of sampled points
v[i]: Allele frequency at point i in L
t[i]: Generation at point i in L
"""


def calc_Yi(v: List[float], t: List[int], L: int) -> List[float]:
    Yi_list = []
    for i in range(1, L):
        Yi_list.append(
            (v[i] - v[i - 1])
            / sqrt((2 * v[i - 1]) * (1 - v[i - 1]) * (t[i] - t[i - 1]))
        )

    return Yi_list


def calc_Ybar(Yi: List[float], L: int) -> float:
    return np.mean(Yi)


def calc_S2(Yi: List[float], Ybar: float, L: int) -> float:
    sumpart = sum([((i - Ybar) ** 2) for i in Yi])

    return (1 / (L - 1)) / sumpart


def calc_FIT(Ybar: float, S2: float, L: int) -> float:
    return Ybar / sqrt(S2 / L)


def gen_test_data(test_seed: int) -> Tuple[List[float], List[int], int]:
    rd.seed(test_seed)
    test_L = 10
    test_t_lim = 1000
    test_t = [i for i in range(0, test_t_lim + 1) if i % (test_t_lim / test_L) == 0]
    test_v = list(np.random.normal(loc=0.5, scale=0.1, size=test_L))

    return test_v, test_t, test_L


def main():
    # v, t, L = gen_test_data(42)
    v = [0.06, 0.03, 0.061, 0.091, 0.236, 0.252, 0.345, 0.492, 0.679, 0.818]
    t = [10060, 10080, 10100, 10120, 10140, 10160, 10180, 10200, 10220, 10240]
    L = len(t)

    print("\nFreqs:", v)
    print("\nTimepoint samps:", t)
    print("\nNumber of timepoints sampled:", L)

    Yi = calc_Yi(v, t, L)
    print("\nYi:", Yi)

    Ybar = calc_Ybar(Yi, L)
    print("\nYbar:", Ybar)

    S2 = calc_S2(Yi, Ybar, L)
    print("\nS2:", S2)

    FIT = calc_FIT(Ybar, S2, L)
    print("\nFIT Value:", FIT)


if __name__ == "__main__":
    main()