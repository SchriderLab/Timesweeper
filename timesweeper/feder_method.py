from math import sqrt
from typing import List, Tuple
import random as rd
import sys

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
    max_gen = rd.randint(10, 100)  # Total num time samples
    test_t_num = rd.randint(3, 10)  # For testing just take evenly-spaced timepoints
    test_t = [i for i in range(max_gen) if i % test_t_num == 0]
    test_L = len(test_t)
    test_v = [
        rd.random() for i in range(test_L)
    ]  # Generate frequencies for each timepoint

    return test_v, test_t, test_L


def main():
    v, t, L = gen_test_data(42)
    print("freqs:", v)
    print("timepoint samps:", t)
    print("max gens:", L)

    Yi = calc_Yi(v, t, L)
    print("Yi:", Yi)

    Ybar = calc_Ybar(Yi, L)
    print("Ybar:", Ybar)

    S2 = calc_S2(Yi, Ybar, L)
    print("S2:", S2)

    FIT = calc_FIT(Ybar, S2, L)
    print("FIT Value:", FIT)


if __name__ == "__main__":
    main()