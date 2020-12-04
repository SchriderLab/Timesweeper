from math import sqrt
from typing import List

# https://www.genetics.org/content/196/2/509

"""
L: # Sampled points
v[i]: Allele frequency at point i in L
t[i]: Generation at point i in L
"""


def calc_Yi(v: List[float], t: List[int], L: int) -> List[float]:
    Yi_list = []
    for i in range(1, L + 1):
        Yi_list.append(
            (v[i] - v[i - 1])
            / sqrt((2 * v[i - 1]) * (1 - v[i - 1]) * (t[i] - t[i - 1]))
        )

    return Yi_list


def calc_Ybar(Yi: List[float], L: int) -> float:
    return (1 / L) * sum(Yi)


def calc_S2(Yi: List[float], Ybar: float, L: int) -> float:
    return (1 / L - 1) * sum([(i - Ybar) ** 2 for i in Yi])


def calc_FIT(Ybar: float, S2: float, L: int) -> float:
    return Ybar / sqrt(S2 / L)
