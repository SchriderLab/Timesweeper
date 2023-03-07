from timesweeper.utils import frequency_increment_test as fit
import numpy as np


def test_getRescaledIncs():
    incs = fit.getRescaledIncs(list(np.linspace(0, 1, 10)), list(range(0, 100, 10)))
    assert incs == [
        0.07905694150420947,
        0.05976143046671968,
        0.05270462766947298,
        0.05000000000000002,
        0.04999999999999997,
        0.05270462766947295,
        0.0597614304667197,
    ]


def test_fit():
    assert (
        fit.fit(list(np.linspace(0, 1, 10)), list(range(0, 100, 10)))[0]
        == 14.864783097221553
    )
