from .. import make_training_features as mtf
import numpy as np


def test_add_missingness():
    assert mtf.add_missingness(np.ones((3, 4)), 0) == np.ones(3, 4)

