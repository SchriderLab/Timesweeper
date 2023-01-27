from timesweeper. import make_training_features as mtf
import numpy as np


def test_add_missingness_0():
    assert np.array_equal(mtf.add_missingness(np.ones((3, 4)), 0), np.ones((3, 4)))


def test_add_missingness_100():
    assert np.array_equal(
        mtf.add_missingness(np.ones((3, 4)), 1.0), np.ones((3, 4)) * -1
    )


def test_check_freq_increase_true():
    test_afs = np.array([[0.1, 0.5, 0.8], [0.1, 0.5, 0.8]]).T
    assert mtf.check_freq_increase(test_afs, 0.25) == True


def test_check_freq_increase_false():
    test_afs = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]).T
    assert mtf.check_freq_increase(test_afs, 0.25) == False
