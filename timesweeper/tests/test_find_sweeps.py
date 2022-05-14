import numpy as np
from ..utils.gen_utils import get_sweep, get_rep_id, add_file_label
from ..find_sweeps_vcf import get_window_idxs
import pytest

sweeps = ["neut", "hard", "soft"]


@pytest.mark.parametrize("sweep", sweeps)
def test_get_sweep(sweep):
    assert get_sweep(f"/foo/bar/{sweep}/baz") == sweep


ids = [1, 2, 3]


@pytest.mark.parametrize("id", ids)
def test_get_rep_id(id):
    assert get_rep_id(f"/foo/bar/{id}/baz")


def test_get_window_idxs():
    assert get_window_idxs(25, 51) == list(range(51))


def test_add_file_label():
    assert add_file_label("foo/bar.baz", "buzz") == "foo/bar_buzz.baz"
