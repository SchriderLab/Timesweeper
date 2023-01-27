import numpy as np
from timesweeper.utils.gen_utils import (
    get_scenario_from_filename,
    get_rep_id,
    add_file_label,
    read_config,
)
from timesweeper.find_sweeps_vcf import get_window_idxs
import pytest

scenarios = ["sdn", "ssv" "neutral"]


@pytest.mark.parametrize("scenario", scenarios)
def test_get_scenario(scenario):
    assert get_scenario_from_filename(f"/foo/bar/{scenario}/baz", scenarios) == scenario


ids = [1, 2, 3]


@pytest.mark.parametrize("id", ids)
def test_get_rep_id(id):
    assert get_rep_id(f"/foo/bar/{id}/baz")


def test_get_window_idxs():
    assert get_window_idxs(25, 51) == list(range(51))


def test_add_file_label():
    assert add_file_label("foo/bar.baz", "buzz") == "foo/bar_buzz.baz"
