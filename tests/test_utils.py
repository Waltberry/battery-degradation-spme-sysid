import numpy as np
import pytest

from battery_deg_spme.utils.checks import assert_nonempty
from battery_deg_spme.utils.math_utils import simple_line_fit_rmse
from battery_deg_spme.utils.selection import clamp_index


def test_assert_nonempty_ok():
    assert_nonempty("x", [1, 2])


def test_assert_nonempty_raises():
    with pytest.raises(ValueError):
        assert_nonempty("x", [])


def test_simple_line_fit_rmse():
    t = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    rmse, p = simple_line_fit_rmse(t, y)
    assert rmse < 1e-12
    assert len(p) == 2


def test_clamp_index():
    assert clamp_index(5, 3) == 2
    assert clamp_index(-1, 3) == 0
    assert clamp_index(1, 3) == 1