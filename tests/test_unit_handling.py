import numpy as np
import pytest

from battery_deg_spme.preprocessing.unit_handling import (
    convert_current_to_amps,
    guess_current_in_amps,
    sanity_report_current_units,
)


def test_sanity_report_current_units_keys():
    x = np.array([1.0, -2.0, 3.0])
    out = sanity_report_current_units(x)
    assert "if_A" in out
    assert "if_mA" in out


def test_guess_current_in_amps_large_values():
    x = np.array([0.1, -0.2, 0.15])
    out, note = guess_current_in_amps(x)
    assert out.shape == x.shape
    assert isinstance(note, str)


def test_convert_current_to_amps_ma():
    x = np.array([1000.0, -2000.0])
    out, note = convert_current_to_amps(x, "mA")
    assert np.allclose(out, np.array([1.0, -2.0]))
    assert "mA" in note


def test_convert_current_to_amps_a():
    x = np.array([1.0, -2.0])
    out, _ = convert_current_to_amps(x, "A")
    assert np.allclose(out, x)


def test_convert_current_to_amps_invalid():
    with pytest.raises(ValueError):
        convert_current_to_amps(np.array([1.0]), "bad")