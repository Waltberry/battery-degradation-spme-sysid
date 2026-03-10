import numpy as np
import pandas as pd
import pytest

from battery_deg_spme.preprocessing.signal_preparation import prepare_cycle_data


def test_prepare_cycle_data_basic():
    df = pd.DataFrame(
        {"I/mA": [-0.01, -0.01, -0.01, -0.01, -0.01], "Ewe/V": [1, 2, 3, 4, 5]},
        index=[0, 1, 2, 3, 4],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        resample=False,
    )
    assert out["t"].shape[0] == 5
    assert out["u"].shape[0] == 5
    assert out["y"].shape[0] == 5


def test_prepare_cycle_data_vref_first():
    df = pd.DataFrame(
        {"I/mA": [-1, -1, -1], "Ewe/V": [3.0, 2.5, 2.0]},
        index=[0, 1, 2],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        v_ref="first",
        resample=False,
    )
    assert np.isclose(out["y"][0, 0], 0.0)
    assert np.isclose(out["y"][1, 0], -0.5)


def test_prepare_cycle_data_vref_mean():
    df = pd.DataFrame(
        {"I/mA": [-1, -1, -1], "Ewe/V": [1.0, 2.0, 3.0]},
        index=[0, 1, 2],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        v_ref="mean",
        resample=False,
    )
    assert np.isclose(np.mean(out["y"][:, 0]), 0.0)


def test_prepare_cycle_data_drop_first_n():
    df = pd.DataFrame(
        {"I/mA": [-1, -1, -1, -1, -1], "Ewe/V": [1, 2, 3, 4, 5]},
        index=[0, 1, 2, 3, 4],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        resample=False,
        drop_first_n=2,
    )
    assert len(out["t"]) == 3
    assert np.isclose(out["t"][0], 0.0)


def test_prepare_cycle_data_positive_discharge_mode():
    df = pd.DataFrame(
        {"I/mA": [1.0, 1.0, 1.0, 0.0], "Ewe/V": [1, 2, 3, 4]},
        index=[0, 1, 2, 3],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        resample=False,
        raw_discharge_sign="positive",
    )
    assert len(out["t"]) == 3


def test_prepare_cycle_data_raises_when_too_few_after_filter():
    df = pd.DataFrame(
        {"I/mA": [0.0, 0.0, -1e-15], "Ewe/V": [1, 2, 3]},
        index=[0, 1, 2],
    )
    with pytest.raises(RuntimeError):
        prepare_cycle_data(
            cycle_df=df,
            i_col="I/mA",
            v_col="Ewe/V",
            force_units="A",
            resample=False,
        )


def test_prepare_cycle_data_resample():
    df = pd.DataFrame(
        {"I/mA": [-1.0, -1.0, -1.0], "Ewe/V": [1.0, 2.0, 3.0]},
        index=[0.0, 1.0, 2.0],
    )
    out = prepare_cycle_data(
        cycle_df=df,
        i_col="I/mA",
        v_col="Ewe/V",
        force_units="A",
        resample=True,
    )
    assert out["dt_med"] == 1.0
    assert out["u"].shape[0] == 3