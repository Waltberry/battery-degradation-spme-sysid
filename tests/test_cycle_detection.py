import pandas as pd

from battery_deg_spme.preprocessing.cycle_detection import (
    find_discharging_cycles_with_meta,
    get_previous_segment_by_iloc,
    select_cycle,
    summarize_cycles,
)


def test_find_discharging_cycles_with_meta_basic():
    df = pd.DataFrame(
        {"I/mA": [0.1, -0.2, -0.3, 0.0, -0.4, -0.5]},
        index=[0, 1, 2, 3, 4, 5],
    )
    cycles, meta = find_discharging_cycles_with_meta(df, i_col="I/mA", tol_i=1e-9)
    assert len(cycles) == 2
    assert len(meta) == 2
    assert meta[0]["start_row"] == 1
    assert meta[0]["end_row"] == 2
    assert meta[1]["start_row"] == 4
    assert meta[1]["end_row"] == 5


def test_find_discharging_cycles_with_min_len():
    df = pd.DataFrame(
        {"I/mA": [0.0, -1.0, -1.0, 0.0, -2.0, 0.0]},
        index=[0, 1, 2, 3, 4, 5],
    )
    cycles, meta = find_discharging_cycles_with_meta(
        df,
        i_col="I/mA",
        tol_i=1e-9,
        min_len=2,
    )
    assert len(cycles) == 1
    assert len(meta) == 1
    assert meta[0]["n_points_core"] == 2


def test_get_previous_segment_by_iloc():
    df = pd.DataFrame({"I/mA": [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])
    seg = get_previous_segment_by_iloc(df, start_pos=3, n_prev_points=2)
    assert len(seg) == 2
    assert list(seg.index) == [1, 2]


def test_include_previous_segment():
    df = pd.DataFrame(
        {"I/mA": [0.5, 0.4, -0.1, -0.2, -0.3]},
        index=[0, 1, 2, 3, 4],
    )
    cycles, meta = find_discharging_cycles_with_meta(
        df,
        i_col="I/mA",
        include_previous_segment=True,
        n_prev_points=2,
    )
    assert len(cycles) == 1
    assert len(cycles[0]) == 5
    assert meta[0]["n_points_core"] == 3


def test_summarize_cycles():
    meta = [
        {"n_points_total": 10, "n_points_core": 8, "duration": 5.0},
        {"n_points_total": 20, "n_points_core": 15, "duration": 9.0},
    ]
    out = summarize_cycles(meta)
    assert out["n_cycles"] == 2
    assert out["total_len_min"] == 10
    assert out["total_len_max"] == 20


def test_select_cycle_index_mode():
    cycles = [pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})]
    meta = [{"duration": 1.0}, {"duration": 2.0}]
    cycle_df, sel_meta, chosen, note = select_cycle(
        cycles=cycles,
        cycle_meta=meta,
        cycle_mode="index",
        cycle_index=1,
    )
    assert chosen == 1
    assert sel_meta["duration"] == 2.0
    assert "index mode" in note.lower()


def test_select_cycle_random_mode_seeded():
    cycles = [pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}), pd.DataFrame({"x": [3]})]
    meta = [{"duration": 1.0}, {"duration": 2.0}, {"duration": 3.0}]
    _, _, chosen1, _ = select_cycle(
        cycles=cycles,
        cycle_meta=meta,
        cycle_mode="random",
        random_seed=42,
    )
    _, _, chosen2, _ = select_cycle(
        cycles=cycles,
        cycle_meta=meta,
        cycle_mode="random",
        random_seed=42,
    )
    assert chosen1 == chosen2