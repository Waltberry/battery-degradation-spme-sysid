from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def get_previous_segment_by_iloc(
    df: pd.DataFrame,
    start_pos: int,
    n_prev_points: int = 10,
) -> pd.DataFrame:
    lo = max(0, start_pos - int(n_prev_points))
    return df.iloc[lo:start_pos].copy()


def find_discharging_cycles_with_meta(
    df: pd.DataFrame,
    i_col: str,
    tol_i: float = 1e-9,
    min_len: Optional[int] = None,
    include_previous_segment: bool = False,
    n_prev_points: int = 10,
):
    i_arr = df[i_col].to_numpy(dtype=float).reshape(-1)
    mask = i_arr < -tol_i

    cycles = []
    cycle_meta = []

    start = None
    for k in range(len(mask)):
        if mask[k] and start is None:
            start = k

        end_now = (start is not None) and ((not mask[k]) or (k == len(mask) - 1))
        if end_now:
            end = k if (not mask[k]) else (k + 1)
            seg_len_core = end - start

            keep_seg = True if min_len is None else (seg_len_core >= int(min_len))
            if keep_seg:
                core_seg = df.iloc[start:end].copy()

                if include_previous_segment:
                    prev_seg = get_previous_segment_by_iloc(
                        df,
                        start,
                        n_prev_points=n_prev_points,
                    )
                    seg = pd.concat([prev_seg, core_seg], axis=0)
                else:
                    seg = core_seg

                cycles.append(seg)

                cycle_meta.append(
                    {
                        "start_row": int(start),
                        "end_row": int(end - 1),
                        "n_points_total": int(len(seg)),
                        "n_points_core": int(seg_len_core),
                        "t_start": float(seg.index[0]),
                        "t_end": float(seg.index[-1]),
                        "duration": float(seg.index[-1] - seg.index[0]),
                        "i_mean": float(np.mean(core_seg[i_col].to_numpy(dtype=float))),
                        "i_min": float(np.min(core_seg[i_col].to_numpy(dtype=float))),
                        "i_max": float(np.max(core_seg[i_col].to_numpy(dtype=float))),
                    }
                )
            start = None

    return cycles, cycle_meta


def summarize_cycles(cycle_meta):
    if len(cycle_meta) == 0:
        raise ValueError("cycle_meta is empty")

    cycle_lengths_total = np.array([m["n_points_total"] for m in cycle_meta], dtype=int)
    cycle_lengths_core = np.array([m["n_points_core"] for m in cycle_meta], dtype=int)
    cycle_durations = np.array([m["duration"] for m in cycle_meta], dtype=float)

    return {
        "n_cycles": int(len(cycle_meta)),
        "total_len_min": int(np.min(cycle_lengths_total)),
        "total_len_median": int(np.median(cycle_lengths_total)),
        "total_len_max": int(np.max(cycle_lengths_total)),
        "core_len_min": int(np.min(cycle_lengths_core)),
        "core_len_median": int(np.median(cycle_lengths_core)),
        "core_len_max": int(np.max(cycle_lengths_core)),
        "duration_min": float(np.min(cycle_durations)),
        "duration_median": float(np.median(cycle_durations)),
        "duration_max": float(np.max(cycle_durations)),
    }


def select_cycle(
    cycles,
    cycle_meta,
    cycle_mode: str = "random",
    cycle_index: int = 0,
    random_seed: Optional[int] = 42,
):
    if len(cycles) == 0:
        raise ValueError("No cycles available for selection.")

    mode = str(cycle_mode).strip().lower()

    if mode == "index":
        chosen = int(cycle_index)
        chosen = max(0, min(chosen, len(cycles) - 1))
        note = f"index mode -> using cycle {chosen}"

    elif mode == "random":
        if random_seed is None:
            chosen = int(np.random.randint(0, len(cycles)))
            note = (
                f"random mode -> RANDOM_SEED=None, selected cycle {chosen} "
                f"(this may change from run to run)"
            )
        else:
            rng = np.random.default_rng(int(random_seed))
            chosen = int(rng.integers(low=0, high=len(cycles)))
            note = (
                f"random mode -> seed={random_seed}, selected cycle {chosen} "
                f"(reuse with cycle_mode='index', cycle_index={chosen})"
            )
    else:
        raise ValueError("cycle_mode must be 'index' or 'random'.")

    return cycles[chosen].copy(), cycle_meta[chosen], chosen, note