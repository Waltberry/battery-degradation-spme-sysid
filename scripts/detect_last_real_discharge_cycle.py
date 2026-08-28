#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
detect_last_real_discharge_cycle.py

Detect the number of discharge cycles directly from the same real MPR data
used by the CT-ID run scripts.

This avoids depending on detected_cycle_metadata.csv from previous result folders.
"""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.io.data_io import load_mpr_as_dataframe
from battery_deg_spme.preprocessing.cycle_detection import find_discharging_cycles_with_meta


PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

OUT_DIR = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/cycle_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    settings = get_default_settings()

    mpr_path_env = os.environ.get("UN_MPR_PATH", "").strip()
    if mpr_path_env:
        settings.data.mpr_path = mpr_path_env

    print("=" * 100)
    print("DETECT REAL DISCHARGE CYCLES")
    print("=" * 100)
    print("MPR path:")
    print(" ", settings.data.mpr_path)

    df_raw = load_mpr_as_dataframe(
        mpr_path=settings.data.mpr_path,
        time_col=settings.data.time_col,
        i_col=settings.data.i_col,
        v_col=settings.data.v_col,
    )

    min_len_for_search = (
        int(settings.cycle.min_cycle_len)
        if getattr(settings.cycle, "use_min_cycle_len", False)
        else None
    )

    cycles, cycle_meta = find_discharging_cycles_with_meta(
        df=df_raw,
        i_col=settings.data.i_col,
        tol_i=1e-9,
        min_len=min_len_for_search,
        include_previous_segment=settings.cycle.include_previous_segment,
        n_prev_points=settings.cycle.n_prev_points,
    )

    n_cycles = len(cycles)
    last_cycle_index = n_cycles - 1

    meta_df = pd.DataFrame(cycle_meta)
    meta_path = OUT_DIR / "detected_cycle_metadata_from_raw_data.csv"
    meta_df.to_csv(meta_path, index=False)

    summary_path = OUT_DIR / "detected_cycle_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"n_cycles={n_cycles}\n")
        f.write(f"last_cycle_index={last_cycle_index}\n")
        f.write(f"mpr_path={settings.data.mpr_path}\n")
        f.write(f"metadata_csv={meta_path}\n")

    print()
    print("Detected discharge cycles:", n_cycles)
    print("Last cycle index:", last_cycle_index)
    print("[saved]", meta_path)
    print("[saved]", summary_path)
    print("=" * 100)

    # Important: print only the number on the final line so bash can capture it.
    print(last_cycle_index)


if __name__ == "__main__":
    main()