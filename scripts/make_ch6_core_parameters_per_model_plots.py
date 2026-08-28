#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_core_parameters_per_model_plots.py

Purpose
-------
Create one readable figure per fitted model showing the four shared core
parameters across the retained cycle window:

    alpha_n
    alpha_p
    g_n
    g_p

This is for visual inspection of each model separately, e.g.

    S7_C1
    S7_C2
    ...
    S17_C4K

Input
-----
results/tables/chapter6_core_parameter_comparison/core_parameters_long.csv

Outputs
-------
results/figures/chapter6_core_parameters_per_model/
figures/chapter6/core_parameters_per_model/

Tables:
results/tables/chapter6_core_parameter_comparison/
    core_parameter_per_model_summary.csv
    core_parameter_per_model_trend_flags.csv
"""

from __future__ import annotations

from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

IN_TABLE = PROJECT / "results" / "tables" / "chapter6_core_parameter_comparison" / "core_parameters_long.csv"

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_core_parameter_comparison"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_core_parameters_per_model"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6" / "core_parameters_per_model"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6" / "core_parameters_per_model"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

CORE_PARAMS = ["alpha_n", "alpha_p", "g_n", "g_p"]

STATE_ORDER = ["S7", "S12", "S14", "S17"]

FINAL_THESIS_MODELS = [
    "S7_C1", "S7_C2", "S7_C3", "S7_C4K",
    "S12_C1", "S12_C2", "S12_C3", "S12_C4",
    "S14_C1", "S14_C2", "S14_C3", "S14_C4",
    "S17_C1", "S17_C2", "S17_C3", "S17_C4K",
]

# Include extra plain C4 models if they exist, useful for C4 vs C4K inspection.
EXTRA_MODELS = [
    "S7_C4",
    "S17_C4",
]

MODEL_ORDER = FINAL_THESIS_MODELS + EXTRA_MODELS

FIG_DPI = 300
TITLE_SIZE = 15
AXIS_SIZE = 12
TICK_SIZE = 10
LINE_WIDTH = 1.9
MARKER_SIZE = 4.2

# Set True if you also want separate alpha_n-only, alpha_p-only, g_n-only, g_p-only
# images for every model.
MAKE_SINGLE_PARAMETER_FIGURES = True


# ============================================================
# Helpers
# ============================================================

def model_sort_key(model_id: str) -> tuple[int, int, str]:
    state = model_id.split("_")[0]
    cand = "_".join(model_id.split("_")[1:])

    state_idx = STATE_ORDER.index(state) if state in STATE_ORDER else 999

    cand_idx = {
        "C1": 0,
        "C2": 1,
        "C3": 2,
        "C4": 3,
        "C4K": 4,
    }.get(cand, 999)

    return state_idx, cand_idx, model_id


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()

    print("[saved]", path)

    try:
        shutil.copy2(path, THESIS_FIG_DIR / path.name)
    except Exception as exc:
        warnings.warn(f"Could not copy to thesis dir: {exc}")

    try:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)
    except Exception as exc:
        warnings.warn(f"Could not copy to Flow thesis dir: {exc}")


def clean_axis(ax) -> None:
    ax.grid(True, alpha=0.30)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)


def normalize_series(y: np.ndarray) -> np.ndarray:
    """
    Median/IQR normalization for trend-shape inspection.
    This makes the plot show whether the parameter rises/falls similarly,
    not whether its absolute magnitude is the same.
    """
    y = np.asarray(y, dtype=float)

    med = np.nanmedian(y)
    q25 = np.nanpercentile(y, 25)
    q75 = np.nanpercentile(y, 75)
    scale = q75 - q25

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        scale = np.nanstd(y)

    if not np.isfinite(scale) or abs(scale) < 1e-15:
        return y * np.nan

    return (y - med) / scale


def slope_per_cycle(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    try:
        return float(np.polyfit(x[mask], y[mask], 1)[0])
    except Exception:
        return np.nan


def load_core_data() -> pd.DataFrame:
    if not IN_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {IN_TABLE}. Run scripts/make_ch6_core_parameter_comparison_only.py first."
        )

    df = pd.read_csv(IN_TABLE)

    required = {
        "model_id",
        "parameter",
        "value",
        "retained_cycle_index",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"{IN_TABLE} is missing required columns: {missing}")

    df = df[df["parameter"].isin(CORE_PARAMS)].copy()
    df = df[df["model_id"].isin(MODEL_ORDER)].copy()

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["retained_cycle_index"] = pd.to_numeric(df["retained_cycle_index"], errors="coerce")

    if "cycle_index" in df.columns:
        df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce")

    df = df.dropna(subset=["value", "retained_cycle_index"]).copy()

    df = df.sort_values(["model_id", "parameter", "retained_cycle_index"]).reset_index(drop=True)

    return df


# ============================================================
# Summary metrics
# ============================================================

def make_per_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model_id, parameter), g in df.groupby(["model_id", "parameter"]):
        g = g.sort_values("retained_cycle_index")

        x = g["retained_cycle_index"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        n = int(np.isfinite(y).sum())

        median = float(np.nanmedian(y)) if n else np.nan
        mean = float(np.nanmean(y)) if n else np.nan
        std = float(np.nanstd(y, ddof=1)) if n > 1 else np.nan
        q25 = float(np.nanpercentile(y, 25)) if n else np.nan
        q75 = float(np.nanpercentile(y, 75)) if n else np.nan
        ymin = float(np.nanmin(y)) if n else np.nan
        ymax = float(np.nanmax(y)) if n else np.nan

        cv_abs = abs(std) / max(abs(median), 1e-15) if np.isfinite(std) and np.isfinite(median) else np.nan
        slope = slope_per_cycle(x, y)

        # crude start-to-end change
        if n >= 2:
            first = float(y[np.isfinite(y)][0])
            last = float(y[np.isfinite(y)][-1])
            denom = max(abs(first), abs(last), 1e-15)
            start_end_rel_change = abs(last - first) / denom
        else:
            first = np.nan
            last = np.nan
            start_end_rel_change = np.nan

        rows.append(
            {
                "model_id": model_id,
                "parameter": parameter,
                "n_cycles": n,
                "mean_value": mean,
                "median_value": median,
                "std_value": std,
                "cv_abs": cv_abs,
                "q25_value": q25,
                "q75_value": q75,
                "min_value": ymin,
                "max_value": ymax,
                "first_value": first,
                "last_value": last,
                "start_end_rel_change": start_end_rel_change,
                "slope_per_retained_cycle": slope,
            }
        )

    out = pd.DataFrame(rows)

    out["model_sort"] = out["model_id"].map(lambda m: model_sort_key(m)[0] * 10 + model_sort_key(m)[1])
    out["param_sort"] = out["parameter"].map({p: i for i, p in enumerate(CORE_PARAMS)})

    out = out.sort_values(["model_sort", "param_sort"]).drop(columns=["model_sort", "param_sort"]).reset_index(drop=True)

    return out


def make_trend_flags(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Simple diagnostic labels for thesis discussion.
    These are not statistical tests; they are descriptive flags.
    """
    out = summary.copy()

    def flag_row(row):
        cv = row["cv_abs"]
        rel = row["start_end_rel_change"]
        slope = row["slope_per_retained_cycle"]

        flags = []

        if np.isfinite(cv):
            if cv < 0.05:
                flags.append("low_cycle_variation")
            elif cv < 0.20:
                flags.append("moderate_cycle_variation")
            else:
                flags.append("high_cycle_variation")

        if np.isfinite(rel):
            if rel < 0.05:
                flags.append("stable_start_to_end")
            elif rel < 0.20:
                flags.append("moderate_start_to_end_change")
            else:
                flags.append("large_start_to_end_change")

        if np.isfinite(slope):
            if abs(slope) < 1e-12:
                flags.append("near_zero_linear_slope")
            elif slope > 0:
                flags.append("positive_linear_trend")
            else:
                flags.append("negative_linear_trend")

        return "; ".join(flags)

    out["trend_flags"] = out.apply(flag_row, axis=1)

    return out


# ============================================================
# Plotting
# ============================================================

def plot_model_four_core_params(df: pd.DataFrame, model_id: str, normalized: bool = False) -> None:
    d = df[df["model_id"] == model_id].copy()

    if len(d) == 0:
        return

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10.8, 10.5),
        sharex=True,
    )

    for ax, parameter in zip(axes, CORE_PARAMS):
        g = d[d["parameter"] == parameter].sort_values("retained_cycle_index")

        if len(g) == 0:
            ax.set_ylabel(parameter, fontsize=AXIS_SIZE)
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center")
            clean_axis(ax)
            continue

        x = g["retained_cycle_index"].to_numpy(dtype=float)
        y = g["value"].to_numpy(dtype=float)

        if normalized:
            y = normalize_series(y)

        ax.plot(
            x,
            y,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
        )

        ax.set_ylabel(parameter if not normalized else f"{parameter}\nnormalized", fontsize=AXIS_SIZE)
        clean_axis(ax)

    title_suffix = "Normalized Core Parameter Trends" if normalized else "Core Parameter Estimates"
    axes[0].set_title(f"{model_id}: {title_suffix}", fontsize=TITLE_SIZE, pad=12)
    axes[-1].set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)

    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_params_{model_id}_{suffix}.png")


def plot_model_single_parameter(df: pd.DataFrame, model_id: str, parameter: str, normalized: bool = False) -> None:
    d = df[
        (df["model_id"] == model_id)
        & (df["parameter"] == parameter)
    ].copy()

    if len(d) == 0:
        return

    d = d.sort_values("retained_cycle_index")

    x = d["retained_cycle_index"].to_numpy(dtype=float)
    y = d["value"].to_numpy(dtype=float)

    if normalized:
        y = normalize_series(y)

    fig, ax = plt.subplots(figsize=(9.8, 5.2))

    ax.plot(
        x,
        y,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
    )

    title_suffix = "normalized trend" if normalized else "estimated value"

    ax.set_title(f"{model_id}: {parameter} ({title_suffix})", fontsize=TITLE_SIZE, pad=12)
    ax.set_xlabel("Retained cycle index", fontsize=AXIS_SIZE)
    ax.set_ylabel(f"{parameter} {'[normalized]' if normalized else ''}", fontsize=AXIS_SIZE)

    clean_axis(ax)
    fig.tight_layout()

    suffix = "normalized" if normalized else "raw"
    savefig(OUT_FIG_DIR / f"fig_ch6_core_{model_id}_{parameter}_{suffix}.png")


def main() -> None:
    print("=" * 100)
    print("PER-MODEL CORE PARAMETER PLOTS")
    print("=" * 100)
    print("Input:", IN_TABLE)
    print("Output figures:", OUT_FIG_DIR)
    print("Output tables:", OUT_TABLE_DIR)
    print("=" * 100)

    df = load_core_data()

    clean_path = OUT_TABLE_DIR / "core_parameters_long_clean_per_model.csv"
    df.to_csv(clean_path, index=False)
    print("[saved]", clean_path)

    summary = make_per_model_summary(df)

    summary_path = OUT_TABLE_DIR / "core_parameter_per_model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("[saved]", summary_path)

    flags = make_trend_flags(summary)

    flags_path = OUT_TABLE_DIR / "core_parameter_per_model_trend_flags.csv"
    flags.to_csv(flags_path, index=False)
    print("[saved]", flags_path)

    models = [m for m in MODEL_ORDER if m in set(df["model_id"])]

    print()
    print("Models found:")
    for m in models:
        n_cycles = df[df["model_id"] == m]["retained_cycle_index"].nunique()
        n_params = df[df["model_id"] == m]["parameter"].nunique()
        print(f"  {m:10s} cycles={n_cycles:3d} core_params={n_params}")

    for model_id in models:
        plot_model_four_core_params(df, model_id, normalized=False)
        plot_model_four_core_params(df, model_id, normalized=True)

        if MAKE_SINGLE_PARAMETER_FIGURES:
            for parameter in CORE_PARAMS:
                plot_model_single_parameter(df, model_id, parameter, normalized=False)
                plot_model_single_parameter(df, model_id, parameter, normalized=True)

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Main per-model figures:")
    for model_id in models:
        print(" ", OUT_FIG_DIR / f"fig_ch6_core_params_{model_id}_raw.png")
        print(" ", OUT_FIG_DIR / f"fig_ch6_core_params_{model_id}_normalized.png")
    print()
    print("Tables:")
    print(" ", summary_path)
    print(" ", flags_path)
    print("=" * 100)


if __name__ == "__main__":
    main()
