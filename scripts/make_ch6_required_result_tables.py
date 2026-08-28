#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_required_result_tables.py

Creates the remaining Chapter 6 required tables:

1. model_complexity_summary.csv
2. fit_quality_by_cycle.csv
3. good_fit_cycles.csv
4. bad_fit_cycles.csv
5. good_fit_mask_definition.txt
6. generalizability_rmse_by_regime.csv
7. real_data_experiment_setup_table.csv
8. real_data_cycle_window_summary.csv

Generalizability uses the final S17_C4K model over all available fitted
cycles, not random cycles.

Selected retained thesis window:
    cycles 34--99

Generalizability regimes:
    A: cycles 0--33
    B: cycles 34--99
    C: cycles 100--199
    D: cycles 200--268
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

FULL16_TABLE_DIR = PROJECT / "results" / "tables" / "full16_model_complexity"
ANCHOR_TABLE_DIR = PROJECT / "results" / "tables" / "anchor_model_screening_6models"

GRID_ROOT = PROJECT / "results" / "real_cycle_ctid_state_order_grid"
KPARAM_ROOT = PROJECT / "results" / "real_warm_continuation_ctid"

OUT_TABLE_DIR = PROJECT / "results" / "tables" / "chapter6_required_outputs"
OUT_FIG_DIR = PROJECT / "results" / "figures" / "chapter6_required_outputs"

THESIS_FIG_DIR = PROJECT / "figures" / "chapter6"
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [OUT_TABLE_DIR, OUT_FIG_DIR, THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


SELECTED_CYCLE_START = 34
SELECTED_CYCLE_END = 99
SELECTED_CYCLES = list(range(SELECTED_CYCLE_START, SELECTED_CYCLE_END + 1))

FINAL_MODEL_ID = "S17_C4K"

GOOD_RMSE_V = 0.002
GOOD_BFR_PERCENT = 98.0
GOOD_R2_PERCENT = 99.95

STATE_ORDER = ["S7", "S12", "S14", "S17"]
DISPLAY_ORDER = ["C1", "C2", "C3", "C4/C4K"]

REGIMES = [
    {
        "regime_id": "A",
        "regime_name": "early pre-retained region",
        "cycle_min": 0,
        "cycle_max": 33,
        "interpretation": "Earlier out-of-window region before the selected repeated-discharge window.",
    },
    {
        "regime_id": "B",
        "regime_name": "selected retained cycling window",
        "cycle_min": 34,
        "cycle_max": 99,
        "interpretation": "Main repeated-discharge window used for model-complexity screening and parameter-trend analysis.",
    },
    {
        "regime_id": "C",
        "regime_name": "later post-retained region",
        "cycle_min": 100,
        "cycle_max": 199,
        "interpretation": "Later out-of-window region after the retained window.",
    },
    {
        "regime_id": "D",
        "regime_name": "late degradation/failure-approach region",
        "cycle_min": 200,
        "cycle_max": 268,
        "interpretation": "Late-cycle region closer to cell degradation/failure, used to test whether the selected structure remains reliable.",
    },
]


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("[saved figure]", path)

    for target_dir in [THESIS_FIG_DIR, FLOW_THESIS_FIG_DIR]:
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target_dir / path.name)
        except Exception as exc:
            warnings.warn(f"Could not copy {path} to {target_dir}: {exc}")


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def standardize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {}

    if "rmse" in out.columns and "best_rmse" not in out.columns:
        rename_map["rmse"] = "best_rmse"
    if "mae" in out.columns and "best_mae" not in out.columns:
        rename_map["mae"] = "best_mae"
    if "r2_percent" in out.columns and "best_r2_percent" not in out.columns:
        rename_map["r2_percent"] = "best_r2_percent"
    if "bfr_percent" in out.columns and "best_bfr_percent" not in out.columns:
        rename_map["bfr_percent"] = "best_bfr_percent"

    if rename_map:
        out = out.rename(columns=rename_map)

    for col in [
        "best_rmse",
        "best_mae",
        "best_bfr_percent",
        "best_r2_percent",
        "median_rmse",
        "mean_rmse",
        "std_rmse",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["best_rmse_mV"] = 1000.0 * out["best_rmse"]
    out["best_mae_mV"] = 1000.0 * out["best_mae"]
    out["median_rmse_mV"] = 1000.0 * out["median_rmse"]
    out["mean_rmse_mV"] = 1000.0 * out["mean_rmse"]

    return out


def classify_good_fit(row: pd.Series) -> tuple[bool, str]:
    rmse = float(row.get("best_rmse", np.nan))
    bfr = float(row.get("best_bfr_percent", np.nan))
    r2 = float(row.get("best_r2_percent", np.nan))

    reasons = []

    if not np.isfinite(rmse):
        reasons.append("missing_or_nonfinite_rmse")
    elif rmse > GOOD_RMSE_V:
        reasons.append(f"rmse_above_{1000.0 * GOOD_RMSE_V:.1f}_mV")

    if not np.isfinite(bfr):
        reasons.append("missing_or_nonfinite_bfr")
    elif bfr < GOOD_BFR_PERCENT:
        reasons.append(f"bfr_below_{GOOD_BFR_PERCENT:.2f}_percent")

    if not np.isfinite(r2):
        reasons.append("missing_or_nonfinite_r2")
    elif r2 < GOOD_R2_PERCENT:
        reasons.append(f"r2_below_{GOOD_R2_PERCENT:.2f}_percent")

    is_good = len(reasons) == 0
    reason = "kept_good_fit" if is_good else "; ".join(reasons)

    return is_good, reason


def assign_regime(cycle: int) -> str:
    for r in REGIMES:
        if r["cycle_min"] <= cycle <= r["cycle_max"]:
            return r["regime_name"]
    return "outside_defined_regimes"


def load_full16_summary() -> pd.DataFrame:
    paths = [
        FULL16_TABLE_DIR / "model_complexity_summary_full16.csv",
        ANCHOR_TABLE_DIR / "model_complexity_summary_selected4.csv",
        ANCHOR_TABLE_DIR / "model_complexity_summary_anchor.csv",
    ]

    p = first_existing(paths)

    if p is None:
        raise FileNotFoundError(
            "Could not find model-complexity summary. Expected one of:\n"
            + "\n".join(str(x) for x in paths)
        )

    print("[loaded model summary]", p)
    return pd.read_csv(p)


def load_full16_long() -> pd.DataFrame:
    paths = [
        FULL16_TABLE_DIR / "full16_cycle_metrics_long.csv",
        ANCHOR_TABLE_DIR / "anchor_cycle_metrics_long.csv",
    ]

    p = first_existing(paths)

    if p is None:
        raise FileNotFoundError(
            "Could not find long model-complexity metrics. Expected one of:\n"
            + "\n".join(str(x) for x in paths)
        )

    print("[loaded long metrics]", p)
    return standardize_metric_columns(pd.read_csv(p))


def make_required_model_complexity_summary() -> pd.DataFrame:
    summary = load_full16_summary()
    long = load_full16_long()

    if "display_state" not in summary.columns:
        if "state_id" in summary.columns:
            summary["display_state"] = summary["state_id"]
        elif "selected_heatmap_row" in summary.columns:
            summary["display_state"] = summary["selected_heatmap_row"]
        else:
            summary["display_state"] = ""

    if "display_order" not in summary.columns:
        if "candidate_id" in summary.columns:
            summary["display_order"] = summary["candidate_id"]
        elif "selected_heatmap_col" in summary.columns:
            summary["display_order"] = summary["selected_heatmap_col"]
        else:
            summary["display_order"] = ""

    if "display_model_id" not in summary.columns:
        if "model_id" in summary.columns:
            summary["display_model_id"] = summary["model_id"]
        else:
            summary["display_model_id"] = ""

    good_info = long.apply(classify_good_fit, axis=1)
    long["is_good_fit"] = [x[0] for x in good_info]

    good_fraction = (
        long.groupby("model_id", as_index=False)
        .agg(
            n_cycles_attempted=("cycle_index", "nunique"),
            n_cycles_successful=("best_rmse", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
            good_fit_fraction=("is_good_fit", "mean"),
        )
    )

    rows = []

    for _, row in summary.iterrows():
        model_id = str(row.get("display_model_id", row.get("model_id", "")))
        dgood = good_fraction[good_fraction["model_id"].astype(str) == model_id]

        if len(dgood):
            n_attempted = int(dgood.iloc[0]["n_cycles_attempted"])
            n_successful = int(dgood.iloc[0]["n_cycles_successful"])
            good_fit_fraction = float(dgood.iloc[0]["good_fit_fraction"])
        else:
            n_attempted = int(row.get("n_cycles", 0)) if pd.notna(row.get("n_cycles", np.nan)) else 0
            n_successful = n_attempted
            good_fit_fraction = np.nan

        rows.append(
            {
                "state_dim": row.get("display_state", row.get("state_id", "")),
                "voltage_order": row.get("display_order", row.get("candidate_id", "")),
                "model_id": model_id,
                "n_cycles_attempted": n_attempted,
                "n_cycles_successful": n_successful,
                "mean_rmse_mV": row.get("mean_best_rmse_mV", np.nan),
                "median_rmse_mV": row.get("median_best_rmse_mV", np.nan),
                "std_rmse_mV": row.get("std_best_rmse_mV", np.nan),
                "min_rmse_mV": row.get("min_best_rmse_mV", np.nan),
                "max_rmse_mV": row.get("max_best_rmse_mV", np.nan),
                "mean_bfr_percent": row.get("mean_bfr_percent", np.nan),
                "median_bfr_percent": row.get("median_bfr_percent", np.nan),
                "mean_r2_percent": row.get("mean_r2_percent", np.nan),
                "median_r2_percent": row.get("median_r2_percent", np.nan),
                "good_fit_fraction": good_fit_fraction,
            }
        )

    out = pd.DataFrame(rows)

    state_sort = {s: i for i, s in enumerate(STATE_ORDER)}
    order_sort = {s: i for i, s in enumerate(DISPLAY_ORDER)}

    out["state_sort"] = out["state_dim"].map(state_sort).fillna(999).astype(int)
    out["order_sort"] = out["voltage_order"].map(order_sort).fillna(999).astype(int)
    out = out.sort_values(["state_sort", "order_sort", "model_id"]).drop(columns=["state_sort", "order_sort"])

    out_path = OUT_TABLE_DIR / "model_complexity_summary.csv"
    out.to_csv(out_path, index=False)
    print("[saved]", out_path)

    return out


def load_final_model_selected_window_metrics() -> pd.DataFrame:
    candidates = []

    try:
        long = load_full16_long()
        d = long[long["model_id"].astype(str).eq(FINAL_MODEL_ID)].copy()
        if len(d):
            candidates.append(d)
    except Exception as exc:
        warnings.warn(f"Could not load final model from full16 long metrics: {exc}")

    all_cycles_paths = [
        KPARAM_ROOT / FINAL_MODEL_ID / "all_cycles_summary.csv",
        PROJECT / "results" / "tables" / "real_warm_continuation_ctid" / FINAL_MODEL_ID / "all_cycles_summary.csv",
    ]

    for p in all_cycles_paths:
        d2 = safe_read_csv(p)
        if d2 is not None and len(d2):
            d2 = standardize_metric_columns(d2)
            d2["model_id"] = FINAL_MODEL_ID
            candidates.append(d2)

    if not candidates:
        raise FileNotFoundError(f"Could not find cycle metrics for {FINAL_MODEL_ID}.")

    df = pd.concat(candidates, ignore_index=True)

    if "cycle_index" not in df.columns:
        raise ValueError("Final model cycle metrics are missing cycle_index column.")

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cycle_index"]).copy()
    df["cycle_index"] = df["cycle_index"].astype(int)

    df = df[df["cycle_index"].between(SELECTED_CYCLE_START, SELECTED_CYCLE_END)].copy()
    df = df.sort_values(["cycle_index", "best_rmse"], na_position="last")
    df = df.drop_duplicates(subset=["cycle_index"], keep="first").reset_index(drop=True)

    return df


def make_fit_quality_tables() -> pd.DataFrame:
    df = load_final_model_selected_window_metrics()

    good_info = df.apply(classify_good_fit, axis=1)
    df["is_good_fit"] = [x[0] for x in good_info]
    df["reason_if_excluded"] = [x[1] for x in good_info]

    out = pd.DataFrame(
        {
            "cycle_index": df["cycle_index"].astype(int),
            "model_id": FINAL_MODEL_ID,
            "rmse_V": df["best_rmse"],
            "rmse_mV": df["best_rmse_mV"],
            "bfr_percent": df["best_bfr_percent"],
            "r2_percent": df["best_r2_percent"],
            "mae_mV": df.get("best_mae_mV", np.nan),
            "is_good_fit": df["is_good_fit"],
            "reason_if_excluded": df["reason_if_excluded"],
        }
    ).sort_values("cycle_index")

    fit_path = OUT_TABLE_DIR / "fit_quality_by_cycle.csv"
    good_path = OUT_TABLE_DIR / "good_fit_cycles.csv"
    bad_path = OUT_TABLE_DIR / "bad_fit_cycles.csv"
    definition_path = OUT_TABLE_DIR / "good_fit_mask_definition.txt"

    out.to_csv(fit_path, index=False)
    out[out["is_good_fit"]].to_csv(good_path, index=False)
    out[~out["is_good_fit"]].to_csv(bad_path, index=False)

    with open(definition_path, "w", encoding="utf-8") as f:
        f.write("Good-fit mask definition for Chapter 6\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Final model: {FINAL_MODEL_ID}\n")
        f.write(f"Selected cycle window: {SELECTED_CYCLE_START}--{SELECTED_CYCLE_END} inclusive\n\n")
        f.write("A retained cycle is classified as a good fit if all of the following hold:\n")
        f.write(f"  1. best RMSE <= {1000.0 * GOOD_RMSE_V:.3f} mV\n")
        f.write(f"  2. BFR >= {GOOD_BFR_PERCENT:.3f} %\n")
        f.write(f"  3. R^2 >= {GOOD_R2_PERCENT:.3f} %\n\n")
        f.write("Cycles that fail one or more criteria are still reported in fit_quality_by_cycle.csv.\n")
        f.write("The mask is used only to identify cycles with consistently strong voltage fits for parameter-trend discussion.\n")
        f.write("It is not used to remove cycles from model-complexity screening.\n\n")
        f.write(f"Number of cycles evaluated: {len(out)}\n")
        f.write(f"Number of good-fit cycles: {int(out['is_good_fit'].sum())}\n")
        f.write(f"Good-fit fraction: {float(out['is_good_fit'].mean()):.6f}\n")

    print("[saved]", fit_path)
    print("[saved]", good_path)
    print("[saved]", bad_path)
    print("[saved]", definition_path)

    return out


def load_all_cycle_metrics_for_generalizability() -> pd.DataFrame:
    paths = [
        PROJECT / "results" / "tables" / "real_warm_continuation_ctid" / FINAL_MODEL_ID / "all_cycles_summary.csv",
        KPARAM_ROOT / FINAL_MODEL_ID / "all_cycles_summary.csv",
    ]

    frames = []

    for p in paths:
        d = safe_read_csv(p)
        if d is None or len(d) == 0:
            continue

        d = standardize_metric_columns(d)

        if "model_id" in d.columns:
            d = d[d["model_id"].astype(str).eq(FINAL_MODEL_ID)].copy()
        else:
            d["model_id"] = FINAL_MODEL_ID

        if len(d):
            d["source_file"] = str(p)
            frames.append(d)

    if not frames:
        raise FileNotFoundError(f"No all-cycle summary found for {FINAL_MODEL_ID}.")

    df = pd.concat(frames, ignore_index=True)

    if "cycle_index" not in df.columns:
        raise ValueError("Generalizability input is missing cycle_index column.")

    df["cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["cycle_index"]).copy()
    df["cycle_index"] = df["cycle_index"].astype(int)

    df = df.sort_values(["cycle_index", "best_rmse"], na_position="last")
    df = df.drop_duplicates(subset=["cycle_index"], keep="first").reset_index(drop=True)

    good_info = df.apply(classify_good_fit, axis=1)
    df["is_good_fit"] = [x[0] for x in good_info]
    df["reason_if_excluded"] = [x[1] for x in good_info]
    df["regime_name"] = df["cycle_index"].apply(assign_regime)

    return df


def make_generalizability_table() -> pd.DataFrame:
    df = load_all_cycle_metrics_for_generalizability()

    rows = []

    for r in REGIMES:
        d = df[df["cycle_index"].between(r["cycle_min"], r["cycle_max"])].copy()

        rows.append(
            {
                "regime_id": r["regime_id"],
                "regime_name": r["regime_name"],
                "model_id": FINAL_MODEL_ID,
                "cycle_min": r["cycle_min"],
                "cycle_max": r["cycle_max"],
                "n_cycles_available": int(d["cycle_index"].nunique()) if len(d) else 0,
                "mean_rmse_mV": float(d["best_rmse_mV"].mean()) if len(d) else np.nan,
                "median_rmse_mV": float(d["best_rmse_mV"].median()) if len(d) else np.nan,
                "std_rmse_mV": float(d["best_rmse_mV"].std()) if len(d) else np.nan,
                "min_rmse_mV": float(d["best_rmse_mV"].min()) if len(d) else np.nan,
                "max_rmse_mV": float(d["best_rmse_mV"].max()) if len(d) else np.nan,
                "mean_bfr_percent": float(d["best_bfr_percent"].mean()) if len(d) else np.nan,
                "median_bfr_percent": float(d["best_bfr_percent"].median()) if len(d) else np.nan,
                "mean_r2_percent": float(d["best_r2_percent"].mean()) if len(d) else np.nan,
                "median_r2_percent": float(d["best_r2_percent"].median()) if len(d) else np.nan,
                "good_fit_fraction": float(d["is_good_fit"].mean()) if len(d) else np.nan,
                "interpretation": r["interpretation"],
            }
        )

    out = pd.DataFrame(rows)

    out_path = OUT_TABLE_DIR / "generalizability_rmse_by_regime.csv"
    detail_path = OUT_TABLE_DIR / "generalizability_rmse_by_cycle_detail.csv"

    out.to_csv(out_path, index=False)
    df.to_csv(detail_path, index=False)

    print("[saved]", out_path)
    print("[saved]", detail_path)

    plot_generalizability_by_cycle(df)
    plot_generalizability_box(df)

    return out


def plot_generalizability_by_cycle(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.8))

    ax.plot(
        df["cycle_index"],
        df["best_rmse_mV"],
        marker="o",
        linewidth=1.5,
        markersize=3.0,
        label=FINAL_MODEL_ID,
    )

    ymax = float(np.nanpercentile(df["best_rmse_mV"], 95)) if len(df) else 1.0
    ymax = max(ymax * 1.35, 3.0)

    for r in REGIMES:
        ax.axvspan(r["cycle_min"], r["cycle_max"], alpha=0.08)
        ax.text(
            0.5 * (r["cycle_min"] + r["cycle_max"]),
            ymax,
            f"{r['regime_id']}",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    ax.axhline(1000.0 * GOOD_RMSE_V, linestyle=":", linewidth=1.6, label="2 mV threshold")

    ax.set_ylim(0, ymax)
    ax.set_title("Model Fit Quality Across Experimental Regimes", fontsize=15, pad=12)
    ax.set_xlabel("Original cycle index", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "generalizability_rmse_by_cycle.png")


def plot_generalizability_box(df: pd.DataFrame) -> None:
    data = []
    labels = []

    for r in REGIMES:
        d = df[df["cycle_index"].between(r["cycle_min"], r["cycle_max"])].copy()
        vals = pd.to_numeric(d["best_rmse_mV"], errors="coerce").dropna().to_numpy()
        if len(vals):
            data.append(vals)
            labels.append(f"{r['regime_id']}\n{r['cycle_min']}-{r['cycle_max']}")

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)

    ax.axhline(1000.0 * GOOD_RMSE_V, linestyle=":", linewidth=1.6, label="2 mV threshold")
    ax.set_title("S17_C4K RMSE Distribution by Experimental Regime", fontsize=15, pad=12)
    ax.set_xlabel("Regime", fontsize=13)
    ax.set_ylabel("Best RMSE [mV]", fontsize=13)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="best")

    fig.tight_layout()
    savefig(OUT_FIG_DIR / "generalizability_rmse_box_by_regime.png")


def find_cycle_data_files() -> list[Path]:
    files = []

    for root in [GRID_ROOT, KPARAM_ROOT]:
        if root.exists():
            files.extend(sorted(root.glob("**/selected_real_cycle_id_data.csv")))

    preferred = [
        p for p in files
        if ("anchor6_" in str(p) or "full16rem_" in str(p))
    ]

    return preferred if preferred else files


def infer_cycle_from_path(path: Path) -> int | None:
    s = str(path)

    patterns = [
        r"cycle_(\d+)_seed",
        r"cycle_(\d+)_",
        r"cycle(\d+)",
    ]

    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return int(m.group(1))

    return None


def choose_column(df: pd.DataFrame, keywords: list[str], exclude: list[str] | None = None) -> str | None:
    exclude = exclude or []

    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("_", "")
        if any(e.lower().replace(" ", "").replace("_", "") in lc for e in exclude):
            continue
        for k in keywords:
            kk = k.lower().replace(" ", "").replace("_", "")
            if kk == lc:
                return c

    for c in df.columns:
        lc = c.lower()
        if any(e.lower() in lc for e in exclude):
            continue
        if any(k.lower() in lc for k in keywords):
            return c

    return None


def summarize_cycle_data_file(path: Path) -> dict | None:
    df = safe_read_csv(path)
    if df is None or len(df) == 0:
        return None

    cycle = infer_cycle_from_path(path)
    if cycle is None:
        return None

    if not (SELECTED_CYCLE_START <= cycle <= SELECTED_CYCLE_END):
        return None

    time_col = choose_column(df, ["time_s", "time", "t_s", "relative_time"], exclude=["cycle"])
    current_col = choose_column(df, ["current_a", "current", "i_a", "I/mA", "I"])
    voltage_col = choose_column(df, ["voltage_v", "voltage", "ewe", "Ewe", "v_v", "V"])

    if time_col is None or current_col is None or voltage_col is None:
        return {
            "cycle_index": cycle,
            "source_file": str(path),
            "n_samples": len(df),
            "time_col": time_col,
            "current_col": current_col,
            "voltage_col": voltage_col,
            "note": "Could not identify one or more required columns.",
        }

    t = pd.to_numeric(df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    i = pd.to_numeric(df[current_col], errors="coerce").dropna().to_numpy(dtype=float)
    v = pd.to_numeric(df[voltage_col], errors="coerce").dropna().to_numpy(dtype=float)

    if len(t) == 0 or len(i) == 0 or len(v) == 0:
        return None

    ts = np.sort(t)
    dt = np.diff(ts)
    dt = dt[np.isfinite(dt) & (dt > 0)]

    return {
        "cycle_index": cycle,
        "source_file": str(path),
        "n_samples": int(len(df)),
        "time_col": time_col,
        "current_col": current_col,
        "voltage_col": voltage_col,
        "cycle_start_time_s": float(np.nanmin(t)),
        "cycle_end_time_s": float(np.nanmax(t)),
        "cycle_duration_s": float(np.nanmax(t) - np.nanmin(t)),
        "sampling_period_median_s": float(np.nanmedian(dt)) if len(dt) else np.nan,
        "sampling_period_mean_s": float(np.nanmean(dt)) if len(dt) else np.nan,
        "current_mean": float(np.nanmean(i)),
        "current_std": float(np.nanstd(i)),
        "current_min": float(np.nanmin(i)),
        "current_max": float(np.nanmax(i)),
        "voltage_mean": float(np.nanmean(v)),
        "voltage_std": float(np.nanstd(v)),
        "voltage_min": float(np.nanmin(v)),
        "voltage_max": float(np.nanmax(v)),
        "note": "",
    }


def make_experiment_setup_table() -> pd.DataFrame:
    files = find_cycle_data_files()

    rows_by_cycle = {}

    for p in files:
        row = summarize_cycle_data_file(p)
        if row is None:
            continue

        cycle = int(row["cycle_index"])
        if cycle not in rows_by_cycle:
            rows_by_cycle[cycle] = row

    cycle_df = pd.DataFrame([rows_by_cycle[k] for k in sorted(rows_by_cycle)])

    cycle_summary_path = OUT_TABLE_DIR / "real_data_cycle_window_summary.csv"
    cycle_df.to_csv(cycle_summary_path, index=False)
    print("[saved]", cycle_summary_path)

    if len(cycle_df) == 0:
        setup = pd.DataFrame(
            [
                {
                    "selected_cycle_start": SELECTED_CYCLE_START,
                    "selected_cycle_end": SELECTED_CYCLE_END,
                    "number_of_retained_discharge_cycles_expected": len(SELECTED_CYCLES),
                    "number_of_retained_discharge_cycles_found": 0,
                    "note": "No selected_real_cycle_id_data.csv files found.",
                }
            ]
        )
    else:
        setup = pd.DataFrame(
            [
                {
                    "selected_cycle_start": SELECTED_CYCLE_START,
                    "selected_cycle_end": SELECTED_CYCLE_END,
                    "number_of_retained_discharge_cycles_expected": len(SELECTED_CYCLES),
                    "number_of_retained_discharge_cycles_found": int(cycle_df["cycle_index"].nunique()),
                    "sampling_period_s": float(cycle_df["sampling_period_median_s"].median()),
                    "sampling_period_mean_s": float(cycle_df["sampling_period_mean_s"].mean()),
                    "selected_window_start_time_s": float(cycle_df["cycle_start_time_s"].min()),
                    "selected_window_end_time_s": float(cycle_df["cycle_end_time_s"].max()),
                    "mean_cycle_duration_s": float(cycle_df["cycle_duration_s"].mean()),
                    "median_cycle_duration_s": float(cycle_df["cycle_duration_s"].median()),
                    "min_cycle_duration_s": float(cycle_df["cycle_duration_s"].min()),
                    "max_cycle_duration_s": float(cycle_df["cycle_duration_s"].max()),
                    "discharge_current_mean": float(cycle_df["current_mean"].mean()),
                    "discharge_current_std": float(cycle_df["current_mean"].std()),
                    "current_range_min": float(cycle_df["current_min"].min()),
                    "current_range_max": float(cycle_df["current_max"].max()),
                    "voltage_range_min": float(cycle_df["voltage_min"].min()),
                    "voltage_range_max": float(cycle_df["voltage_max"].max()),
                    "data_preprocessing_steps": (
                        "Discharge cycles 34--99 were retained as the selected repeated-discharge window. "
                        "The identification scripts used automatic current-sign handling so discharge is treated consistently. "
                        "The identification data were downsampled to 1.0 s according to UN_ID_DOWNSAMPLE_DT. "
                        "Voltage smoothing was disabled unless UN_SMOOTH_VOLTAGE_WINDOW was set above 1."
                    ),
                    "note": (
                        "Computed from selected_real_cycle_id_data.csv files found in completed anchor/full16 runs. "
                        "Check real_data_cycle_window_summary.csv for per-cycle details."
                    ),
                }
            ]
        )

    setup_path = OUT_TABLE_DIR / "real_data_experiment_setup_table.csv"
    setup.to_csv(setup_path, index=False)
    print("[saved]", setup_path)

    return setup


def main() -> None:
    print("=" * 100)
    print("CHAPTER 6 REQUIRED RESULT TABLES")
    print("=" * 100)
    print("PROJECT:", PROJECT)
    print("OUT_TABLE_DIR:", OUT_TABLE_DIR)
    print("OUT_FIG_DIR:", OUT_FIG_DIR)
    print("Selected cycles:", SELECTED_CYCLE_START, "to", SELECTED_CYCLE_END)
    print("Final model:", FINAL_MODEL_ID)
    print("=" * 100)

    print("\n[1/4] model_complexity_summary.csv")
    make_required_model_complexity_summary()

    print("\n[2/4] fit_quality_by_cycle.csv and good-fit mask")
    make_fit_quality_tables()

    print("\n[3/4] generalizability_rmse_by_regime.csv")
    make_generalizability_table()

    print("\n[4/4] real_data_experiment_setup_table.csv")
    make_experiment_setup_table()

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print("Required tables:")
    for name in [
        "model_complexity_summary.csv",
        "fit_quality_by_cycle.csv",
        "good_fit_cycles.csv",
        "bad_fit_cycles.csv",
        "good_fit_mask_definition.txt",
        "generalizability_rmse_by_regime.csv",
        "generalizability_rmse_by_cycle_detail.csv",
        "real_data_experiment_setup_table.csv",
        "real_data_cycle_window_summary.csv",
    ]:
        print(" ", OUT_TABLE_DIR / name)

    print()
    print("Figures:")
    for name in [
        "generalizability_rmse_by_cycle.png",
        "generalizability_rmse_box_by_regime.png",
    ]:
        print(" ", OUT_FIG_DIR / name)
        print(" ", THESIS_FIG_DIR / name)
    print("=" * 100)


if __name__ == "__main__":
    main()
