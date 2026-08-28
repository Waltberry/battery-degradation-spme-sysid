#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_ch6_thesis_plots.py

Purpose
-------
Create thesis-ready Chapter 6 figures from already-saved real-data results.

This script does NOT rerun CT-ID.

It creates:

    figures/chapter6/fig_ch6_raw_data_full.png
    figures/chapter6/fig_ch6_retained_raw_current_voltage.png
    figures/chapter6/fig_ch6_retained_discharge_time_axis.png
    figures/chapter6/fig_ch6_retained_discharge_cycle_axis.png
    figures/chapter6/fig_ch6_representative_discharge_cycle_current_voltage.png
    figures/chapter6/fig_ch6_representative_well_fitted_response.png

    figures/chapter6/fig_ch6_good_fit_mask.png
    figures/chapter6/fig_ch6_best_rmse.png
    figures/chapter6/fig_ch6_bfr.png
    figures/chapter6/fig_ch6_core_parameters.png
    figures/chapter6/fig_ch6_electrolyte_k_parameters.png
    figures/chapter6/fig_ch6_beta_coefficients.png

The retained window follows the S17-from-34 analysis:
    original cycle 34 becomes retained cycle 0.
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

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# Project paths
# ============================================================

REAL_PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
FLOW_PROJECT = Path("/home/onyero.ofuzim/projects/Battery_Analysis/Flow Battery Project")

PROJECT_CANDIDATES = [
    REAL_PROJECT,
    FLOW_PROJECT,
]

# Main Chapter 6 outputs are now written inside the real-data project.
THESIS_FIG_DIR = REAL_PROJECT / "figures" / "chapter6"
THESIS_TABLE_DIR = REAL_PROJECT / "results" / "tables" / "chapter6_thesis_plots"
BACKUP_FIG_DIR = REAL_PROJECT / "results" / "figures" / "chapter6_thesis_plots"

# Also copy final thesis figures to the Flow Battery thesis folder.
MIRROR_TO_FLOW_THESIS = True
FLOW_THESIS_FIG_DIR = FLOW_PROJECT / "figures" / "chapter6"

for p in [THESIS_FIG_DIR, THESIS_TABLE_DIR, BACKUP_FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# Retained cycle window
# ============================================================

START_ORIGINAL_CYCLE = 34
END_ORIGINAL_CYCLE = 100

RANDOM_CYCLE_SEED = 17


# ============================================================
# Existing S17 diagnostic source locations
# ============================================================

S17_FIG_SOURCE_DIRS = [
    REAL_PROJECT / "results" / "figures" / "kparam_s17_from34_good_fit_diagnostics",
    FLOW_PROJECT / "results" / "figures" / "kparam_s17_from34_good_fit_diagnostics",
]

S17_TABLE_SOURCE_DIRS = [
    REAL_PROJECT / "results" / "tables" / "kparam_s17_from34_good_fit_diagnostics",
    FLOW_PROJECT / "results" / "tables" / "kparam_s17_from34_good_fit_diagnostics",
]

REAL_WARM_TABLE_DIRS = [
    REAL_PROJECT / "results" / "tables" / "real_warm_continuation_ctid" / "S17_C4K",
    FLOW_PROJECT / "results" / "tables" / "real_warm_continuation_ctid" / "S17_C4K",
]


# ============================================================
# Final thesis figure names
# ============================================================

EXISTING_FIGURE_MAP = {
    # source name/glob: final thesis name
    "01_s17_from34_good_fit_mask.png": "fig_ch6_good_fit_mask.png",
    "02_s17_from34_best_rmse_mV.png": "fig_ch6_best_rmse.png",
    "03_s17_from34_bfr_r2.png": "fig_ch6_bfr.png",
    "05_s17_from34_core_parameters_linear.png": "fig_ch6_core_parameters.png",
    "07_s17_from34_k_parameters_linear.png": "fig_ch6_electrolyte_k_parameters.png",
    "11_s17_from34_beta_coefficients_linear.png": "fig_ch6_beta_coefficients.png",
}

# Clean thesis titles to place inside the copied diagnostic figures.
# File names remain descriptive, but plot titles should be short and thesis-worthy.
EXISTING_FIGURE_TITLES = {
    "fig_ch6_good_fit_mask.png": "Good Fit Mask",
    "fig_ch6_best_rmse.png": "Best RMSE",
    "fig_ch6_bfr.png": "BFR",
    "fig_ch6_core_parameters.png": "Core Parameters",
    "fig_ch6_electrolyte_k_parameters.png": "Electrolyte K Parameters",
    "fig_ch6_beta_coefficients.png": "Beta Coefficients",
}


# ============================================================
# Helpers
# ============================================================

def savefig(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

    BACKUP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, BACKUP_FIG_DIR / path.name)

    if MIRROR_TO_FLOW_THESIS:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, FLOW_THESIS_FIG_DIR / path.name)

    print("[saved]", path)


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def find_existing_figure(filename: str) -> Path | None:
    candidates = []

    for d in S17_FIG_SOURCE_DIRS:
        candidates.append(d / filename)

    for p in candidates:
        if p.exists():
            return p

    # fallback broad search
    for root in PROJECT_CANDIDATES:
        hits = sorted(root.glob(f"results/figures/**/{filename}"))
        if hits:
            return hits[0]

    return None


def retitle_png(path: Path, title: str) -> None:
    """
    Replace the old title region in an already-saved PNG and add a clean thesis title.

    This is used for diagnostic figures that already exist as PNGs. It avoids
    rerunning the diagnostic plotting script while still giving the figure a
    thesis-worthy title inside the plot.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[warning] Pillow is not available; could not retitle:", path)
        return

    if not path.exists():
        return

    img = Image.open(path).convert("RGB")
    w, h = img.size

    # Cover the old title band. The value is conservative enough to remove
    # old titles without destroying the plot body in the saved diagnostic figures.
    band_h = max(85, int(0.105 * h))

    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, w, band_h], fill="white")

    font_size = max(26, int(0.035 * h))

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]

    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass

    if font is None:
        font = ImageFont.load_default()

    # Center title.
    try:
        draw.text((w / 2, band_h / 2), title, fill="black", font=font, anchor="mm")
    except TypeError:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((w - tw) / 2, (band_h - th) / 2), title, fill="black", font=font)

    img.save(path)


def copy_existing_figure(src_name: str, dst_name: str) -> bool:
    src = find_existing_figure(src_name)

    if src is None:
        print("[missing existing figure]", src_name)
        return False

    THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_FIG_DIR.mkdir(parents=True, exist_ok=True)

    dst = THESIS_FIG_DIR / dst_name
    shutil.copy2(src, dst)

    # Clean the title inside the copied figure.
    clean_title = EXISTING_FIGURE_TITLES.get(dst_name)
    if clean_title:
        retitle_png(dst, clean_title)

    shutil.copy2(dst, BACKUP_FIG_DIR / dst.name)

    if MIRROR_TO_FLOW_THESIS:
        FLOW_THESIS_FIG_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, FLOW_THESIS_FIG_DIR / dst.name)

    print("[copied and titled]", src, "->", dst)
    return True


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_good_fit_tables() -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Returns:
        all_summary, good_summary, good_best
    """
    all_summary = None
    good_summary = None
    good_best = None

    for d in S17_TABLE_SOURCE_DIRS:
        p_all = d / "s17_from34_all_cycles_with_good_fit_flags.csv"
        p_good = d / "s17_from34_good_cycles_only_summary.csv"
        p_best = d / "s17_from34_good_cycles_only_best_runs.csv"

        if p_all.exists():
            all_summary = safe_read_csv(p_all)
        if p_good.exists():
            good_summary = safe_read_csv(p_good)
        if p_best.exists():
            good_best = safe_read_csv(p_best)

        if all_summary is not None or good_summary is not None or good_best is not None:
            break

    # fallback to real warm continuation summary
    if all_summary is None:
        for d in REAL_WARM_TABLE_DIRS:
            p = d / "all_cycles_summary.csv"
            if p.exists():
                all_summary = safe_read_csv(p)
                break

    if good_best is None:
        for d in REAL_WARM_TABLE_DIRS:
            p = d / "all_cycles_best_runs.csv"
            if p.exists():
                good_best = safe_read_csv(p)
                break

    return all_summary, good_summary, good_best


def standardize_best_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {}

    if "rmse" in df.columns and "best_rmse" not in df.columns:
        rename_map["rmse"] = "best_rmse"
    if "mae" in df.columns and "best_mae" not in df.columns:
        rename_map["mae"] = "best_mae"
    if "r2_percent" in df.columns and "best_r2_percent" not in df.columns:
        rename_map["r2_percent"] = "best_r2_percent"
    if "bfr_percent" in df.columns and "best_bfr_percent" not in df.columns:
        rename_map["bfr_percent"] = "best_bfr_percent"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def reindex_cycle_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "cycle_index" in df.columns:
        df["original_cycle_index"] = pd.to_numeric(df["cycle_index"], errors="coerce").astype("Int64")
        df["retained_cycle_index"] = df["original_cycle_index"].astype(float) - START_ORIGINAL_CYCLE

    return df


# ============================================================
# Raw MPR data loading
# ============================================================

def load_raw_mpr_and_cycles():
    from battery_deg_spme.config.settings import get_default_settings
    from battery_deg_spme.io.data_io import load_mpr_as_dataframe
    from battery_deg_spme.preprocessing.cycle_detection import find_discharging_cycles_with_meta

    settings = get_default_settings()

    df = load_mpr_as_dataframe(
        mpr_path=settings.data.mpr_path,
        time_col=settings.data.time_col,
        i_col=settings.data.i_col,
        v_col=settings.data.v_col,
    )

    min_len_for_search = (
        int(settings.cycle.min_cycle_len)
        if settings.cycle.use_min_cycle_len
        else None
    )

    cycles, cycle_meta = find_discharging_cycles_with_meta(
        df=df,
        i_col=settings.data.i_col,
        tol_i=1e-9,
        min_len=min_len_for_search,
        include_previous_segment=settings.cycle.include_previous_segment,
        n_prev_points=settings.cycle.n_prev_points,
    )

    return df, cycles, cycle_meta, settings


def retained_cycle_indices(cycles: list[pd.DataFrame]) -> list[int]:
    end = min(END_ORIGINAL_CYCLE, len(cycles) - 1)
    if START_ORIGINAL_CYCLE > end:
        raise RuntimeError(
            f"Retained cycle window invalid. Detected {len(cycles)} cycles, "
            f"but START_ORIGINAL_CYCLE={START_ORIGINAL_CYCLE}."
        )

    return list(range(START_ORIGINAL_CYCLE, end + 1))


# ============================================================
# Raw-data plots
# ============================================================

def plot_raw_data_full(df: pd.DataFrame, settings) -> None:
    t = df.index.to_numpy(dtype=float)
    i = pd.to_numeric(df[settings.data.i_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[settings.data.v_col], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, i, linewidth=1.1)
    axes[0].set_title("Raw Data")
    axes[0].set_ylabel("Current")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t, v, linewidth=1.1)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.35)

    savefig(THESIS_FIG_DIR / "fig_ch6_raw_data_full.png")


def plot_retained_raw_current_voltage(cycles: list[pd.DataFrame], idxs: list[int], settings) -> None:
    retained = []

    for new_k, old_k in enumerate(idxs):
        cyc = cycles[old_k].copy()
        cyc["retained_cycle_index"] = new_k
        cyc["original_cycle_index"] = old_k
        retained.append(cyc)

    data = pd.concat(retained, axis=0)

    t = data.index.to_numpy(dtype=float)
    i = pd.to_numeric(data[settings.data.i_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(data[settings.data.v_col], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(t, i, linewidth=1.1)
    axes[0].set_title("Retained Raw Data")
    axes[0].set_ylabel("Current")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t, v, linewidth=1.1)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.35)

    savefig(THESIS_FIG_DIR / "fig_ch6_retained_raw_current_voltage.png")


def plot_retained_discharge_time_axis(cycles: list[pd.DataFrame], idxs: list[int], settings) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for new_k, old_k in enumerate(idxs):
        cyc = cycles[old_k]

        t = cyc.index.to_numpy(dtype=float)
        i = pd.to_numeric(cyc[settings.data.i_col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(cyc[settings.data.v_col], errors="coerce").to_numpy(dtype=float)

        axes[0].plot(t, i, linewidth=0.9, alpha=0.55)
        axes[1].plot(t, v, linewidth=0.9, alpha=0.55)

    axes[0].set_title("Discharge Cycles")
    axes[0].set_ylabel("Current")
    axes[0].grid(True, alpha=0.35)

    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.35)

    savefig(THESIS_FIG_DIR / "fig_ch6_retained_discharge_time_axis.png")


def plot_retained_discharge_cycle_axis(cycles: list[pd.DataFrame], idxs: list[int], settings) -> None:
    pieces = []

    for new_k, old_k in enumerate(idxs):
        cyc = cycles[old_k].copy()

        t = cyc.index.to_numpy(dtype=float)
        if len(t) < 2:
            continue

        t0 = float(t[0])
        t1 = float(t[-1])
        denom = max(t1 - t0, 1e-12)

        x_cycle = new_k + (t - t0) / denom

        part = pd.DataFrame(
            {
                "x_cycle": x_cycle,
                "retained_cycle_index": new_k,
                "original_cycle_index": old_k,
                "current": pd.to_numeric(cyc[settings.data.i_col], errors="coerce").to_numpy(dtype=float),
                "voltage": pd.to_numeric(cyc[settings.data.v_col], errors="coerce").to_numpy(dtype=float),
            }
        )

        pieces.append(part)

    data = pd.concat(pieces, ignore_index=True)
    data.to_csv(THESIS_TABLE_DIR / "retained_discharge_cycle_axis_data.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    for _, g in data.groupby("retained_cycle_index"):
        axes[0].plot(g["x_cycle"], g["current"], linewidth=0.8, alpha=0.65)
        axes[1].plot(g["x_cycle"], g["voltage"], linewidth=0.8, alpha=0.65)

    n_cycles = len(idxs)

    for ax in axes:
        ax.set_xlim(0, n_cycles)
        ax.grid(True, alpha=0.35)

        for k in range(0, n_cycles + 1, 10):
            ax.axvline(k, linewidth=0.5, alpha=0.25)

    axes[0].set_title("Discharge Cycles by Cycle Index")
    axes[0].set_ylabel("Current")

    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Retained cycle index")

    savefig(THESIS_FIG_DIR / "fig_ch6_retained_discharge_cycle_axis.png")


def plot_representative_discharge_cycle(cycles: list[pd.DataFrame], idxs: list[int], settings) -> None:
    rng = np.random.default_rng(RANDOM_CYCLE_SEED)
    old_k = int(rng.choice(idxs))
    new_k = old_k - START_ORIGINAL_CYCLE

    cyc = cycles[old_k].copy()

    t = cyc.index.to_numpy(dtype=float)
    t_rel = t - t[0]

    i = pd.to_numeric(cyc[settings.data.i_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(cyc[settings.data.v_col], errors="coerce").to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6.5), sharex=True)

    axes[0].plot(t_rel, i, linewidth=2.0)
    axes[0].set_title("Representative Discharge Cycle")
    axes[0].set_ylabel("Current")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(t_rel, v, linewidth=2.0)
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Relative time [s]")
    axes[1].grid(True, alpha=0.35)

    savefig(THESIS_FIG_DIR / "fig_ch6_representative_discharge_cycle_current_voltage.png")

    pd.DataFrame(
        {
            "relative_time_s": t_rel,
            "current": i,
            "voltage_V": v,
            "original_cycle_index": old_k,
            "retained_cycle_index": new_k,
        }
    ).to_csv(THESIS_TABLE_DIR / "representative_discharge_cycle_current_voltage.csv", index=False)


# ============================================================
# Representative fitted response
# ============================================================

def find_response_csv_for_best_row(row: pd.Series) -> Path | None:
    for col in ["source_folder", "best_source_folder"]:
        if col in row.index and pd.notna(row[col]):
            folder = Path(str(row[col]))

            p = folder / "best_measured_estimated_response.csv"
            if p.exists():
                return p

            manifest = folder / "best_manifest.csv"
            if manifest.exists():
                m = safe_read_csv(manifest)
                if m is not None and len(m) and "response_csv" in m.columns:
                    p2 = Path(str(m.iloc[0]["response_csv"]))
                    if p2.exists():
                        return p2

    return None


def plot_representative_well_fitted_response(good_best: pd.DataFrame | None) -> None:
    if good_best is None or len(good_best) == 0:
        print("[skip] no good_best table available for representative fitted response")
        return

    d = good_best.copy()

    if "model_id" in d.columns:
        d = d[d["model_id"].astype(str).eq("S17_C4K")].copy()

    if len(d) == 0:
        print("[skip] no S17_C4K rows in good_best")
        return

    d = standardize_best_columns(d)

    if "best_rmse" in d.columns:
        d = d.sort_values("best_rmse").reset_index(drop=True)
    elif "rmse" in d.columns:
        d = d.sort_values("rmse").reset_index(drop=True)
    else:
        d = d.sort_values("cycle_index").reset_index(drop=True)

    # Use a very good fit, but not necessarily the first row if response CSV is missing.
    selected = None
    selected_csv = None

    for _, row in d.iterrows():
        response_csv = find_response_csv_for_best_row(row)
        if response_csv is not None:
            selected = row
            selected_csv = response_csv
            break

    if selected is None or selected_csv is None:
        print("[skip] could not find any best_measured_estimated_response.csv")
        return

    resp = pd.read_csv(selected_csv)

    needed = ["t_s", "measured_voltage_V", "estimated_voltage_V"]
    missing = [c for c in needed if c not in resp.columns]
    if missing:
        print("[skip] response CSV missing columns:", missing, selected_csv)
        return

    t = pd.to_numeric(resp["t_s"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(resp["measured_voltage_V"], errors="coerce").to_numpy(dtype=float)
    yh = pd.to_numeric(resp["estimated_voltage_V"], errors="coerce").to_numpy(dtype=float)

    residual = y - yh
    if "residual_V" in resp.columns:
        residual = pd.to_numeric(resp["residual_V"], errors="coerce").to_numpy(dtype=float)

    cycle_original = int(selected["cycle_index"]) if "cycle_index" in selected.index and pd.notna(selected["cycle_index"]) else -1
    cycle_retained = cycle_original - START_ORIGINAL_CYCLE if cycle_original >= 0 else -1

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)

    axes[0].plot(t, y, linewidth=2.6, label="Measured voltage")
    axes[0].plot(t, yh, "--", linewidth=2.2, label="Fitted voltage")
    axes[0].set_title("Representative Well-Fitted Response")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].plot(t, residual, linewidth=1.6)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Residual [V]")
    axes[1].grid(True, alpha=0.35)

    savefig(THESIS_FIG_DIR / "fig_ch6_representative_well_fitted_response.png")

    out = resp.copy()
    out["source_response_csv"] = str(selected_csv)
    out["original_cycle_index"] = cycle_original
    out["retained_cycle_index"] = cycle_retained
    out.to_csv(THESIS_TABLE_DIR / "representative_well_fitted_response.csv", index=False)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 100)
    print("CHAPTER 6 THESIS PLOTS")
    print("=" * 100)
    print("REAL_PROJECT:", REAL_PROJECT)
    print("FLOW_PROJECT:", FLOW_PROJECT)
    print("THESIS_FIG_DIR:", THESIS_FIG_DIR)
    print("BACKUP_FIG_DIR:", BACKUP_FIG_DIR)
    print("MIRROR_TO_FLOW_THESIS:", MIRROR_TO_FLOW_THESIS)
    print("=" * 100)

    # --------------------------------------------------------
    # A. Copy already-good S17 diagnostic plots to thesis names
    # --------------------------------------------------------
    print("\nCopying existing S17 diagnostic figures...")

    for src_name, dst_name in EXISTING_FIGURE_MAP.items():
        copy_existing_figure(src_name, dst_name)

    # --------------------------------------------------------
    # B. Create raw-data and retained-cycle figures
    # --------------------------------------------------------
    print("\nCreating raw-data and retained-cycle figures...")

    try:
        df, cycles, cycle_meta, settings = load_raw_mpr_and_cycles()
        idxs = retained_cycle_indices(cycles)

        print("Raw data shape:", df.shape)
        print("Detected discharge cycles:", len(cycles))
        print("Retained original cycles:", idxs[0], "to", idxs[-1])
        print("Retained cycle count:", len(idxs))

        plot_raw_data_full(df, settings)
        plot_retained_raw_current_voltage(cycles, idxs, settings)
        plot_retained_discharge_time_axis(cycles, idxs, settings)
        plot_retained_discharge_cycle_axis(cycles, idxs, settings)
        plot_representative_discharge_cycle(cycles, idxs, settings)

    except Exception as exc:
        print("[warning] raw-data plots were not created:", repr(exc))
        (THESIS_TABLE_DIR / "raw_data_plot_warning.txt").write_text(repr(exc), encoding="utf-8")

    # --------------------------------------------------------
    # C. Create representative measured-vs-fitted response
    # --------------------------------------------------------
    print("\nCreating representative fitted response figure...")

    all_summary, good_summary, good_best = load_good_fit_tables()
    plot_representative_well_fitted_response(good_best)

    # Save any available tables with retained-cycle reindexing.
    for name, table in [
        ("s17_all_summary_reindexed.csv", all_summary),
        ("s17_good_summary_reindexed.csv", good_summary),
        ("s17_good_best_reindexed.csv", good_best),
    ]:
        if table is not None:
            out = reindex_cycle_column(table)
            out.to_csv(THESIS_TABLE_DIR / name, index=False)
            print("[saved table]", THESIS_TABLE_DIR / name)

    print()
    print("=" * 100)
    print("DONE — Chapter 6 thesis figures")
    print("=" * 100)
    for p in sorted(THESIS_FIG_DIR.glob("fig_ch6_*.png")):
        print(" ", p)
    print("=" * 100)


if __name__ == "__main__":
    main()
