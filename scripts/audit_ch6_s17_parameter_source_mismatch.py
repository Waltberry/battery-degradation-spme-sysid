#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

THESIS_GOOD_BEST = (
    PROJECT
    / "results/tables/kparam_s17_from34_good_fit_diagnostics"
    / "s17_from34_good_cycles_only_best_runs.csv"
)

ALL_SUMMARY = (
    PROJECT
    / "results/tables/real_warm_continuation_ctid/S17_C4K"
    / "all_cycles_summary.csv"
)

ALL_BEST = (
    PROJECT
    / "results/tables/real_warm_continuation_ctid/S17_C4K"
    / "all_cycles_best_runs.csv"
)

NEW_FINAL8_CORE = (
    PROJECT
    / "results/tables/chapter6_s7_s17_final8_parameter_review"
    / "final8_core_parameters_long.csv"
)

OUT_DIR = PROJECT / "results/tables/chapter6_s7_s17_final8_parameter_review"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOOD_RMSE_V = 0.002
GOOD_BFR_PERCENT = 98.0
GOOD_R2_PERCENT = 99.95

PARAM_MAP = {
    "alpha_n": "alpha_n_hat",
    "alpha_p": "alpha_p_hat",
    "g_n": "g_n_hat",
    "g_p": "g_p_hat",
}

def standardize_summary(df):
    df = df.copy()
    ren = {}
    if "rmse" in df.columns and "best_rmse" not in df.columns:
        ren["rmse"] = "best_rmse"
    if "bfr_percent" in df.columns and "best_bfr_percent" not in df.columns:
        ren["bfr_percent"] = "best_bfr_percent"
    if "r2_percent" in df.columns and "best_r2_percent" not in df.columns:
        ren["r2_percent"] = "best_r2_percent"
    if ren:
        df = df.rename(columns=ren)
    return df

def load_reconstructed_good_best():
    summary = standardize_summary(pd.read_csv(ALL_SUMMARY))
    best = pd.read_csv(ALL_BEST)

    summary["cycle_index"] = summary["cycle_index"].astype(int)
    best["cycle_index"] = best["cycle_index"].astype(int)

    summary = summary[(summary["cycle_index"] >= 34) & (summary["cycle_index"] <= 100)].copy()
    best = best[(best["cycle_index"] >= 34) & (best["cycle_index"] <= 100)].copy()

    good = summary[
        (summary["best_rmse"] <= GOOD_RMSE_V)
        & (summary["best_bfr_percent"] >= GOOD_BFR_PERCENT)
        & (summary["best_r2_percent"] >= GOOD_R2_PERCENT)
    ][["cycle_index"]].drop_duplicates()

    out = best.merge(good, on="cycle_index", how="inner")
    out = out.sort_values("cycle_index").reset_index(drop=True)
    out["retained_cycle_index"] = out["cycle_index"] - 34
    return out

def main():
    print("=" * 100)
    print("AUDIT: S17_C4K PARAMETER SOURCE MISMATCH")
    print("=" * 100)

    print("\nFiles:")
    for p in [THESIS_GOOD_BEST, ALL_SUMMARY, ALL_BEST, NEW_FINAL8_CORE]:
        print(" ", p, "EXISTS" if p.exists() else "MISSING")

    reconstructed = load_reconstructed_good_best()
    reconstructed.to_csv(OUT_DIR / "audit_reconstructed_s17_good_best_from_all_tables.csv", index=False)

    print("\nReconstructed good-best table:")
    print(" rows:", len(reconstructed))
    print(" cycles:", reconstructed["cycle_index"].min(), "to", reconstructed["cycle_index"].max())
    print(" columns:", [c for c in reconstructed.columns if c.endswith("_hat")][:30])

    if THESIS_GOOD_BEST.exists():
        thesis = pd.read_csv(THESIS_GOOD_BEST)
        thesis["cycle_index"] = thesis["cycle_index"].astype(int)
        thesis = thesis.sort_values("cycle_index").reset_index(drop=True)
        thesis["retained_cycle_index"] = thesis["cycle_index"] - 34

        print("\nExisting thesis good-best table:")
        print(" rows:", len(thesis))
        print(" cycles:", thesis["cycle_index"].min(), "to", thesis["cycle_index"].max())

        rows = []
        for p_short, p_col in PARAM_MAP.items():
            if p_col not in thesis.columns or p_col not in reconstructed.columns:
                continue

            m = thesis[["cycle_index", p_col]].merge(
                reconstructed[["cycle_index", p_col]],
                on="cycle_index",
                suffixes=("_thesis", "_reconstructed"),
            )

            diff = m[f"{p_col}_thesis"] - m[f"{p_col}_reconstructed"]

            rows.append(
                {
                    "parameter": p_short,
                    "column": p_col,
                    "n_common_cycles": len(m),
                    "max_abs_diff": float(np.nanmax(np.abs(diff))) if len(m) else np.nan,
                    "mean_abs_diff": float(np.nanmean(np.abs(diff))) if len(m) else np.nan,
                }
            )

        compare = pd.DataFrame(rows)
        compare.to_csv(OUT_DIR / "audit_thesis_vs_reconstructed_s17_c4k_parameter_diff.csv", index=False)

        print("\nThesis vs reconstructed from all_cycles_best_runs:")
        print(compare.to_string(index=False))

    if NEW_FINAL8_CORE.exists():
        new = pd.read_csv(NEW_FINAL8_CORE)
        new = new[(new["model_id"] == "S17_C4K") & (new["parameter"].isin(PARAM_MAP.keys()))].copy()
        new["cycle_index"] = new["cycle_index"].astype(int)

        rows = []
        for p_short, p_col in PARAM_MAP.items():
            if p_col not in reconstructed.columns:
                continue

            a = new[new["parameter"] == p_short][["cycle_index", "value"]].rename(columns={"value": "new_value"})
            b = reconstructed[["cycle_index", p_col]].rename(columns={p_col: "thesis_source_value"})
            m = a.merge(b, on="cycle_index", how="inner")

            if len(m):
                diff = m["new_value"] - m["thesis_source_value"]
                rows.append(
                    {
                        "parameter": p_short,
                        "n_common_cycles": len(m),
                        "new_min": float(m["new_value"].min()),
                        "new_max": float(m["new_value"].max()),
                        "thesis_source_min": float(m["thesis_source_value"].min()),
                        "thesis_source_max": float(m["thesis_source_value"].max()),
                        "max_abs_diff": float(np.nanmax(np.abs(diff))),
                        "mean_abs_diff": float(np.nanmean(np.abs(diff))),
                    }
                )

        mismatch = pd.DataFrame(rows)
        mismatch.to_csv(OUT_DIR / "audit_new_final8_vs_thesis_source_s17_c4k_parameter_diff.csv", index=False)

        print("\nNew final8 core table vs thesis source:")
        print(mismatch.to_string(index=False))

    print("\nSaved audit tables to:")
    print(" ", OUT_DIR)
    print("=" * 100)

if __name__ == "__main__":
    main()
