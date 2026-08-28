#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import re

PROJECT = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

SEARCH_ROOTS = [
    PROJECT / "results",
    PROJECT / "results" / "tables",
    PROJECT / "results" / "figures",
]

PATTERNS = [
    "real_cycle_ctid_state_order_grid",
    "real_warm_continuation_ctid",
    "kparam",
    "S7",
    "S17",
]

def read_csv_safe(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def summarize_csv(p: Path):
    df = read_csv_safe(p)
    if df is None:
        return None

    cols = list(df.columns)

    cycle_cols = [c for c in cols if "cycle" in c.lower()]
    model_cols = [c for c in cols if "model" in c.lower() or "state" in c.lower() or "candidate" in c.lower()]
    metric_cols = [c for c in cols if any(k in c.lower() for k in ["rmse", "bfr", "r2", "mae"])]

    out = {
        "path": str(p),
        "n_rows": len(df),
        "n_cols": len(cols),
        "cycle_cols": ",".join(cycle_cols),
        "model_cols": ",".join(model_cols),
        "metric_cols": ",".join(metric_cols),
    }

    for c in cycle_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals):
            out[f"{c}_min"] = int(vals.min())
            out[f"{c}_max"] = int(vals.max())
            out[f"{c}_nunique"] = int(vals.nunique())

    for c in model_cols:
        try:
            out[f"{c}_unique"] = ",".join(map(str, sorted(df[c].dropna().astype(str).unique())))
        except Exception:
            pass

    return out

def main():
    print("=" * 100)
    print("AUDIT REAL CT-ID RESULTS")
    print("=" * 100)
    print("PROJECT:", PROJECT)

    csvs = []
    for root in SEARCH_ROOTS:
        if root.exists():
            csvs.extend(sorted(root.glob("**/*.csv")))

    keep = []
    for p in csvs:
        s = str(p)
        if any(k in s for k in PATTERNS):
            keep.append(p)

    print("Candidate CSV files:", len(keep))

    rows = []
    for p in keep:
        info = summarize_csv(p)
        if info:
            rows.append(info)

    out = pd.DataFrame(rows)
    out_path = PROJECT / "results" / "tables" / "real_ctid_result_audit.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("[saved]", out_path)

    # Print the most useful summary files.
    useful = out[
        out["path"].str.contains(
            "all_cycles_summary|all_cycles_best_runs|summary|good|diagnostic",
            case=False,
            regex=True,
        )
    ].copy()

    pd.set_option("display.max_colwidth", 120)
    print()
    print("Useful-looking files:")
    print(useful[["path", "n_rows", "cycle_cols", "model_cols", "metric_cols"]].head(80).to_string(index=False))

if __name__ == "__main__":
    main()
