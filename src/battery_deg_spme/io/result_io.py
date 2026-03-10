from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> Path:
    p = ensure_parent(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_cycle_metrics_table(df: pd.DataFrame, out_path: str | Path) -> Path:
    p = ensure_parent(out_path)
    df.to_csv(p, index=False)
    return p


def save_cycle_parameter_table(df: pd.DataFrame, out_path: str | Path) -> Path:
    p = ensure_parent(out_path)
    df.to_csv(p, index=False)
    return p


def save_dataframe(df: pd.DataFrame, out_path: str | Path, index: bool = False) -> Path:
    p = ensure_parent(out_path)
    df.to_csv(p, index=index)
    return p


def make_cycle_figure_path(
    results_dir: str | Path,
    cycle_id: int,
    stage_name: str,
    suffix: str,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"cycle_{int(cycle_id):04d}_{stage_name}_{suffix}.png"
    return results_dir / filename


def make_metrics_path(
    results_dir: str | Path,
    name: str,
    ext: str = ".csv",
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{name}{ext}"


def make_json_path(
    results_dir: str | Path,
    name: str,
) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / f"{name}.json"