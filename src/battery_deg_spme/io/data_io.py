from __future__ import annotations

from pathlib import Path

import pandas as pd

from .file_loader import load_mpr, resolve_data_path


def load_mpr_as_dataframe(
    mpr_path: str | Path,
    time_col: str,
    i_col: str,
    v_col: str,
    data_subdir: str = "raw",
) -> pd.DataFrame:
    """
    Load a BioLogic .mpr file and return a clean DataFrame indexed by time.

    Returns a DataFrame with:
        index = time_col
        columns = [i_col, v_col]
    """
    mpr_file = load_mpr(mpr_path, data_subdir=data_subdir)
    df0 = pd.DataFrame(mpr_file.data)

    missing = [c for c in [time_col, i_col, v_col] if c not in df0.columns]
    if missing:
        raise KeyError(f"Missing required columns in MPR data: {missing}")

    df = (
        df0[[time_col, i_col, v_col]]
        .dropna()
        .sort_values(by=time_col)
        .set_index(time_col)
    )

    if df.empty:
        raise RuntimeError("Loaded MPR dataframe is empty after cleaning.")

    return df


def save_dataframe_csv(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
    return out_path


def load_dataframe_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, **kwargs)


def resolve_repo_data_file(path_like: str | Path, data_subdir: str = "raw") -> Path:
    return resolve_data_path(path_like, data_subdir=data_subdir)