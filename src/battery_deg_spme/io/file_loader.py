from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from galvani import BioLogic

try:
    import PSData  # noqa: F401
except ImportError:
    PSData = None


TIME_COLUMN_NAME = "Time (HH:mm:ss.SSS)"


def get_repo_root() -> Path:
    """
    Resolve repository root from:
        src/battery_deg_spme/io/file_loader.py -> repo root
    """
    return Path(__file__).resolve().parents[3]


def get_data_dir(data_subdir: str = "raw") -> Path:
    """
    Return one of:
        repo/data/raw
        repo/data/interim
        repo/data/processed
    """
    repo_root = get_repo_root()
    data_dir = repo_root / "data" / data_subdir
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def resolve_data_path(file_name: str | Path, data_subdir: str = "raw") -> Path:
    """
    Resolve a file path for repository-managed data.

    Rules:
    - If file_name is an absolute existing path, use it directly.
    - Otherwise resolve relative to repo/data/<data_subdir>.
    """
    p = Path(file_name)

    if p.is_absolute():
        if not p.exists():
            raise FileNotFoundError(f"File does not exist: {p}")
        return p

    candidate = get_data_dir(data_subdir=data_subdir) / p
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find file '{file_name}' under {get_data_dir(data_subdir=data_subdir)}"
        )
    return candidate


def load_mpr(file_name: str | Path, data_subdir: str = "raw"):
    """
    Load a BioLogic .mpr file.

    Parameters
    ----------
    file_name : str | Path
        Either an absolute path or a path relative to repo/data/<data_subdir>.
    data_subdir : str
        Usually 'raw'.

    Returns
    -------
    BioLogic.MPRfile
    """
    file_path = resolve_data_path(file_name, data_subdir=data_subdir)
    return BioLogic.MPRfile(str(file_path))


def load_excel(file_name: str | Path, data_subdir: str = "raw") -> pd.ExcelFile:
    """
    Load an Excel workbook from repo/data/<data_subdir>.
    """
    file_path = resolve_data_path(file_name, data_subdir=data_subdir)
    return pd.ExcelFile(file_path)


def load_txt(file_name: str | Path, data_subdir: str = "raw") -> pd.DataFrame:
    """
    Load a whitespace-delimited text file for quick inspection.

    This is a generic helper and may need adaptation depending on the file layout.
    """
    file_path = resolve_data_path(file_name, data_subdir=data_subdir)

    df = pd.read_csv(file_path, sep=r"\s+", skiprows=5, header=None, engine="python")
    df.columns = ["Time Step", "voltage_vp", "flow-time"]
    return df


def load_csv_files(
    file_names: Iterable[str | Path],
    sub_folder: str = "",
    data_subdir: str = "raw",
) -> dict[str, pd.DataFrame]:
    """
    Load multiple CSV files from repo/data/<data_subdir>/<sub_folder>.
    """
    base_dir = get_data_dir(data_subdir=data_subdir)
    if sub_folder:
        base_dir = base_dir / sub_folder

    if not base_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {base_dir}")

    dataframes: dict[str, pd.DataFrame] = {}

    for file_name in file_names:
        file_path = base_dir / Path(file_name)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        key = (
            file_path.name.replace(" ", "_")
            .replace(".", "_")
            .replace("(", "")
            .replace(")", "")
        )
        dataframes[key] = df

    return dataframes


def load_csv(
    file_name: str | Path,
    sub_folder: str = "",
    data_subdir: str = "raw",
) -> pd.DataFrame:
    """
    Convenience wrapper to load a single CSV file.
    """
    dfs = load_csv_files(
        [file_name],
        sub_folder=sub_folder,
        data_subdir=data_subdir,
    )
    key = (
        Path(file_name).name.replace(" ", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )
    return dfs[key]


def load_psdata_as_table(file_name: str | Path, data_subdir: str = "raw") -> pd.DataFrame:
    """
    Fallback loader to inspect a .psdata-like text file as a raw table.
    """
    file_path = resolve_data_path(file_name, data_subdir=data_subdir)

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )
    return df