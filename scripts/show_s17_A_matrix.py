#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
show_s17_A_matrix.py

Purpose
-------
Print and save the S17 continuous-time A matrix.

S17 structure:
    negative solid states: 4
    positive solid states: 4
    electrolyte states:    9

Total:
    4 + 4 + 9 = 17 states

State order:
    x0  = n0
    x1  = n1
    x2  = n2
    x3  = n3_surf

    x4  = p0
    x5  = p1
    x6  = p2
    x7  = p3_surf

    x8  = e0_neg
    x9  = e1_neg
    x10 = e2_neg

    x11 = e3_sep
    x12 = e4_sep
    x13 = e5_sep

    x14 = e6_pos
    x15 = e7_pos
    x16 = e8_pos

Important:
    S17 has the same 4-state solid blocks as S14.
    The difference is the electrolyte block:
        S14 electrolyte: 6 states
        S17 electrolyte: 9 states
"""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
from scipy.linalg import block_diag


# ============================================================
# Paths and configuration
# ============================================================
PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")

OUT_DIR = PROJECT_DIR / "results/tables/s17_A_matrix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CSV = PROJECT_DIR / "results/tables/real_warm_continuation_ctid/S17_C4/all_cycles_best_runs.csv"

# Pick which fitted cycle to use for the numeric fitted A matrix.
# Default: 34, because you are removing cycles 0--33 for trusted plots.
CYCLE_INDEX = int(os.environ.get("S17_A_CYCLE", "34"))

# Geometry defaults used in the fitting script.
L_N = float(os.environ.get("UN_L_N", "80e-6"))
L_SEP = float(os.environ.get("UN_L_SEP", "25e-6"))
L_P = float(os.environ.get("UN_L_P", "75e-6"))


# ============================================================
# State names
# ============================================================
def solid_n_names() -> list[str]:
    return ["n0", "n1", "n2", "n3_surf"]


def solid_p_names() -> list[str]:
    return ["p0", "p1", "p2", "p3_surf"]


def electrolyte_names_s17() -> list[str]:
    return [
        "e0_neg",
        "e1_neg",
        "e2_neg",
        "e3_sep",
        "e4_sep",
        "e5_sep",
        "e6_pos",
        "e7_pos",
        "e8_pos",
    ]


def state_names_s17() -> list[str]:
    return solid_n_names() + solid_p_names() + electrolyte_names_s17()


# ============================================================
# Matrix builders
# ============================================================
def build_solid_A_general(n: int, alpha: float) -> np.ndarray:
    """
    Solid diffusion chain.

    For n = 4:

        [-24a,  24a,    0,    0]
        [ 16a, -40a,  24a,    0]
        [   0,  16a, -40a,  24a]
        [   0,    0,  16a, -16a]

    This is the same 4-state solid block form used by S14 and S17.
    """
    if n < 2:
        raise ValueError("solid node count must be >= 2")

    if n == 2:
        return np.array(
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
            dtype=np.float64,
        )

    lower = 4.0 * n * alpha
    upper = 6.0 * n * alpha

    A = np.zeros((n, n), dtype=np.float64)

    A[0, 0] = -upper
    A[0, 1] = upper

    for i in range(1, n - 1):
        A[i, i - 1] = lower
        A[i, i] = -(lower + upper)
        A[i, i + 1] = upper

    A[n - 1, n - 2] = lower
    A[n - 1, n - 1] = -lower

    return A


def electrolyte_region_counts(n_e: int) -> tuple[int, int, int]:
    """
    S17 uses 9 electrolyte states:
        3 negative-region nodes,
        3 separator nodes,
        3 positive-region nodes.
    """
    if n_e not in (3, 6, 9):
        raise ValueError("Supported electrolyte node counts are 3, 6, and 9.")

    m = n_e // 3
    return m, m, m


def build_electrolyte_A_general(
    n_e: int,
    K_e: float,
    L_n: float,
    L_sep: float,
    L_p: float,
) -> np.ndarray:
    """
    Electrolyte diffusion chain across:

        negative electrode | separator | positive electrode

    Coupling between neighboring electrolyte nodes is:

        coupling = K_e / distance^2

    where distance is the finite-volume center-to-center distance.
    """
    m_n, m_sep, m_p = electrolyte_region_counts(n_e)

    region_lengths = [L_n, L_sep, L_p]
    region_counts = [m_n, m_sep, m_p]

    region_ids = []
    for rid, count in enumerate(region_counts):
        region_ids.extend([rid] * count)

    A = np.zeros((n_e, n_e), dtype=np.float64)

    for j in range(n_e - 1):
        r_left = region_ids[j]
        r_right = region_ids[j + 1]

        if r_left == r_right:
            L = region_lengths[r_left]
            m = region_counts[r_left]
            dist = L / m
        else:
            L_left = region_lengths[r_left]
            L_right = region_lengths[r_right]
            m_left = region_counts[r_left]
            m_right = region_counts[r_right]
            dist = L_left / (2.0 * m_left) + L_right / (2.0 * m_right)

        w = float(K_e) / (dist**2)

        A[j, j] -= w
        A[j, j + 1] += w
        A[j + 1, j] += w
        A[j + 1, j + 1] -= w

    return A


# ============================================================
# I/O helpers
# ============================================================
def save_matrix_csv(A: np.ndarray, path: Path, names: list[str]) -> None:
    """
    Save matrix using names matching the matrix size.
    """
    A = np.asarray(A, dtype=np.float64)

    if A.shape[0] != len(names) or A.shape[1] != len(names):
        raise ValueError(
            f"Name length mismatch for {path.name}: "
            f"A shape = {A.shape}, names = {len(names)}"
        )

    df = pd.DataFrame(A, index=names, columns=names)
    df.to_csv(path)
    print(f"[saved] {path}")


def print_matrix(title: str, A: np.ndarray, names: list[str], precision: int = 4) -> None:
    A = np.asarray(A, dtype=np.float64)

    if A.shape[0] != len(names) or A.shape[1] != len(names):
        raise ValueError(
            f"Name length mismatch while printing {title}: "
            f"A shape = {A.shape}, names = {len(names)}"
        )

    df = pd.DataFrame(A, index=names, columns=names)

    print()
    print("=" * 140)
    print(title)
    print("=" * 140)

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        260,
        "display.precision",
        precision,
        "display.float_format",
        lambda x: f"{x:.4e}",
    ):
        print(df)

    print("=" * 140)


def load_cycle_parameters(cycle_index: int) -> tuple[float, float, float]:
    """
    Load alpha_n_hat, alpha_p_hat, and K_e_hat from S17_C4 best-run table.
    """
    if not BEST_CSV.exists():
        raise FileNotFoundError(f"Missing S17 best-run table: {BEST_CSV}")

    df = pd.read_csv(BEST_CSV)
    df["cycle_index"] = df["cycle_index"].astype(int)

    row = df[df["cycle_index"] == cycle_index]

    if len(row) == 0:
        raise RuntimeError(f"Cycle {cycle_index} not found in {BEST_CSV}")

    row = row.iloc[0]

    alpha_n = float(row["alpha_n_hat"])
    alpha_p = float(row["alpha_p_hat"])
    K_e = float(row["K_e_hat"])

    return alpha_n, alpha_p, K_e


# ============================================================
# Main
# ============================================================
def main() -> None:
    # ------------------------------------------------------------
    # Coefficient/structure matrices
    # ------------------------------------------------------------
    A_solid_coeff = build_solid_A_general(4, 1.0)

    A_electrolyte_coeff = build_electrolyte_A_general(
        n_e=9,
        K_e=1.0,
        L_n=L_N,
        L_sep=L_SEP,
        L_p=L_P,
    )

    A17_coeff = block_diag(
        A_solid_coeff,
        A_solid_coeff,
        A_electrolyte_coeff,
    )

    # ------------------------------------------------------------
    # Numeric fitted matrix from selected cycle
    # ------------------------------------------------------------
    alpha_n, alpha_p, K_e = load_cycle_parameters(CYCLE_INDEX)

    A_n = build_solid_A_general(4, alpha_n)
    A_p = build_solid_A_general(4, alpha_p)

    A_e = build_electrolyte_A_general(
        n_e=9,
        K_e=K_e,
        L_n=L_N,
        L_sep=L_SEP,
        L_p=L_P,
    )

    A17_numeric = block_diag(A_n, A_p, A_e)

    # ------------------------------------------------------------
    # Save matrices
    # ------------------------------------------------------------
    save_matrix_csv(
        A_solid_coeff,
        OUT_DIR / "solid_4_state_A_coefficient_alpha_equals_1.csv",
        solid_n_names(),
    )

    save_matrix_csv(
        A_electrolyte_coeff,
        OUT_DIR / "electrolyte_9_state_A_coefficient_Ke_equals_1.csv",
        electrolyte_names_s17(),
    )

    save_matrix_csv(
        A17_coeff,
        OUT_DIR / "S17_A_coefficient_matrix_alpha_n_alpha_p_Ke_equals_1.csv",
        state_names_s17(),
    )

    save_matrix_csv(
        A17_numeric,
        OUT_DIR / f"S17_A_numeric_cycle_{CYCLE_INDEX:04d}.csv",
        state_names_s17(),
    )

    pd.DataFrame(
        [
            {
                "cycle_index": CYCLE_INDEX,
                "alpha_n_hat": alpha_n,
                "alpha_p_hat": alpha_p,
                "K_e_hat": K_e,
                "L_n": L_N,
                "L_sep": L_SEP,
                "L_p": L_P,
            }
        ]
    ).to_csv(
        OUT_DIR / f"S17_A_numeric_cycle_{CYCLE_INDEX:04d}_parameters.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Print state order and matrices
    # ------------------------------------------------------------
    print()
    print("S17 A-matrix construction")
    print("=" * 140)
    print("State order:")
    for i, name in enumerate(state_names_s17()):
        print(f"  x[{i:02d}] = {name}")

    print()
    print("Selected fitted cycle:", CYCLE_INDEX)
    print("alpha_n_hat:", alpha_n)
    print("alpha_p_hat:", alpha_p)
    print("K_e_hat:", K_e)
    print("L_n:", L_N)
    print("L_sep:", L_SEP)
    print("L_p:", L_P)
    print("=" * 140)

    print()
    print("4-state solid block structure:")
    print()
    print("A_solid = alpha *")
    print(np.array2string(A_solid_coeff, precision=1, suppress_small=False))

    print()
    print("So:")
    print("  A_n = alpha_n * A_solid_coeff")
    print("  A_p = alpha_p * A_solid_coeff")
    print()
    print("This is why the S17 solid blocks look like the S14 solid blocks.")
    print("The difference is the S17 electrolyte block, which is 9 x 9.")

    print_matrix(
        title="4-state solid coefficient matrix, alpha = 1",
        A=A_solid_coeff,
        names=solid_n_names(),
        precision=4,
    )

    print_matrix(
        title="9-state electrolyte coefficient matrix, K_e = 1",
        A=A_electrolyte_coeff,
        names=electrolyte_names_s17(),
        precision=4,
    )

    print_matrix(
        title="S17 full coefficient matrix: blockdiag(A_solid, A_solid, A_electrolyte)",
        A=A17_coeff,
        names=state_names_s17(),
        precision=4,
    )

    print_matrix(
        title=f"S17 numeric fitted A matrix from cycle {CYCLE_INDEX}",
        A=A17_numeric,
        names=state_names_s17(),
        precision=6,
    )

    print()
    print("Saved matrices to:")
    print(OUT_DIR)
    print()


if __name__ == "__main__":
    main()
