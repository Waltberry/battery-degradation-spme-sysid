#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_final_thesis_s7_s17_c4_sim_chunk.py

Final thesis synthetic CT-ID runner for S7_C4K and S17_C4K only.

Purpose
-------
This replaces the old final-thesis C1/C2/C3/C4 candidate sweep.

Old study:
    C1, C2, C3, C4
    one reduced-7 state model
    1000 fits per candidate

New study:
    S7_C4K and S17_C4K
    supervisor C4K voltage output only
    direct electrolyte k/b parameterization
    one fixed synthetic dataset per model
    1000 multistart fits per model

Each Slurm array task runs:
    one state model
    one seed chunk
    usually 100 seeds

Total run:
    2 models x 10 chunks x 100 seeds = 2000 fits
    = 1000 fits per model

Models
------
S7_C4K:
    negative solid: 2 states
    positive solid: 2 states
    electrolyte:    3 states
    independent electrolyte couplings: k1, k2
    electrolyte B gains: b_en, b_ep

S17_C4K:
    negative solid: 4 states
    positive solid: 4 states
    electrolyte:    9 states
    independent electrolyte couplings: k1, k2, k3, k4, k5
    expanded edge couplings:
        [k1, k1, k2, k3, k3, k4, k5, k5]
    electrolyte B gains: b_en, b_ep

Output model
------------
Supervisor C4K output:

    Vhat = C
         + xp + ap2*xp^2 + ap3*xp^3 + ap4*xp^4
         - (xn + an2*xn^2 + an3*xn^3 + an4*xn^4)
         + D1*I
         + ze + E2*ze^2 + E3*ze^3 + E4*ze^4

The linear coefficients of xp, -xn, and ze are fixed to 1.

Trainable beta vector:
    beta = [
        C,
        ap2, ap3, ap4,
        an2, an3, an4,
        D1,
        E2, E3, E4
    ]

Outputs
-------
results/final_thesis_s7_s17_c4_sim_1000/<RUN_TAG>/
    config.json
    synthetic_dataset.csv
    all_runs.csv
    best_run.csv
    failed_runs.csv
    parameter_long.csv
    beta_coefficients.csv
    truth_phi_rank.csv
    best_measured_estimated_response.csv
    best_state_trajectory.csv
    best_feature_matrix_phi.csv

results/figures/final_thesis_s7_s17_c4_sim_1000/<RUN_TAG>/
    best_measured_vs_estimated.png
    best_residual_vs_time.png
"""

from __future__ import annotations

import os
import gc
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLBACKEND", "Agg")

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_THREADS", "8")))

os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)
os.environ["XLA_FLAGS"] = (
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={N_THREADS}"
)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from scipy.stats import chi2

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
except ImportError as exc:
    raise ImportError("JAX is required. Activate your jaxsys environment.") from exc

try:
    from jax_sysid import CTModel
except ImportError:
    try:
        from jax_sysid.models import CTModel
    except ImportError as exc:
        raise ImportError("jax-sysid is required in this environment.") from exc

try:
    import diffrax
    DIFFRAX_AVAILABLE = True
except ImportError:
    diffrax = None
    DIFFRAX_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    output_root: str = "results/final_thesis_s7_s17_c4_sim_1000"
    figure_root: str = "results/figures/final_thesis_s7_s17_c4_sim_1000"

    # Model selection
    state_id: str = os.environ.get("UN_STATE_VARIANT", "S7")
    candidate_id: str = "C4"
    model_suffix: str = "C4K"

    # Simulation
    sim_dt: float = float(os.environ.get("UN_SIM_DT", "0.1"))
    sim_t_end: float = float(os.environ.get("UN_SIM_T_END", "500.0"))
    input_profile: str = "step"
    t_step: float = float(os.environ.get("UN_T_STEP", "90.0"))
    i_before: float = float(os.environ.get("UN_I_BEFORE", "0.0"))
    i_after: float = float(os.environ.get("UN_I_AFTER", "2.0"))

    enable_id_downsample: bool = True
    id_downsample_dt: float = float(os.environ.get("UN_ID_DOWNSAMPLE_DT", "1.0"))

    # Multistart controls
    seed0: int = int(os.environ.get("UN_SEED0", "200"))
    n_multistart: int = int(os.environ.get("UN_N_MULTISTART", "100"))

    # Truth dynamics
    alpha_n_true: float = float(os.environ.get("UN_ALPHA_N_TRUE", "0.0064"))
    alpha_p_true: float = float(os.environ.get("UN_ALPHA_P_TRUE", "0.0048"))

    # Direct electrolyte truth k values.
    # S7 uses first 2.
    # S17 uses all 5.
    k1_true: float = float(os.environ.get("UN_K1_TRUE", "0.42"))
    k2_true: float = float(os.environ.get("UN_K2_TRUE", "0.35"))
    k3_true: float = float(os.environ.get("UN_K3_TRUE", "0.90"))
    k4_true: float = float(os.environ.get("UN_K4_TRUE", "0.38"))
    k5_true: float = float(os.environ.get("UN_K5_TRUE", "0.72"))

    g_n_true: float = float(os.environ.get("UN_G_N_TRUE", "1.20e-4"))
    g_p_true: float = float(os.environ.get("UN_G_P_TRUE", "1.00e-4"))

    b_en_true: float = float(os.environ.get("UN_B_EN_TRUE", "0.030"))
    b_ep_true: float = float(os.environ.get("UN_B_EP_TRUE", "0.030"))

    theta_n0_true: float = float(os.environ.get("UN_THETA_N0_TRUE", "0.50"))
    theta_p0_true: float = float(os.environ.get("UN_THETA_P0_TRUE", "0.60"))
    theta_min: float = float(os.environ.get("UN_THETA_MIN", "0.02"))
    theta_max: float = float(os.environ.get("UN_THETA_MAX", "0.98"))

    ce0: float = float(os.environ.get("UN_CE0", "1000.0"))
    ce_min: float = float(os.environ.get("UN_CE_MIN", "1.0"))

    # Truth beta for C4K output.
    beta_C_true: float = float(os.environ.get("UN_BETA_C_TRUE", "2.55"))
    beta_xppow2_true: float = float(os.environ.get("UN_BETA_XP2_TRUE", "0.30"))
    beta_xppow3_true: float = float(os.environ.get("UN_BETA_XP3_TRUE", "-0.15"))
    beta_xppow4_true: float = float(os.environ.get("UN_BETA_XP4_TRUE", "0.05"))
    beta_minusxnpow2_true: float = float(os.environ.get("UN_BETA_XN2_TRUE", "0.25"))
    beta_minusxnpow3_true: float = float(os.environ.get("UN_BETA_XN3_TRUE", "-0.10"))
    beta_minusxnpow4_true: float = float(os.environ.get("UN_BETA_XN4_TRUE", "0.03"))
    beta_I_true: float = float(os.environ.get("UN_BETA_I_TRUE", "-0.004"))
    beta_zepow2_true: float = float(os.environ.get("UN_BETA_ZE2_TRUE", "-0.08"))
    beta_zepow3_true: float = float(os.environ.get("UN_BETA_ZE3_TRUE", "0.04"))
    beta_zepow4_true: float = float(os.environ.get("UN_BETA_ZE4_TRUE", "-0.01"))

    add_noise: bool = os.environ.get("UN_ADD_NOISE", "False").lower() == "true"
    noise_std: float = float(os.environ.get("UN_NOISE_STD", "1e-4"))
    synthetic_seed: int = int(os.environ.get("UN_SYNTHETIC_SEED", "11"))

    # Initialization
    init_dyn_jitter: float = float(os.environ.get("UN_INIT_DYN_JITTER", "0.30"))
    init_gain_jitter: float = float(os.environ.get("UN_INIT_GAIN_JITTER", "0.30"))
    init_C_jitter: float = float(os.environ.get("UN_INIT_C_JITTER", "0.05"))
    init_beta_scale: float = float(os.environ.get("UN_INIT_BETA_SCALE", "1e-2"))
    init_D1_center: float = float(os.environ.get("UN_INIT_D1_CENTER", "-0.004"))
    init_D1_jitter: float = float(os.environ.get("UN_INIT_D1_JITTER", "0.002"))

    # Floors
    dyn_floor: float = float(os.environ.get("UN_DYN_FLOOR", "1e-12"))
    gain_floor: float = float(os.environ.get("UN_GAIN_FLOOR", "1e-10"))

    # Optimization
    adam_epochs: int = int(os.environ.get("UN_ADAM_EPOCHS", "500"))
    adam_eta: float = float(os.environ.get("UN_ADAM_ETA", "2e-3"))
    lbfgs_epochs: int = int(os.environ.get("UN_LBFGS_EPOCHS", "8000"))
    lbfgs_tol: float = float(os.environ.get("UN_LBFGS_TOL", "1e-12"))
    lbfgs_memory: int = int(os.environ.get("UN_LBFGS_MEMORY", "30"))
    iprint: int = int(os.environ.get("UN_IPRINT", "0"))

    rho_x0: float = float(os.environ.get("UN_RHO_X0", "1e-8"))
    rho_th: float = float(os.environ.get("UN_RHO_TH", "1e-8"))
    tau_th: float = float(os.environ.get("UN_TAU_TH", "0.0"))

    # Diagnostics
    svd_tol: float = float(os.environ.get("UN_SVD_TOL", "1e-10"))
    ljung_box_lag: int = int(os.environ.get("UN_LJUNG_BOX_LAG", "20"))

    # Plotting
    make_plots: bool = os.environ.get("UN_MAKE_PLOTS", "True").lower() == "true"
    save_plots: bool = os.environ.get("UN_SAVE_PLOTS", "True").lower() == "true"
    show_plots: bool = os.environ.get("UN_SHOW_PLOTS", "False").lower() == "true"


CFG = Config()

STATE_VARIANTS: dict[str, dict[str, int]] = {
    "S7": {"n_solid_n": 2, "n_solid_p": 2, "n_electrolyte": 3},
    "S17": {"n_solid_n": 4, "n_solid_p": 4, "n_electrolyte": 9},
}

if CFG.state_id not in STATE_VARIANTS:
    raise ValueError("Use UN_STATE_VARIANT=S7 or S17.")

STATE_SPEC = STATE_VARIANTS[CFG.state_id]
MODEL_ID = f"{CFG.state_id}_{CFG.model_suffix}"
NX = STATE_SPEC["n_solid_n"] + STATE_SPEC["n_solid_p"] + STATE_SPEC["n_electrolyte"]
NY = 1
NU = 1

RUN_TAG = os.environ.get(
    "UN_RUN_TAG",
    (
        f"final_thesis_{MODEL_ID}_chunk_0_"
        f"{CFG.n_multistart}seeds_{CFG.seed0}_to_{CFG.seed0 + CFG.n_multistart - 1}_"
        f"tf_{CFG.sim_t_end}_dt_{CFG.id_downsample_dt}"
    ),
)

OUT_DIR = Path(CFG.output_root) / RUN_TAG
FIG_DIR = Path(CFG.figure_root) / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("FINAL THESIS SYNTHETIC CT-ID — S7/S17 C4K")
print("=" * 100)
print("Working directory:", Path.cwd())
print("Threads:", N_THREADS)
print("JAX x64:", jax.config.read("jax_enable_x64"))
print("STATE:", CFG.state_id)
print("MODEL_ID:", MODEL_ID)
print("NX:", NX)
print("RUN_TAG:", RUN_TAG)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("=" * 100)


# ============================================================
# Helpers
# ============================================================
def ensure_1d(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def ensure_2d_col(x: Any, dtype=float) -> np.ndarray:
    return ensure_1d(x, dtype=dtype).reshape(-1, 1)


def rmse(y: np.ndarray, yh: np.ndarray) -> float:
    y = ensure_1d(y)
    yh = ensure_1d(yh)
    n = min(len(y), len(yh))
    return float(np.sqrt(np.mean((y[:n] - yh[:n]) ** 2)))


def mae(y: np.ndarray, yh: np.ndarray) -> float:
    y = ensure_1d(y)
    yh = ensure_1d(yh)
    n = min(len(y), len(yh))
    return float(np.mean(np.abs(y[:n] - yh[:n])))


def r2_percent(y: np.ndarray, yh: np.ndarray) -> float:
    y = ensure_1d(y)
    yh = ensure_1d(yh)
    n = min(len(y), len(yh))
    y = y[:n]
    yh = yh[:n]
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return np.nan
    return float(100.0 * (1.0 - np.sum((y - yh) ** 2) / denom))


def bfr_percent(y: np.ndarray, yh: np.ndarray) -> float:
    y = ensure_1d(y)
    yh = ensure_1d(yh)
    n = min(len(y), len(yh))
    y = y[:n]
    yh = yh[:n]
    denom = float(np.linalg.norm(y - np.mean(y)))
    if denom <= 1e-15:
        return np.nan
    return float(100.0 * (1.0 - np.linalg.norm(y - yh) / denom))


def softplus_inverse_np(y: float, floor: float = 0.0) -> float:
    z = max(float(y) - float(floor), 1e-12)
    if z > 30.0:
        return z
    return float(np.log(np.expm1(z)))


def positive_from_raw_np(raw: float, floor: float) -> float:
    return floor + float(np.log1p(np.exp(float(raw))))


def positive_from_raw_jax(raw, floor: float):
    return floor + jax.nn.softplus(raw)


def matrix_rank_condition_svd_np(M: np.ndarray, tol: float = 1e-10) -> dict[str, Any]:
    M0 = np.asarray(M, dtype=np.float64)
    if M0.ndim != 2 or M0.size == 0:
        return {"rank": 0, "n_rows": 0, "n_cols": 0, "condition_number": np.nan}
    try:
        svals = np.linalg.svd(M0, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "rank": np.nan,
            "n_rows": int(M0.shape[0]),
            "n_cols": int(M0.shape[1]),
            "condition_number": np.nan,
        }
    threshold = tol * max(float(svals[0]), 1e-30)
    rank = int(np.sum(svals > threshold))
    cond = float(svals[0] / max(float(svals[-1]), 1e-300))
    return {
        "rank": rank,
        "n_rows": int(M0.shape[0]),
        "n_cols": int(M0.shape[1]),
        "condition_number": cond,
    }


def autocorr(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = ensure_1d(x, dtype=np.float64)
    x = x - np.mean(x)
    denom = float(np.dot(x, x)) + 1e-15
    lags = np.arange(0, max_lag + 1)
    vals = []
    for lag in lags:
        vals.append(float(np.dot(x[lag:], x[: len(x) - lag]) / denom))
    return lags, np.asarray(vals)


def ljung_box_test(residual: np.ndarray, lag: int) -> dict[str, float]:
    e = ensure_1d(residual)
    n = len(e)
    if n <= lag + 1:
        return {"Q": np.nan, "p_value": np.nan, "lag": lag}
    _, acf_vals = autocorr(e, lag)
    rho = acf_vals[1 : lag + 1]
    Q = n * (n + 2.0) * np.sum((rho**2) / (n - np.arange(1, lag + 1)))
    p_value = 1.0 - chi2.cdf(Q, df=lag)
    return {"Q": float(Q), "p_value": float(p_value), "lag": int(lag)}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=240, bbox_inches="tight")
    print("[saved figure]", path)
    if CFG.show_plots:
        plt.show()
    plt.close()


# ============================================================
# State indexing
# ============================================================
def make_state_indices(state_spec: dict[str, int]) -> dict[str, int]:
    n_n = int(state_spec["n_solid_n"])
    n_p = int(state_spec["n_solid_p"])
    n_e = int(state_spec["n_electrolyte"])

    idx_n0 = 0
    idx_p0 = n_n
    idx_e0 = n_n + n_p

    return {
        "idx_n_start": idx_n0,
        "idx_n_end": idx_n0 + n_n - 1,
        "idx_p_start": idx_p0,
        "idx_p_end": idx_p0 + n_p - 1,
        "idx_e_start": idx_e0,
        "idx_e_left": idx_e0,
        "idx_e_right": idx_e0 + n_e - 1,
        "idx_e_end": idx_e0 + n_e - 1,
        "nx": n_n + n_p + n_e,
    }


IDX = make_state_indices(STATE_SPEC)


def n_k_independent(n_e: int) -> int:
    if n_e == 3:
        return 2
    if n_e == 9:
        return 5
    raise ValueError("Only n_e=3 and n_e=9 are supported.")


def k_names_for_ne(n_e: int) -> list[str]:
    return [f"k{i}" for i in range(1, n_k_independent(n_e) + 1)]


# ============================================================
# State-space builders: NumPy
# ============================================================
def build_solid_A_np(n: int, alpha: float) -> np.ndarray:
    if n == 2:
        return np.array(
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
            dtype=np.float64,
        )

    if n == 4:
        return np.array(
            [
                [-24.0 * alpha, 24.0 * alpha, 0.0, 0.0],
                [16.0 * alpha, -40.0 * alpha, 24.0 * alpha, 0.0],
                [0.0, 16.0 * alpha, -40.0 * alpha, 24.0 * alpha],
                [0.0, 0.0, 16.0 * alpha, -16.0 * alpha],
            ],
            dtype=np.float64,
        )

    raise ValueError("Only solid sizes 2 and 4 are supported.")


def build_solid_B_np(n: int, g: float) -> np.ndarray:
    B = np.zeros((n, 1), dtype=np.float64)
    B[-1, 0] = float(g)
    return B


def expand_edge_couplings_np(n_e: int, k_ind: np.ndarray) -> np.ndarray:
    k = np.asarray(k_ind, dtype=np.float64).reshape(-1)

    if n_e == 3:
        return np.array([k[0], k[1]], dtype=np.float64)

    if n_e == 9:
        return np.array(
            [k[0], k[0], k[1], k[2], k[2], k[3], k[4], k[4]],
            dtype=np.float64,
        )

    raise ValueError("Unsupported electrolyte size.")


def build_electrolyte_A_np(n_e: int, k_ind: np.ndarray) -> np.ndarray:
    q = expand_edge_couplings_np(n_e, k_ind)
    A = np.zeros((n_e, n_e), dtype=np.float64)

    for j, kj in enumerate(q):
        A[j, j] -= kj
        A[j, j + 1] += kj
        A[j + 1, j] += kj
        A[j + 1, j + 1] -= kj

    return A


def build_electrolyte_B_np(n_e: int, b_en: float, b_ep: float) -> np.ndarray:
    B = np.zeros((n_e, 1), dtype=np.float64)

    if n_e == 3:
        B[0, 0] = float(b_en)
        B[2, 0] = -float(b_ep)
        return B

    if n_e == 9:
        B[0:3, 0] = float(b_en)
        B[3:6, 0] = 0.0
        B[6:9, 0] = -float(b_ep)
        return B

    raise ValueError("Unsupported electrolyte size.")


def assemble_A_B_np(
    alpha_n: float,
    alpha_p: float,
    k_ind: np.ndarray,
    g_n: float,
    g_p: float,
    b_en: float,
    b_ep: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_n = STATE_SPEC["n_solid_n"]
    n_p = STATE_SPEC["n_solid_p"]
    n_e = STATE_SPEC["n_electrolyte"]

    A_n = build_solid_A_np(n_n, alpha_n)
    A_p = build_solid_A_np(n_p, alpha_p)
    A_e = build_electrolyte_A_np(n_e, k_ind)

    B_n = build_solid_B_np(n_n, g_n)
    B_p = build_solid_B_np(n_p, g_p)
    B_e = build_electrolyte_B_np(n_e, b_en, b_ep)

    A = block_diag(A_n, A_p, A_e)
    B = np.vstack([B_n, B_p, B_e])

    return A, B


# ============================================================
# State-space builders: JAX
# ============================================================
def build_solid_A_jax(n: int, alpha):
    if n == 2:
        return jnp.array(
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
            dtype=jnp.float64,
        )

    if n == 4:
        return jnp.array(
            [
                [-24.0 * alpha, 24.0 * alpha, 0.0, 0.0],
                [16.0 * alpha, -40.0 * alpha, 24.0 * alpha, 0.0],
                [0.0, 16.0 * alpha, -40.0 * alpha, 24.0 * alpha],
                [0.0, 0.0, 16.0 * alpha, -16.0 * alpha],
            ],
            dtype=jnp.float64,
        )

    raise ValueError("Only solid sizes 2 and 4 are supported.")


def build_solid_B_jax(n: int, g):
    B = jnp.zeros((n,), dtype=jnp.float64)
    B = B.at[-1].set(g)
    return B


def expand_edge_couplings_jax(n_e: int, k_ind):
    if n_e == 3:
        return jnp.array([k_ind[0], k_ind[1]], dtype=jnp.float64)

    if n_e == 9:
        return jnp.array(
            [k_ind[0], k_ind[0], k_ind[1], k_ind[2], k_ind[2], k_ind[3], k_ind[4], k_ind[4]],
            dtype=jnp.float64,
        )

    raise ValueError("Unsupported electrolyte size.")


def build_electrolyte_A_jax(n_e: int, k_ind):
    q = expand_edge_couplings_jax(n_e, k_ind)
    A = jnp.zeros((n_e, n_e), dtype=jnp.float64)

    for j in range(n_e - 1):
        kj = q[j]
        A = A.at[j, j].add(-kj)
        A = A.at[j, j + 1].add(kj)
        A = A.at[j + 1, j].add(kj)
        A = A.at[j + 1, j + 1].add(-kj)

    return A


def build_electrolyte_B_jax(n_e: int, b_en, b_ep):
    B = jnp.zeros((n_e,), dtype=jnp.float64)

    if n_e == 3:
        B = B.at[0].set(b_en)
        B = B.at[2].set(-b_ep)
        return B

    if n_e == 9:
        B = B.at[0:3].set(b_en)
        B = B.at[3:6].set(0.0)
        B = B.at[6:9].set(-b_ep)
        return B

    raise ValueError("Unsupported electrolyte size.")


# ============================================================
# Truth and feature helpers
# ============================================================
BETA_NAMES = [
    "beta_C",
    "beta_xppow2",
    "beta_xppow3",
    "beta_xppow4",
    "beta_minusxnpow2",
    "beta_minusxnpow3",
    "beta_minusxnpow4",
    "beta_I",
    "beta_zepow2",
    "beta_zepow3",
    "beta_zepow4",
]


def truth_k_vector(cfg: Config) -> np.ndarray:
    if cfg.state_id == "S7":
        return np.array([cfg.k1_true, cfg.k2_true], dtype=np.float64)

    if cfg.state_id == "S17":
        return np.array([cfg.k1_true, cfg.k2_true, cfg.k3_true, cfg.k4_true, cfg.k5_true], dtype=np.float64)

    raise ValueError("Unsupported state_id.")


def truth_beta_vector(cfg: Config) -> np.ndarray:
    return np.array(
        [
            cfg.beta_C_true,
            cfg.beta_xppow2_true,
            cfg.beta_xppow3_true,
            cfg.beta_xppow4_true,
            cfg.beta_minusxnpow2_true,
            cfg.beta_minusxnpow3_true,
            cfg.beta_minusxnpow4_true,
            cfg.beta_I_true,
            cfg.beta_zepow2_true,
            cfg.beta_zepow3_true,
            cfg.beta_zepow4_true,
        ],
        dtype=np.float64,
    )


def c4_feature_matrix_np(X: np.ndarray, I: np.ndarray, cfg: Config) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=np.float64)
    I = ensure_1d(I, dtype=np.float64)

    x_n = np.clip(cfg.theta_n0_true + X[:, IDX["idx_n_end"]], cfg.theta_min, cfg.theta_max)
    x_p = np.clip(cfg.theta_p0_true + X[:, IDX["idx_p_end"]], cfg.theta_min, cfg.theta_max)

    ce_left = np.maximum(cfg.ce0 + X[:, IDX["idx_e_left"]], cfg.ce_min)
    ce_right = np.maximum(cfg.ce0 + X[:, IDX["idx_e_right"]], cfg.ce_min)
    z_e = ce_left / ce_right

    Phi = np.column_stack(
        [
            np.ones_like(I),
            x_p**2,
            x_p**3,
            x_p**4,
            -(x_n**2),
            -(x_n**3),
            -(x_n**4),
            I,
            z_e**2,
            z_e**3,
            z_e**4,
        ]
    )

    aux = {"x_n": x_n, "x_p": x_p, "z_e": z_e}
    return Phi, BETA_NAMES.copy(), aux


def c4_voltage_np(X: np.ndarray, I: np.ndarray, beta: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    Phi, names, aux = c4_feature_matrix_np(X, I, cfg)
    x_n = aux["x_n"]
    x_p = aux["x_p"]
    z_e = aux["z_e"]

    fixed = x_p - x_n + z_e
    y = fixed + Phi @ beta

    aux["fixed_branch"] = fixed
    aux["beta_branch"] = Phi @ beta
    aux["voltage"] = y

    return y, aux


def simulate_truth_dataset(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    t = np.arange(0.0, cfg.sim_t_end + 0.5 * cfg.sim_dt, cfg.sim_dt, dtype=np.float64)
    I = np.where(t < cfg.t_step, cfg.i_before, cfg.i_after).astype(np.float64)

    k_ind = truth_k_vector(cfg)
    beta = truth_beta_vector(cfg)

    A, B = assemble_A_B_np(
        cfg.alpha_n_true,
        cfg.alpha_p_true,
        k_ind,
        cfg.g_n_true,
        cfg.g_p_true,
        cfg.b_en_true,
        cfg.b_ep_true,
    )

    X = np.zeros((len(t), NX), dtype=np.float64)

    def f(x, u):
        return A @ x + B.reshape(-1) * float(u)

    for i in range(len(t) - 1):
        dt = cfg.sim_dt
        x = X[i]
        u = I[i]

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        X[i + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    Y, aux = c4_voltage_np(X, I, beta, cfg)

    if cfg.add_noise:
        rng = np.random.default_rng(cfg.synthetic_seed)
        Y = Y + cfg.noise_std * rng.normal(size=len(Y))

    if cfg.enable_id_downsample and cfg.id_downsample_dt > cfg.sim_dt:
        step = max(int(round(cfg.id_downsample_dt / cfg.sim_dt)), 1)
        idx = np.arange(0, len(t), step, dtype=int)
        t_id = t[idx]
        I_id = I[idx]
        Y_id = Y[idx]
        X_id = X[idx]
    else:
        t_id = t
        I_id = I
        Y_id = Y
        X_id = X

    truth = {
        "model_id": MODEL_ID,
        "state_id": cfg.state_id,
        "candidate_id": cfg.candidate_id,
        "alpha_n_true": cfg.alpha_n_true,
        "alpha_p_true": cfg.alpha_p_true,
        "g_n_true": cfg.g_n_true,
        "g_p_true": cfg.g_p_true,
        "b_en_true": cfg.b_en_true,
        "b_ep_true": cfg.b_ep_true,
        "theta_n0_true": cfg.theta_n0_true,
        "theta_p0_true": cfg.theta_p0_true,
        **{f"k{i+1}_true": float(v) for i, v in enumerate(k_ind)},
        **{name.replace("beta_", "beta_") + "_true": float(v) for name, v in zip(BETA_NAMES, beta)},
    }

    return t_id, I_id, Y_id, X_id, truth


T_id, I_id, y_id, X_truth, TRUTH = simulate_truth_dataset(CFG)
U_id = ensure_2d_col(I_id)
Y_id = ensure_2d_col(y_id)
Ts = float(np.median(np.diff(T_id)))

pd.DataFrame(
    {
        "t_s": T_id,
        "current_A": I_id,
        "voltage_V": y_id,
        **{f"x{i}": X_truth[:, i] for i in range(X_truth.shape[1])},
    }
).to_csv(OUT_DIR / "synthetic_dataset.csv", index=False)

with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump({"config": asdict(CFG), "truth": TRUTH}, f, indent=2, default=str)

print("\nSynthetic dataset:")
print("  samples:", len(T_id))
print("  Ts:", Ts)
print("  current start/end:", float(I_id[0]), float(I_id[-1]))
print("  voltage start/end:", float(y_id[0]), float(y_id[-1]))


# ============================================================
# JAX model functions
# ============================================================
def unpack_physical_params_np(params) -> dict[str, Any]:
    raw_alpha_n, raw_alpha_p, raw_k, raw_gn, raw_gp, raw_be, beta = params

    alpha_n = positive_from_raw_np(float(raw_alpha_n[0]), CFG.dyn_floor)
    alpha_p = positive_from_raw_np(float(raw_alpha_p[0]), CFG.dyn_floor)
    k_ind = np.array([positive_from_raw_np(v, CFG.dyn_floor) for v in np.asarray(raw_k).reshape(-1)], dtype=np.float64)
    g_n = positive_from_raw_np(float(raw_gn[0]), CFG.gain_floor)
    g_p = positive_from_raw_np(float(raw_gp[0]), CFG.gain_floor)
    b_en = positive_from_raw_np(float(raw_be[0]), CFG.gain_floor)
    b_ep = positive_from_raw_np(float(raw_be[1]), CFG.gain_floor)

    q_edges = expand_edge_couplings_np(STATE_SPEC["n_electrolyte"], k_ind)

    out = {
        "alpha_n": alpha_n,
        "alpha_p": alpha_p,
        "g_n": g_n,
        "g_p": g_p,
        "b_en": b_en,
        "b_ep": b_ep,
        "beta": np.asarray(beta, dtype=np.float64).reshape(-1),
    }

    for i, val in enumerate(k_ind):
        out[f"k{i+1}"] = float(val)

    for i, val in enumerate(q_edges):
        out[f"k_edge{i+1}"] = float(val)

    for name, val in zip(BETA_NAMES, out["beta"]):
        out[name] = float(val)

    return out


def state_fcn(x, u, t, params):
    raw_alpha_n, raw_alpha_p, raw_k, raw_gn, raw_gp, raw_be, beta = params

    alpha_n = positive_from_raw_jax(raw_alpha_n[0], CFG.dyn_floor)
    alpha_p = positive_from_raw_jax(raw_alpha_p[0], CFG.dyn_floor)
    k_ind = positive_from_raw_jax(raw_k, CFG.dyn_floor)
    g_n = positive_from_raw_jax(raw_gn[0], CFG.gain_floor)
    g_p = positive_from_raw_jax(raw_gp[0], CFG.gain_floor)
    b_en = positive_from_raw_jax(raw_be[0], CFG.gain_floor)
    b_ep = positive_from_raw_jax(raw_be[1], CFG.gain_floor)

    n_n = STATE_SPEC["n_solid_n"]
    n_p = STATE_SPEC["n_solid_p"]
    n_e = STATE_SPEC["n_electrolyte"]

    A_n = build_solid_A_jax(n_n, alpha_n)
    A_p = build_solid_A_jax(n_p, alpha_p)
    A_e = build_electrolyte_A_jax(n_e, k_ind)

    B_n = build_solid_B_jax(n_n, g_n)
    B_p = build_solid_B_jax(n_p, g_p)
    B_e = build_electrolyte_B_jax(n_e, b_en, b_ep)

    x_n = x[0:n_n]
    x_p = x[n_n:n_n + n_p]
    x_e = x[n_n + n_p:n_n + n_p + n_e]

    dx_n = A_n @ x_n + B_n * u[0]
    dx_p = A_p @ x_p + B_p * u[0]
    dx_e = A_e @ x_e + B_e * u[0]

    return jnp.concatenate([dx_n, dx_p, dx_e])


def output_fcn(x, u, t, params):
    raw_alpha_n, raw_alpha_p, raw_k, raw_gn, raw_gp, raw_be, beta = params

    x_n = jnp.clip(CFG.theta_n0_true + x[IDX["idx_n_end"]], CFG.theta_min, CFG.theta_max)
    x_p = jnp.clip(CFG.theta_p0_true + x[IDX["idx_p_end"]], CFG.theta_min, CFG.theta_max)

    ce_left = jnp.maximum(CFG.ce0 + x[IDX["idx_e_left"]], CFG.ce_min)
    ce_right = jnp.maximum(CFG.ce0 + x[IDX["idx_e_right"]], CFG.ce_min)
    z_e = ce_left / ce_right

    fixed = x_p - x_n + z_e

    phi = jnp.array(
        [
            1.0,
            x_p**2,
            x_p**3,
            x_p**4,
            -(x_n**2),
            -(x_n**3),
            -(x_n**4),
            u[0],
            z_e**2,
            z_e**3,
            z_e**4,
        ],
        dtype=jnp.float64,
    )

    y = fixed + jnp.dot(phi, beta)
    return jnp.array([y], dtype=jnp.float64)


# ============================================================
# Initialization
# ============================================================
def make_initial_params(seed: int, cfg: Config):
    rng = np.random.default_rng(seed)

    k_truth = truth_k_vector(cfg)
    beta_truth = truth_beta_vector(cfg)

    alpha_n0 = cfg.alpha_n_true * np.exp(cfg.init_dyn_jitter * rng.normal())
    alpha_p0 = cfg.alpha_p_true * np.exp(cfg.init_dyn_jitter * rng.normal())
    k0 = k_truth * np.exp(cfg.init_dyn_jitter * rng.normal(size=len(k_truth)))

    g_n0 = cfg.g_n_true * np.exp(cfg.init_gain_jitter * rng.normal())
    g_p0 = cfg.g_p_true * np.exp(cfg.init_gain_jitter * rng.normal())
    b_en0 = cfg.b_en_true * np.exp(cfg.init_gain_jitter * rng.normal())
    b_ep0 = cfg.b_ep_true * np.exp(cfg.init_gain_jitter * rng.normal())

    beta0 = np.zeros_like(beta_truth)
    beta0[0] = beta_truth[0] + cfg.init_C_jitter * rng.normal()
    beta0[1:] = beta_truth[1:] + cfg.init_beta_scale * rng.normal(size=len(beta_truth) - 1)
    beta0[7] = cfg.init_D1_center + cfg.init_D1_jitter * rng.normal()

    return [
        np.array([softplus_inverse_np(alpha_n0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(alpha_p0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(v, cfg.dyn_floor) for v in k0], dtype=np.float64),
        np.array([softplus_inverse_np(g_n0, cfg.gain_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(g_p0, cfg.gain_floor)], dtype=np.float64),
        np.array(
            [
                softplus_inverse_np(b_en0, cfg.gain_floor),
                softplus_inverse_np(b_ep0, cfg.gain_floor),
            ],
            dtype=np.float64,
        ),
        beta0.astype(np.float64),
    ]


# ============================================================
# Fit one model
# ============================================================
def fit_one_model(seed: int, cfg: Config) -> dict[str, Any]:
    params0 = make_initial_params(seed, cfg)

    model = CTModel(
        NX,
        NY,
        NU,
        state_fcn=state_fcn,
        output_fcn=output_fcn,
        x0=np.zeros(NX, dtype=np.float64),
    )

    model.init(params0)

    model.loss(
        rho_x0=cfg.rho_x0,
        rho_th=cfg.rho_th,
        tau_th=cfg.tau_th,
    )

    model.optimization(
        adam_eta=cfg.adam_eta,
        adam_epochs=cfg.adam_epochs,
        lbfgs_epochs=cfg.lbfgs_epochs,
        iprint=cfg.iprint,
        memory=cfg.lbfgs_memory,
        lbfgs_tol=cfg.lbfgs_tol,
    )

    if DIFFRAX_AVAILABLE:
        model.integration_options(
            interpolation_type="zoh",
            ode_solver=diffrax.Heun(),
            dt0=Ts / 10.0,
            max_steps=100000,
        )

    model.fit(Y_id, U_id, T_id)

    Yhat, Xhat = model.predict(model.x0, U_id, T_id)

    Yhat = np.asarray(Yhat, dtype=np.float64).reshape(-1, 1)
    Xhat = np.asarray(Xhat, dtype=np.float64)

    yhat = Yhat.reshape(-1)
    ytrue = Y_id.reshape(-1)
    residual = ytrue - yhat

    phys = unpack_physical_params_np(model.params)

    Phi_fit, phi_names, aux = c4_feature_matrix_np(Xhat, I_id, cfg)

    rank_phi = matrix_rank_condition_svd_np(Phi_fit, cfg.svd_tol)
    rank_X = matrix_rank_condition_svd_np(Xhat, cfg.svd_tol)

    lb = ljung_box_test(residual, min(cfg.ljung_box_lag, max(len(residual) - 2, 1)))

    return {
        "seed": seed,
        "model": model,
        "params": model.params,
        "x0": np.asarray(model.x0, dtype=np.float64),
        "Yhat": Yhat,
        "Xhat": Xhat,
        "yhat": yhat,
        "residual": residual,
        "rmse": rmse(ytrue, yhat),
        "mae": mae(ytrue, yhat),
        "r2_percent": r2_percent(ytrue, yhat),
        "bfr_percent": bfr_percent(ytrue, yhat),
        "phys": phys,
        "Phi": Phi_fit,
        "phi_names": phi_names,
        "rank_phi": rank_phi,
        "rank_X": rank_X,
        "ljung_box": lb,
    }


# ============================================================
# Multistart loop
# ============================================================
results: list[dict[str, Any]] = []
failed_rows: list[dict[str, Any]] = []

print("\n" + "=" * 100)
print("MULTISTART CT-ID")
print("=" * 100)
print("Model:", MODEL_ID)
print("Seeds:", CFG.seed0, "to", CFG.seed0 + CFG.n_multistart - 1)
print("=" * 100)

for j in range(CFG.n_multistart):
    seed = CFG.seed0 + j

    print("\n" + "-" * 100)
    print(f"{MODEL_ID} | seed {seed} | {j + 1}/{CFG.n_multistart}")
    print("-" * 100)

    try:
        res = fit_one_model(seed, CFG)
        results.append(res)

        print(
            f"{MODEL_ID} seed={seed} | "
            f"RMSE={res['rmse']:.6e} | "
            f"MAE={res['mae']:.6e} | "
            f"R2={res['r2_percent']:.6f}% | "
            f"BFR={res['bfr_percent']:.6f}% | "
            f"rankPhi={res['rank_phi']['rank']}/{res['rank_phi']['n_cols']} | "
            f"rankX={res['rank_X']['rank']}/{res['rank_X']['n_cols']}"
        )

    except Exception as exc:
        print("[FAILED]", seed, repr(exc))
        failed_rows.append(
            {
                "model_id": MODEL_ID,
                "state_id": CFG.state_id,
                "candidate_id": CFG.candidate_id,
                "seed": seed,
                "error": repr(exc),
            }
        )

    gc.collect()


if len(results) == 0:
    pd.DataFrame(failed_rows).to_csv(OUT_DIR / "failed_runs.csv", index=False)
    raise RuntimeError("All multistart fits failed.")


# ============================================================
# Save outputs
# ============================================================
rows = []

for res in results:
    phys = res["phys"]

    row = {
        "model_id": MODEL_ID,
        "state_id": CFG.state_id,
        "candidate_id": CFG.candidate_id,
        "seed": int(res["seed"]),
        "rmse": float(res["rmse"]),
        "mae": float(res["mae"]),
        "r2_percent": float(res["r2_percent"]),
        "bfr_percent": float(res["bfr_percent"]),
        "rank_phi_raw": res["rank_phi"]["rank"],
        "ncols_phi_raw": res["rank_phi"]["n_cols"],
        "cond_phi_raw": res["rank_phi"]["condition_number"],
        "rank_X_raw": res["rank_X"]["rank"],
        "ncols_X_raw": res["rank_X"]["n_cols"],
        "cond_X_raw": res["rank_X"]["condition_number"],
        "ljung_box_Q": res["ljung_box"]["Q"],
        "ljung_box_p_value": res["ljung_box"]["p_value"],
        "alpha_n_hat": phys["alpha_n"],
        "alpha_p_hat": phys["alpha_p"],
        "g_n_hat": phys["g_n"],
        "g_p_hat": phys["g_p"],
        "b_en_hat": phys["b_en"],
        "b_ep_hat": phys["b_ep"],
    }

    for name in k_names_for_ne(STATE_SPEC["n_electrolyte"]):
        row[f"{name}_hat"] = phys[name]

    for i in range(1, STATE_SPEC["n_electrolyte"]):
        key = f"k_edge{i}_hat"
        row[key] = phys.get(f"k_edge{i}", np.nan)

    for name in BETA_NAMES:
        row[name] = phys[name]

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
df_best = df_all.head(1).copy()
best_seed = int(df_best.iloc[0]["seed"])
best_res = [r for r in results if int(r["seed"]) == best_seed][0]

df_failed = pd.DataFrame(failed_rows)
if len(df_failed) == 0:
    df_failed = pd.DataFrame(columns=["model_id", "state_id", "candidate_id", "seed", "error"])

df_all.to_csv(OUT_DIR / "all_runs.csv", index=False)
df_best.to_csv(OUT_DIR / "best_run.csv", index=False)
df_failed.to_csv(OUT_DIR / "failed_runs.csv", index=False)

# Save raw params and x0.
np.savez(
    OUT_DIR / "best_params_raw.npz",
    **{f"param_{i}": np.asarray(p, dtype=np.float64) for i, p in enumerate(best_res["params"])},
    x0=np.asarray(best_res["x0"], dtype=np.float64),
)

pd.DataFrame(
    {
        "t_s": T_id,
        "current_A": I_id,
        "measured_voltage_V": Y_id.reshape(-1),
        "estimated_voltage_V": best_res["yhat"],
        "residual_V": best_res["residual"],
    }
).to_csv(OUT_DIR / "best_measured_estimated_response.csv", index=False)

pd.DataFrame(best_res["Xhat"], columns=[f"x{i}" for i in range(best_res["Xhat"].shape[1])]).assign(
    t_s=T_id
).to_csv(OUT_DIR / "best_state_trajectory.csv", index=False)

pd.DataFrame(best_res["Phi"], columns=best_res["phi_names"]).assign(
    t_s=T_id
).to_csv(OUT_DIR / "best_feature_matrix_phi.csv", index=False)

# Parameter long table.
param_rows = []
for _, row in df_all.iterrows():
    for col in df_all.columns:
        if col.endswith("_hat") or col.startswith("beta_"):
            param_rows.append(
                {
                    "model_id": row["model_id"],
                    "state_id": row["state_id"],
                    "candidate_id": row["candidate_id"],
                    "seed": int(row["seed"]),
                    "parameter": col,
                    "value": row[col],
                    "rmse": row["rmse"],
                }
            )

pd.DataFrame(param_rows).to_csv(OUT_DIR / "parameter_long.csv", index=False)

# Beta table.
beta_cols = [c for c in df_all.columns if c.startswith("beta_")]
df_all[["model_id", "state_id", "candidate_id", "seed", "rmse"] + beta_cols].to_csv(
    OUT_DIR / "beta_coefficients.csv",
    index=False,
)

# Truth rank.
Phi_truth, truth_phi_names, truth_aux = c4_feature_matrix_np(X_truth, I_id,