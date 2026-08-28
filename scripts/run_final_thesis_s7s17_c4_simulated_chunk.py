#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_final_thesis_s7s17_c4_simulated_chunk.py

Fully simulated final-thesis CT-ID runner for the two final state candidates:

    S7_C4   : 2 negative solid states + 2 positive solid states + 3 electrolyte states
    S17_C4  : 4 negative solid states + 4 positive solid states + 9 electrolyte states

This replaces the older C1/C2/C3/C4 output-candidate sweep.  The output model is
fixed to the supervisor C4 form used in the direct-k/B continuation work:

    Vhat = C
         + xp + ap2*xp^2 + ap3*xp^3 + ap4*xp^4
         - (xn + an2*xn^2 + an3*xn^3 + an4*xn^4)
         + D1*I
         + ze + E2*ze^2 + E3*ze^3 + E4*ze^4

The linear coefficients of xp, xn, and ze are fixed to 1.  The trainable beta is:

    beta = [C, ap2, ap3, ap4, an2, an3, an4, D1, E2, E3, E4]

Important
---------
This is fully synthetic.  It does not load MPR files and does not use real cycle data.

Each Slurm array task should run one state model and one seed chunk.  A typical thesis run is:

    2 models x 10 chunks x 100 seeds = 2000 fits
    = 1000 fits per model

Main outputs per chunk
----------------------
results/final_thesis_ctid_s7s17_c4_1000/<run_tag>/
    all_runs.csv
    best_run.csv
    failed_runs.csv
    beta_coefficients.csv
    parameter_long.csv
    summary.csv
    config.json
    synthetic_id_data.csv
    truth_state_trajectory.csv
    truth_voltage_components.csv
    best_params_raw.npz
    best_measured_estimated_response.csv
    best_state_trajectory.csv
    best_feature_matrix_phi.csv
    best_voltage_components.csv
    best_manifest.csv
"""

from __future__ import annotations

import os
import gc
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------
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

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.linalg import block_diag, expm
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


# =====================================================================
# Configuration
# =====================================================================
@dataclass
class Config:
    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------
    output_root: str = "results/final_thesis_ctid_s7s17_c4_1000"
    figure_root: str = "results/figures/final_thesis_ctid_s7s17_c4_1000"

    # -----------------------------------------------------------------
    # Model choice
    # -----------------------------------------------------------------
    state_id: str = os.environ.get("UN_STATE_VARIANT", "S7")
    candidate_id: str = os.environ.get("UN_OUTPUT_CANDIDATE", "C4")

    # Truth-state choice.
    #   match: use the same state model as state_id.
    #   S7 or S17: force the same generated dataset for both fitted models.
    truth_state_id: str = os.environ.get("UN_TRUTH_STATE_VARIANT", "match")

    # -----------------------------------------------------------------
    # Synthetic simulation controls
    # -----------------------------------------------------------------
    sim_dt: float = float(os.environ.get("UN_SIM_DT", "0.1"))
    sim_t_end: float = float(os.environ.get("UN_SIM_T_END", "500.0"))

    enable_id_downsample: bool = os.environ.get("UN_ENABLE_ID_DOWNSAMPLE", "True").lower() == "true"
    id_downsample_dt: float = float(os.environ.get("UN_ID_DOWNSAMPLE_DT", "1.0"))
    id_downsample_use_interp: bool = os.environ.get("UN_ID_DOWNSAMPLE_USE_INTERP", "False").lower() == "true"

    input_profile: str = os.environ.get("UN_INPUT_PROFILE", "step")
    t_step: float = float(os.environ.get("UN_T_STEP", "90.0"))
    i_before: float = float(os.environ.get("UN_I_BEFORE", "0.0"))
    i_after: float = float(os.environ.get("UN_I_AFTER", "2.0"))

    add_noise: bool = os.environ.get("UN_ADD_NOISE", "False").lower() == "true"
    noise_std: float = float(os.environ.get("UN_NOISE_STD", "1e-4"))
    random_seed: int = int(os.environ.get("UN_RANDOM_SEED", "11"))

    # -----------------------------------------------------------------
    # Fitting seeds for this chunk
    # -----------------------------------------------------------------
    seed0: int = int(os.environ.get("UN_SEED0", "200"))
    n_multistart: int = int(os.environ.get("UN_N_MULTISTART", "100"))
    chunk_index: int = int(os.environ.get("UN_CHUNK_INDEX", "0"))

    # -----------------------------------------------------------------
    # Truth physical parameters for the synthetic data
    # -----------------------------------------------------------------
    alpha_n_true: float = float(os.environ.get("UN_ALPHA_N_TRUE", "0.0064"))
    alpha_p_true: float = float(os.environ.get("UN_ALPHA_P_TRUE", "0.0048"))

    # If UN_K_TRUE_CSV is empty, defaults depend on truth state:
    #   S7 : 0.050,0.040
    #   S17: 0.060,0.050,0.045,0.040,0.035
    k_true_csv: str = os.environ.get("UN_K_TRUE_CSV", "")

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

    # Signs
    solid_input_sign_n: float = +1.0
    solid_input_sign_p: float = +1.0
    electrolyte_input_left_sign: float = +1.0
    electrolyte_input_right_sign: float = -1.0

    # -----------------------------------------------------------------
    # Truth output beta for exact C4 synthetic voltage
    # -----------------------------------------------------------------
    beta_C_true: float = float(os.environ.get("UN_BETA_C_TRUE", "2.600"))
    beta_ap2_true: float = float(os.environ.get("UN_BETA_AP2_TRUE", "0.180"))
    beta_ap3_true: float = float(os.environ.get("UN_BETA_AP3_TRUE", "-0.060"))
    beta_ap4_true: float = float(os.environ.get("UN_BETA_AP4_TRUE", "0.015"))
    beta_an2_true: float = float(os.environ.get("UN_BETA_AN2_TRUE", "0.100"))
    beta_an3_true: float = float(os.environ.get("UN_BETA_AN3_TRUE", "-0.030"))
    beta_an4_true: float = float(os.environ.get("UN_BETA_AN4_TRUE", "0.008"))
    beta_D1_true: float = float(os.environ.get("UN_BETA_D1_TRUE", "-0.004"))
    beta_E2_true: float = float(os.environ.get("UN_BETA_E2_TRUE", "0.020"))
    beta_E3_true: float = float(os.environ.get("UN_BETA_E3_TRUE", "-0.010"))
    beta_E4_true: float = float(os.environ.get("UN_BETA_E4_TRUE", "0.002"))

    # -----------------------------------------------------------------
    # Cold-start initialization controls
    # -----------------------------------------------------------------
    init_dyn_jitter: float = float(os.environ.get("UN_INIT_DYN_JITTER", "0.30"))
    init_gain_jitter: float = float(os.environ.get("UN_INIT_GAIN_JITTER", "0.30"))
    init_C_jitter: float = float(os.environ.get("UN_INIT_C_JITTER", "0.05"))
    init_beta_scale: float = float(os.environ.get("UN_INIT_BETA_SCALE", "1e-2"))
    init_D1_center: float = float(os.environ.get("UN_INIT_D1_CENTER", "-0.004"))
    init_D1_jitter: float = float(os.environ.get("UN_INIT_D1_JITTER", "0.002"))

    alpha_n_nominal: float = float(os.environ.get("UN_ALPHA_N_INIT", "0.0064"))
    alpha_p_nominal: float = float(os.environ.get("UN_ALPHA_P_INIT", "0.0048"))
    k_direct_nominal: float = float(os.environ.get("UN_K_DIRECT_INIT", "0.05"))
    g_n_nominal: float = float(os.environ.get("UN_G_N_INIT", "1.20e-4"))
    g_p_nominal: float = float(os.environ.get("UN_G_P_INIT", "1.00e-4"))
    b_en_nominal: float = float(os.environ.get("UN_B_EN_INIT", "0.03"))
    b_ep_nominal: float = float(os.environ.get("UN_B_EP_INIT", "0.03"))

    dyn_floor: float = float(os.environ.get("UN_DYN_FLOOR", "1e-12"))
    gain_floor: float = float(os.environ.get("UN_GAIN_FLOOR", "1e-10"))

    # -----------------------------------------------------------------
    # Optimization controls
    # -----------------------------------------------------------------
    adam_epochs: int = int(os.environ.get("UN_ADAM_EPOCHS", "500"))
    adam_eta: float = float(os.environ.get("UN_ADAM_ETA", "2e-3"))
    lbfgs_epochs: int = int(os.environ.get("UN_LBFGS_EPOCHS", "8000"))
    lbfgs_tol: float = float(os.environ.get("UN_LBFGS_TOL", "1e-12"))
    lbfgs_memory: int = int(os.environ.get("UN_LBFGS_MEMORY", "30"))
    iprint: int = int(os.environ.get("UN_IPRINT", "0"))

    rho_x0: float = float(os.environ.get("UN_RHO_X0", "1e-8"))
    rho_th: float = float(os.environ.get("UN_RHO_TH", "1e-8"))
    tau_th: float = float(os.environ.get("UN_TAU_TH", "0.0"))

    # -----------------------------------------------------------------
    # Diagnostics and plotting
    # -----------------------------------------------------------------
    svd_tol: float = float(os.environ.get("UN_SVD_TOL", "1e-10"))
    ljung_box_lag: int = int(os.environ.get("UN_LJUNG_BOX_LAG", "20"))
    hist_bins: int = int(os.environ.get("UN_HIST_BINS", "1000"))

    make_plots: bool = os.environ.get("UN_MAKE_PLOTS", "True").lower() == "true"
    save_plots: bool = os.environ.get("UN_SAVE_PLOTS", "True").lower() == "true"
    show_plots: bool = os.environ.get("UN_SHOW_PLOTS", "False").lower() == "true"


CFG = Config()

STATE_VARIANTS: dict[str, dict[str, int]] = {
    "S7": {"n_solid_n": 2, "n_solid_p": 2, "n_electrolyte": 3},
    "S17": {"n_solid_n": 4, "n_solid_p": 4, "n_electrolyte": 9},
}

if CFG.state_id not in STATE_VARIANTS:
    raise ValueError("UN_STATE_VARIANT must be S7 or S17.")

if CFG.candidate_id != "C4":
    raise ValueError("This script is for the final C4 output only. Use UN_OUTPUT_CANDIDATE=C4.")

if CFG.truth_state_id == "match":
    TRUTH_STATE_ID = CFG.state_id
else:
    TRUTH_STATE_ID = CFG.truth_state_id

if TRUTH_STATE_ID not in STATE_VARIANTS:
    raise ValueError("UN_TRUTH_STATE_VARIANT must be match, S7, or S17.")

STATE_SPEC = STATE_VARIANTS[CFG.state_id]
TRUTH_STATE_SPEC = STATE_VARIANTS[TRUTH_STATE_ID]
NX = STATE_SPEC["n_solid_n"] + STATE_SPEC["n_solid_p"] + STATE_SPEC["n_electrolyte"]
TRUTH_NX = TRUTH_STATE_SPEC["n_solid_n"] + TRUTH_STATE_SPEC["n_solid_p"] + TRUTH_STATE_SPEC["n_electrolyte"]
DEGREE = 4
MODEL_ID = f"{CFG.state_id}_{CFG.candidate_id}"

RUN_TAG = os.environ.get(
    "UN_RUN_TAG",
    (
        f"final_thesis_{MODEL_ID}_chunk_{CFG.chunk_index}_"
        f"{CFG.n_multistart}seeds_{CFG.seed0}_to_{CFG.seed0 + CFG.n_multistart - 1}_"
        f"truth_{TRUTH_STATE_ID}_tf_{CFG.sim_t_end}_dt_{CFG.id_downsample_dt}"
    ),
)

OUT_DIR = Path(CFG.output_root) / RUN_TAG
FIG_DIR = Path(CFG.figure_root) / RUN_TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("FINAL THESIS FULLY SYNTHETIC CT-ID — S7/S17 C4")
print("=" * 100)
print("Working directory:", Path.cwd())
print("Threads:", N_THREADS)
print("JAX x64:", jax.config.read("jax_enable_x64"))
print("MODEL_ID:", MODEL_ID)
print("FIT STATE:", CFG.state_id, "NX=", NX)
print("TRUTH STATE:", TRUTH_STATE_ID, "TRUTH_NX=", TRUTH_NX)
print("RUN_TAG:", RUN_TAG)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("=" * 100)


# =====================================================================
# Basic helpers
# =====================================================================
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


def sse(y: np.ndarray, yh: np.ndarray) -> float:
    y = ensure_1d(y)
    yh = ensure_1d(yh)
    n = min(len(y), len(yh))
    e = y[:n] - yh[:n]
    return float(np.sum(e**2))


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


def relative_error_percent(true_value: float, estimate: float) -> float:
    denom = max(abs(float(true_value)), 1e-15)
    return float(100.0 * abs(float(estimate) - float(true_value)) / denom)


def softplus_inverse_np(y: float, floor: float = 0.0) -> float:
    z = max(float(y) - float(floor), 1e-12)
    if z > 30.0:
        return z
    return float(np.log(np.expm1(z)))


def positive_from_raw_np(raw: float, floor: float) -> float:
    return floor + float(np.log1p(np.exp(float(raw))))


def positive_from_raw_jax(raw, floor: float):
    return floor + jax.nn.softplus(raw)


def c2d_zoh(A: np.ndarray, B: np.ndarray, Ts: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m), dtype=np.float64)
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * Ts)
    return Md[:n, :n], Md[:n, n:]


def matrix_rank_condition_svd_np(M: np.ndarray, tol: float = 1e-10) -> dict[str, Any]:
    M0 = np.asarray(M, dtype=np.float64)
    if M0.ndim != 2 or M0.size == 0:
        return {
            "rank": 0,
            "n_rows": int(M0.shape[0]) if M0.ndim == 2 else 0,
            "n_cols": int(M0.shape[1]) if M0.ndim == 2 else 0,
            "condition_number": np.nan,
            "singular_values": [],
        }
    try:
        svals = np.linalg.svd(M0, compute_uv=False)
    except np.linalg.LinAlgError:
        return {
            "rank": np.nan,
            "n_rows": int(M0.shape[0]),
            "n_cols": int(M0.shape[1]),
            "condition_number": np.nan,
            "singular_values": [],
        }
    threshold = tol * max(float(svals[0]), 1e-30)
    rank = int(np.sum(svals > threshold))
    cond = float(svals[0] / max(float(svals[-1]), 1e-300))
    return {
        "rank": rank,
        "n_rows": int(M0.shape[0]),
        "n_cols": int(M0.shape[1]),
        "condition_number": cond,
        "singular_values": [float(v) for v in svals],
    }


def autocorr(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = ensure_1d(x, dtype=np.float64)
    x = x - np.mean(x)
    denom = float(np.dot(x, x)) + 1e-15
    lags = np.arange(0, max_lag + 1)
    vals = []
    for lag in lags:
        vals.append(float(np.dot(x[lag:], x[: len(x) - lag]) / denom))
    return lags, np.asarray(vals, dtype=np.float64)


def ljung_box_test(residual: np.ndarray, lag: int) -> dict[str, float]:
    e = ensure_1d(residual, dtype=np.float64)
    n = len(e)
    if n <= lag + 1:
        return {"Q": np.nan, "p_value": np.nan, "lag": lag}
    _, acf_vals = autocorr(e, lag)
    rho = acf_vals[1 : lag + 1]
    Q = n * (n + 2.0) * np.sum((rho**2) / (n - np.arange(1, lag + 1)))
    p_value = 1.0 - chi2.cdf(Q, df=lag)
    return {"Q": float(Q), "p_value": float(p_value), "lag": int(lag)}


def make_current_profile(t: np.ndarray, cfg: Config) -> np.ndarray:
    t = ensure_1d(t, dtype=np.float64)
    if cfg.input_profile != "step":
        raise ValueError("This final thesis synthetic run is configured for step input only.")
    return np.where(t < cfg.t_step, cfg.i_before, cfg.i_after)


def downsample_id_source(
    t: np.ndarray,
    u: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
    target_dt: float,
    use_interp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = ensure_1d(t, dtype=np.float64)
    u = ensure_1d(u, dtype=np.float64)
    y = ensure_1d(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if len(t) < 2:
        return t.copy(), u.copy(), y.copy(), X.copy()
    native_dt = float(np.median(np.diff(t)))
    if not np.isfinite(target_dt) or target_dt <= native_dt + 1e-12:
        return t.copy(), u.copy(), y.copy(), X.copy()
    if use_interp:
        t0 = float(t[0])
        tf = float(t[-1])
        n_steps = int(np.floor((tf - t0) / target_dt))
        t_new = t0 + target_dt * np.arange(n_steps + 1, dtype=np.float64)
        u_new = np.interp(t_new, t, u)
        y_new = np.interp(t_new, t, y)
        X_new = np.zeros((len(t_new), X.shape[1]), dtype=np.float64)
        for j in range(X.shape[1]):
            X_new[:, j] = np.interp(t_new, t, X[:, j])
        return t_new, u_new, y_new, X_new
    step = max(int(round(target_dt / native_dt)), 1)
    idx = np.arange(0, len(t), step, dtype=int)
    return t[idx].copy(), u[idx].copy(), y[idx].copy(), X[idx, :].copy()


def save_or_show(path: Path | None = None) -> None:
    if CFG.save_plots and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=230, bbox_inches="tight")
        print("[saved figure]", path)
    if CFG.show_plots:
        plt.show()
    plt.close()


# =====================================================================
# State-space builders: S7/S17 direct k/B parameterization
# =====================================================================
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


def n_k_independent(n_e: int) -> int:
    if n_e == 3:
        return 2
    if n_e == 9:
        return 5
    raise ValueError("This script supports only n_e=3 for S7 and n_e=9 for S17.")


def expand_edge_couplings_np(n_e: int, k_ind: np.ndarray) -> np.ndarray:
    k = np.asarray(k_ind, dtype=np.float64).reshape(-1)
    if n_e == 3:
        if len(k) != 2:
            raise ValueError("S7 requires 2 independent electrolyte couplings.")
        return np.array([k[0], k[1]], dtype=np.float64)
    if n_e == 9:
        if len(k) != 5:
            raise ValueError("S17 requires 5 independent electrolyte couplings.")
        return np.array([k[0], k[0], k[1], k[2], k[2], k[3], k[4], k[4]], dtype=np.float64)
    raise ValueError("Unsupported electrolyte size.")


def expand_edge_couplings_jax(n_e: int, k_ind):
    if n_e == 3:
        return jnp.array([k_ind[0], k_ind[1]], dtype=jnp.float64)
    if n_e == 9:
        return jnp.array(
            [
                k_ind[0], k_ind[0], k_ind[1], k_ind[2],
                k_ind[2], k_ind[3], k_ind[4], k_ind[4],
            ],
            dtype=jnp.float64,
        )
    raise ValueError("Unsupported electrolyte size.")


def build_solid_A_general(n: int, alpha: float) -> np.ndarray:
    if n == 2:
        return np.array(
            [[-8.0 * alpha, 8.0 * alpha], [8.0 * alpha, -8.0 * alpha]],
            dtype=np.float64,
        )
    if n < 2:
        raise ValueError("solid state count must be >= 2")
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


def build_solid_A_general_jax(n: int, alpha):
    if n == 2:
        return jnp.array(
            [[-8.0 * alpha, 8.0 * alpha], [8.0 * alpha, -8.0 * alpha]],
            dtype=jnp.float64,
        )
    lower = 4.0 * n * alpha
    upper = 6.0 * n * alpha
    A = jnp.zeros((n, n), dtype=jnp.float64)
    A = A.at[0, 0].set(-upper)
    A = A.at[0, 1].set(upper)
    for i in range(1, n - 1):
        A = A.at[i, i - 1].set(lower)
        A = A.at[i, i].set(-(lower + upper))
        A = A.at[i, i + 1].set(upper)
    A = A.at[n - 1, n - 2].set(lower)
    A = A.at[n - 1, n - 1].set(-lower)
    return A


def build_solid_B_general(n: int, gain: float, sign: float) -> np.ndarray:
    B = np.zeros((n, 1), dtype=np.float64)
    B[-1, 0] = sign * float(gain)
    return B


def build_electrolyte_A_direct_np(n_e: int, k_ind: np.ndarray) -> np.ndarray:
    edges = expand_edge_couplings_np(n_e, k_ind)
    A = np.zeros((n_e, n_e), dtype=np.float64)
    for j, w in enumerate(edges):
        A[j, j] -= w
        A[j, j + 1] += w
        A[j + 1, j] += w
        A[j + 1, j + 1] -= w
    return A


def build_electrolyte_A_direct_jax(n_e: int, k_ind):
    edges = expand_edge_couplings_jax(n_e, k_ind)
    A = jnp.zeros((n_e, n_e), dtype=jnp.float64)
    for j in range(n_e - 1):
        w = edges[j]
        A = A.at[j, j].add(-w)
        A = A.at[j, j + 1].add(w)
        A = A.at[j + 1, j].add(w)
        A = A.at[j + 1, j + 1].add(-w)
    return A


def build_electrolyte_B_direct_np(n_e: int, b_en: float, b_ep: float, cfg: Config) -> np.ndarray:
    B = np.zeros((n_e, 1), dtype=np.float64)
    if n_e == 3:
        B[0, 0] = cfg.electrolyte_input_left_sign * float(b_en)
        B[1, 0] = 0.0
        B[2, 0] = cfg.electrolyte_input_right_sign * float(b_ep)
        return B
    if n_e == 9:
        B[0:3, 0] = cfg.electrolyte_input_left_sign * float(b_en)
        B[3:6, 0] = 0.0
        B[6:9, 0] = cfg.electrolyte_input_right_sign * float(b_ep)
        return B
    raise ValueError("Unsupported electrolyte size.")


def build_electrolyte_B_direct_jax(n_e: int, b_en, b_ep, cfg: Config):
    B = jnp.zeros((n_e,), dtype=jnp.float64)
    if n_e == 3:
        B = B.at[0].set(cfg.electrolyte_input_left_sign * b_en)
        B = B.at[1].set(0.0)
        B = B.at[2].set(cfg.electrolyte_input_right_sign * b_ep)
        return B
    if n_e == 9:
        for idx in range(0, 3):
            B = B.at[idx].set(cfg.electrolyte_input_left_sign * b_en)
        for idx in range(3, 6):
            B = B.at[idx].set(0.0)
        for idx in range(6, 9):
            B = B.at[idx].set(cfg.electrolyte_input_right_sign * b_ep)
        return B
    raise ValueError("Unsupported electrolyte size.")


def assemble_A_B_np(
    state_spec: dict[str, int],
    alpha_n: float,
    alpha_p: float,
    k_ind: np.ndarray,
    g_n: float,
    g_p: float,
    b_en: float,
    b_ep: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_n = int(state_spec["n_solid_n"])
    n_p = int(state_spec["n_solid_p"])
    n_e = int(state_spec["n_electrolyte"])
    A_n = build_solid_A_general(n_n, alpha_n)
    A_p = build_solid_A_general(n_p, alpha_p)
    A_e = build_electrolyte_A_direct_np(n_e, k_ind)
    B_n = build_solid_B_general(n_n, g_n, cfg.solid_input_sign_n)
    B_p = build_solid_B_general(n_p, g_p, cfg.solid_input_sign_p)
    B_e = build_electrolyte_B_direct_np(n_e, b_en, b_ep, cfg)
    A = block_diag(A_n, A_p, A_e)
    B = np.vstack([B_n, B_p, B_e])
    idx = make_state_indices(state_spec)
    return A, B, idx


# =====================================================================
# C4 output features and truth beta
# =====================================================================
def beta_names_c4() -> list[str]:
    return [
        "C",
        "xp^2", "xp^3", "xp^4",
        "-xn^2", "-xn^3", "-xn^4",
        "I",
        "ze^2", "ze^3", "ze^4",
    ]


def truth_beta_c4(cfg: Config) -> np.ndarray:
    return np.array(
        [
            cfg.beta_C_true,
            cfg.beta_ap2_true,
            cfg.beta_ap3_true,
            cfg.beta_ap4_true,
            cfg.beta_an2_true,
            cfg.beta_an3_true,
            cfg.beta_an4_true,
            cfg.beta_D1_true,
            cfg.beta_E2_true,
            cfg.beta_E3_true,
            cfg.beta_E4_true,
        ],
        dtype=np.float64,
    )


def parse_truth_k(cfg: Config, state_spec: dict[str, int]) -> np.ndarray:
    n_e = int(state_spec["n_electrolyte"])
    n_k = n_k_independent(n_e)
    if cfg.k_true_csv.strip():
        vals = [float(v.strip()) for v in cfg.k_true_csv.split(",") if v.strip()]
        if len(vals) != n_k:
            raise ValueError(f"UN_K_TRUE_CSV must contain {n_k} values for {TRUTH_STATE_ID}.")
        return np.asarray(vals, dtype=np.float64)
    if n_e == 3:
        return np.asarray([0.050, 0.040], dtype=np.float64)
    if n_e == 9:
        return np.asarray([0.060, 0.050, 0.045, 0.040, 0.035], dtype=np.float64)
    raise ValueError("Unsupported truth state.")


def candidate_feature_matrix_np(
    X: np.ndarray,
    I: np.ndarray,
    state_spec: dict[str, int],
    theta_n0: float,
    theta_p0: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    X = np.asarray(X, dtype=np.float64)
    I = ensure_1d(I, dtype=np.float64)
    idx = make_state_indices(state_spec)
    x_n = theta_n0 + X[:, idx["idx_n_end"]]
    x_p = theta_p0 + X[:, idx["idx_p_end"]]
    x_n = np.clip(x_n, cfg.theta_min, cfg.theta_max)
    x_p = np.clip(x_p, cfg.theta_min, cfg.theta_max)
    ce_left = np.maximum(cfg.ce0 + X[:, idx["idx_e_left"]], cfg.ce_min)
    ce_right = np.maximum(cfg.ce0 + X[:, idx["idx_e_right"]], cfg.ce_min)
    z_e = np.maximum(ce_left / ce_right, 1e-12)
    fixed = x_p - x_n + z_e
    names = beta_names_c4()
    cols = [
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
    Phi = np.vstack(cols).T.astype(np.float64)
    return Phi, fixed, names, {
        "x_n": x_n,
        "x_p": x_p,
        "z_e": z_e,
        "ce_left": ce_left,
        "ce_right": ce_right,
        "fixed_xp_minus_xn_plus_ze": fixed,
    }


def split_components_np(
    X: np.ndarray,
    I: np.ndarray,
    phys: dict[str, Any],
    state_spec: dict[str, int],
    cfg: Config,
) -> dict[str, Any]:
    Phi, fixed, names, vars_ = candidate_feature_matrix_np(
        X=X,
        I=I,
        state_spec=state_spec,
        theta_n0=phys["theta_n0"],
        theta_p0=phys["theta_p0"],
        cfg=cfg,
    )
    beta = phys["beta"]
    yhat = fixed + Phi @ beta
    component_cols = {}
    for j, name in enumerate(names):
        component_cols[name] = Phi[:, j] * beta[j]
    x_p = vars_["x_p"]
    x_n = vars_["x_n"]
    z_e = vars_["z_e"]
    constant = component_cols["C"]
    ocp_positive = x_p + component_cols["xp^2"] + component_cols["xp^3"] + component_cols["xp^4"]
    ocp_negative = -x_n + component_cols["-xn^2"] + component_cols["-xn^3"] + component_cols["-xn^4"]
    current_branch = component_cols["I"]
    electrolyte_branch = z_e + component_cols["ze^2"] + component_cols["ze^3"] + component_cols["ze^4"]
    ocp_total = ocp_positive + ocp_negative
    return {
        "Phi": Phi,
        "fixed": fixed,
        "names": names,
        "V_hat": yhat,
        "constant": constant,
        "ocp_positive": ocp_positive,
        "ocp_negative": ocp_negative,
        "ocp_total": ocp_total,
        "current_branch": current_branch,
        "electrolyte_branch": electrolyte_branch,
        "x_n": vars_["x_n"],
        "x_p": vars_["x_p"],
        "z_e": vars_["z_e"],
        "ce_left": vars_["ce_left"],
        "ce_right": vars_["ce_right"],
        "component_cols": component_cols,
    }


# =====================================================================
# Synthetic truth generation
# =====================================================================
def simulate_synthetic_truth(cfg: Config) -> dict[str, Any]:
    t = np.arange(0.0, cfg.sim_t_end + 0.5 * cfg.sim_dt, cfg.sim_dt)
    I = make_current_profile(t, cfg)
    k_true = parse_truth_k(cfg, TRUTH_STATE_SPEC)
    beta_true = truth_beta_c4(cfg)

    A, B, idx = assemble_A_B_np(
        state_spec=TRUTH_STATE_SPEC,
        alpha_n=cfg.alpha_n_true,
        alpha_p=cfg.alpha_p_true,
        k_ind=k_true,
        g_n=cfg.g_n_true,
        g_p=cfg.g_p_true,
        b_en=cfg.b_en_true,
        b_ep=cfg.b_ep_true,
        cfg=cfg,
    )

    Ad, Bd = c2d_zoh(A, B, cfg.sim_dt)
    X = np.zeros((len(t), TRUTH_NX), dtype=np.float64)
    for k in range(len(t) - 1):
        X[k + 1, :] = (Ad @ X[k, :].reshape(-1, 1) + Bd * I[k]).reshape(-1)

    phys_true = {
        "theta_n0": cfg.theta_n0_true,
        "theta_p0": cfg.theta_p0_true,
        "beta": beta_true,
    }
    comps = split_components_np(
        X=X,
        I=I,
        phys=phys_true,
        state_spec=TRUTH_STATE_SPEC,
        cfg=cfg,
    )
    V = np.asarray(comps["V_hat"], dtype=np.float64).reshape(-1)

    if cfg.add_noise:
        rng = np.random.default_rng(cfg.random_seed)
        V = V + rng.normal(0.0, cfg.noise_std, size=len(V))

    return {
        "t": t,
        "I": I,
        "X": X,
        "V": V,
        "A": A,
        "B": B,
        "idx": idx,
        "k_ind_true": k_true,
        "k_edges_true": expand_edge_couplings_np(int(TRUTH_STATE_SPEC["n_electrolyte"]), k_true),
        "beta_true": beta_true,
        "eig_cont": np.linalg.eigvals(A),
        "components": comps,
    }


truth = simulate_synthetic_truth(CFG)

t = truth["t"]
I = truth["I"]
X_truth = truth["X"]
V_truth = truth["V"]

print("\nSynthetic truth summary:")
print("  truth_state:", TRUTH_STATE_ID)
print("  sim_dt:", CFG.sim_dt)
print("  samples:", len(t))
print("  V start/final:", float(V_truth[0]), float(V_truth[-1]))
print("  V min/max:", float(np.min(V_truth)), float(np.max(V_truth)))
print("  k_ind_true:", truth["k_ind_true"])
print("  k_edges_true:", truth["k_edges_true"])
print("  continuous poles:")
print(np.sort_complex(truth["eig_cont"]))

if CFG.enable_id_downsample:
    t_id, i_id, v_id, X_id = downsample_id_source(
        t=t,
        u=I,
        y=V_truth,
        X=X_truth,
        target_dt=CFG.id_downsample_dt,
        use_interp=CFG.id_downsample_use_interp,
    )
else:
    t_id, i_id, v_id, X_id = t.copy(), I.copy(), V_truth.copy(), X_truth.copy()

Ts = float(np.median(np.diff(t_id))) if len(t_id) >= 2 else CFG.sim_dt
T_id = t_id - t_id[0]
U_id = ensure_2d_col(i_id, dtype=np.float64)
Y_id = ensure_2d_col(v_id, dtype=np.float64)

print("\nCT-ID signal summary:")
print("  ID samples:", len(T_id))
print("  ID Ts:", Ts)
print("  Y shape:", Y_id.shape)
print("  U shape:", U_id.shape)

pd.DataFrame(
    {
        "t_s": T_id,
        "current_A_discharge_positive": i_id,
        "synthetic_voltage_V": v_id,
    }
).to_csv(OUT_DIR / "synthetic_id_data.csv", index=False)

state_data_truth = {"t_s": t_id}
for j in range(X_id.shape[1]):
    state_data_truth[f"x_truth_{j}"] = X_id[:, j]
pd.DataFrame(state_data_truth).to_csv(OUT_DIR / "truth_state_trajectory.csv", index=False)

truth_comps_full = truth["components"]
truth_component_data = {
    "t_s": t,
    "current_A_discharge_positive": I,
    "voltage_V": V_truth,
    "constant": truth_comps_full["constant"],
    "ocp_positive": truth_comps_full["ocp_positive"],
    "ocp_negative": truth_comps_full["ocp_negative"],
    "ocp_total": truth_comps_full["ocp_total"],
    "current_branch": truth_comps_full["current_branch"],
    "electrolyte_branch": truth_comps_full["electrolyte_branch"],
    "x_n": truth_comps_full["x_n"],
    "x_p": truth_comps_full["x_p"],
    "z_e": truth_comps_full["z_e"],
    "ce_left": truth_comps_full["ce_left"],
    "ce_right": truth_comps_full["ce_right"],
}
pd.DataFrame(truth_component_data).to_csv(OUT_DIR / "truth_voltage_components.csv", index=False)

if CFG.make_plots:
    plt.figure(figsize=(12.0, 5.6))
    plt.plot(T_id, v_id, linewidth=2.3)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Synthetic voltage data, truth={TRUTH_STATE_ID}, fit model={MODEL_ID}")
    plt.tight_layout()
    save_or_show(FIG_DIR / "synthetic_voltage_data.png")

    plt.figure(figsize=(12.0, 4.6))
    plt.plot(T_id, i_id, linewidth=2.0)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title("Synthetic step current input")
    plt.tight_layout()
    save_or_show(FIG_DIR / "synthetic_current_input.png")


# Truth feature rank for the generated truth state trajectory
Phi_truth, _, Phi_names, _ = candidate_feature_matrix_np(
    X_id,
    i_id,
    state_spec=TRUTH_STATE_SPEC,
    theta_n0=CFG.theta_n0_true,
    theta_p0=CFG.theta_p0_true,
    cfg=CFG,
)
truth_phi_rank = matrix_rank_condition_svd_np(Phi_truth, CFG.svd_tol)

print("\nTruth C4 feature rank check:")
print("  columns:", Phi_names)
print(f"  raw rank: {truth_phi_rank['rank']}/{truth_phi_rank['n_cols']}")
print(f"  condition number: {truth_phi_rank['condition_number']:.6e}")

pd.DataFrame(
    [
        {
            "model_id": MODEL_ID,
            "truth_state_id": TRUTH_STATE_ID,
            "columns": ", ".join(Phi_names),
            "rank": truth_phi_rank["rank"],
            "n_cols": truth_phi_rank["n_cols"],
            "condition_number": truth_phi_rank["condition_number"],
            "singular_values_json": json.dumps(truth_phi_rank["singular_values"]),
        }
    ]
).to_csv(OUT_DIR / "truth_phi_rank.csv", index=False)


# =====================================================================
# Parameter pack / unpack / initialization
# =====================================================================
def make_initial_beta_c4(seed: int, y0: float, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta = np.zeros(len(beta_names_c4()), dtype=np.float64)

    # At zero initial states:
    # x_p = theta_p0, x_n = theta_n0, z_e = ce0/ce0 = 1.
    # Since the output has fixed branch x_p - x_n + z_e,
    # initialize C near y0 - fixed0, not y0.
    fixed0 = cfg.theta_p0_true - cfg.theta_n0_true + 1.0
    beta[0] = float(y0) - fixed0 + cfg.init_C_jitter * rng.normal()

    beta[1:4] = cfg.init_beta_scale * rng.normal(size=3)
    beta[4:7] = cfg.init_beta_scale * rng.normal(size=3)
    beta[7] = cfg.init_D1_center + cfg.init_D1_jitter * rng.normal()
    beta[8:11] = cfg.init_beta_scale * rng.normal(size=3)

    return beta


def make_initial_params(seed: int, y0: float, cfg: Config) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    n_e = int(STATE_SPEC["n_electrolyte"])
    n_k = n_k_independent(n_e)
    alpha_n0 = cfg.alpha_n_nominal * np.exp(cfg.init_dyn_jitter * rng.normal())
    alpha_p0 = cfg.alpha_p_nominal * np.exp(cfg.init_dyn_jitter * rng.normal())
    k0 = cfg.k_direct_nominal * np.exp(cfg.init_dyn_jitter * rng.normal(size=n_k))
    g_n0 = cfg.g_n_nominal * np.exp(cfg.init_gain_jitter * rng.normal())
    g_p0 = cfg.g_p_nominal * np.exp(cfg.init_gain_jitter * rng.normal())
    b_e0 = np.array(
        [
            cfg.b_en_nominal * np.exp(cfg.init_gain_jitter * rng.normal()),
            cfg.b_ep_nominal * np.exp(cfg.init_gain_jitter * rng.normal()),
        ],
        dtype=np.float64,
    )
    beta0 = make_initial_beta_c4(seed, y0, cfg)
    params = [
        np.array([softplus_inverse_np(alpha_n0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(alpha_p0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(v, cfg.dyn_floor) for v in k0], dtype=np.float64),
        np.array([softplus_inverse_np(g_n0, cfg.gain_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(g_p0, cfg.gain_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(v, cfg.gain_floor) for v in b_e0], dtype=np.float64),
        np.asarray(beta0, dtype=np.float64),
    ]
    x0 = np.zeros(NX, dtype=np.float64)
    return params, x0


def unpack_params_np(params: list[np.ndarray], state_spec: dict[str, int], cfg: Config) -> dict[str, Any]:
    raw_alpha_n = float(np.asarray(params[0]).reshape(-1)[0])
    raw_alpha_p = float(np.asarray(params[1]).reshape(-1)[0])
    raw_k = np.asarray(params[2], dtype=np.float64).reshape(-1)
    raw_g_n = float(np.asarray(params[3]).reshape(-1)[0])
    raw_g_p = float(np.asarray(params[4]).reshape(-1)[0])
    raw_b_e = np.asarray(params[5], dtype=np.float64).reshape(-1)
    beta = np.asarray(params[6], dtype=np.float64).reshape(-1)
    alpha_n = positive_from_raw_np(raw_alpha_n, cfg.dyn_floor)
    alpha_p = positive_from_raw_np(raw_alpha_p, cfg.dyn_floor)
    k_ind = np.array([positive_from_raw_np(v, cfg.dyn_floor) for v in raw_k], dtype=np.float64)
    g_n = positive_from_raw_np(raw_g_n, cfg.gain_floor)
    g_p = positive_from_raw_np(raw_g_p, cfg.gain_floor)
    b_en = positive_from_raw_np(raw_b_e[0], cfg.gain_floor)
    b_ep = positive_from_raw_np(raw_b_e[1], cfg.gain_floor)
    A, B, idx = assemble_A_B_np(
        state_spec=state_spec,
        alpha_n=alpha_n,
        alpha_p=alpha_p,
        k_ind=k_ind,
        g_n=g_n,
        g_p=g_p,
        b_en=b_en,
        b_ep=b_ep,
        cfg=cfg,
    )
    return {
        "alpha_n": alpha_n,
        "alpha_p": alpha_p,
        "k_ind": k_ind,
        "k_edges": expand_edge_couplings_np(int(state_spec["n_electrolyte"]), k_ind),
        "g_n": g_n,
        "g_p": g_p,
        "b_en": b_en,
        "b_ep": b_ep,
        "theta_n0": cfg.theta_n0_true,
        "theta_p0": cfg.theta_p0_true,
        "beta": beta,
        "A": A,
        "B": B,
        "idx": idx,
        "ct_poles": np.linalg.eigvals(A),
    }


def unpack_params_jax(params, cfg: Config):
    raw_alpha_n = params[0][0]
    raw_alpha_p = params[1][0]
    raw_k = params[2]
    raw_g_n = params[3][0]
    raw_g_p = params[4][0]
    raw_b_e = params[5]
    beta = params[6]
    alpha_n = positive_from_raw_jax(raw_alpha_n, cfg.dyn_floor)
    alpha_p = positive_from_raw_jax(raw_alpha_p, cfg.dyn_floor)
    k_ind = cfg.dyn_floor + jax.nn.softplus(raw_k)
    g_n = positive_from_raw_jax(raw_g_n, cfg.gain_floor)
    g_p = positive_from_raw_jax(raw_g_p, cfg.gain_floor)
    b_en = positive_from_raw_jax(raw_b_e[0], cfg.gain_floor)
    b_ep = positive_from_raw_jax(raw_b_e[1], cfg.gain_floor)
    return {
        "alpha_n": alpha_n,
        "alpha_p": alpha_p,
        "k_ind": k_ind,
        "g_n": g_n,
        "g_p": g_p,
        "b_en": b_en,
        "b_ep": b_ep,
        "theta_n0": cfg.theta_n0_true,
        "theta_p0": cfg.theta_p0_true,
        "beta": beta,
    }


# =====================================================================
# JAX CT model factories
# =====================================================================
def make_state_fcn(state_spec: dict[str, int], cfg: Config):
    n_n = int(state_spec["n_solid_n"])
    n_p = int(state_spec["n_solid_p"])
    n_e = int(state_spec["n_electrolyte"])
    idx = make_state_indices(state_spec)

    def state_fcn(x, u, t, params):
        phys = unpack_params_jax(params, cfg)
        alpha_n = phys["alpha_n"]
        alpha_p = phys["alpha_p"]
        k_ind = phys["k_ind"]
        g_n = phys["g_n"]
        g_p = phys["g_p"]
        b_en = phys["b_en"]
        b_ep = phys["b_ep"]
        I_in = u[0]
        xn = x[idx["idx_n_start"]: idx["idx_n_start"] + n_n]
        xp = x[idx["idx_p_start"]: idx["idx_p_start"] + n_p]
        ce = x[idx["idx_e_start"]: idx["idx_e_start"] + n_e]
        A_n = build_solid_A_general_jax(n_n, alpha_n)
        A_p = build_solid_A_general_jax(n_p, alpha_p)
        A_e = build_electrolyte_A_direct_jax(n_e, k_ind)
        dxn = A_n @ xn
        dxp = A_p @ xp
        dce = A_e @ ce
        dxn = dxn.at[n_n - 1].add(cfg.solid_input_sign_n * g_n * I_in)
        dxp = dxp.at[n_p - 1].add(cfg.solid_input_sign_p * g_p * I_in)
        B_e = build_electrolyte_B_direct_jax(n_e, b_en, b_ep, cfg)
        dce = dce + B_e * I_in
        return jnp.concatenate([dxn, dxp, dce])

    return state_fcn


def make_output_fcn(state_spec: dict[str, int], cfg: Config):
    idx = make_state_indices(state_spec)

    def output_fcn(x, u, t, params):
        phys = unpack_params_jax(params, cfg)
        theta_n0 = phys["theta_n0"]
        theta_p0 = phys["theta_p0"]
        beta = phys["beta"]
        I_in = u[0]
        x_n = theta_n0 + x[idx["idx_n_end"]]
        x_p = theta_p0 + x[idx["idx_p_end"]]
        x_n = jnp.clip(x_n, cfg.theta_min, cfg.theta_max)
        x_p = jnp.clip(x_p, cfg.theta_min, cfg.theta_max)
        ce_left = jnp.maximum(cfg.ce0 + x[idx["idx_e_left"]], cfg.ce_min)
        ce_right = jnp.maximum(cfg.ce0 + x[idx["idx_e_right"]], cfg.ce_min)
        z_e = jnp.maximum(ce_left / ce_right, 1e-12)
        fixed = x_p - x_n + z_e
        feats = jnp.array(
            [
                1.0,
                x_p**2,
                x_p**3,
                x_p**4,
                -(x_n**2),
                -(x_n**3),
                -(x_n**4),
                I_in,
                z_e**2,
                z_e**3,
                z_e**4,
            ],
            dtype=jnp.float64,
        )
        y = fixed + jnp.dot(feats, beta)
        return jnp.array([y], dtype=jnp.float64)

    return output_fcn


# =====================================================================
# Fit one seed
# =====================================================================
def fit_one_seed(seed: int) -> dict[str, Any]:
    y0 = float(Y_id.reshape(-1)[0])
    params0, x0_init = make_initial_params(seed, y0, CFG)

    model = CTModel(
        NX,
        1,
        1,
        state_fcn=make_state_fcn(STATE_SPEC, CFG),
        output_fcn=make_output_fcn(STATE_SPEC, CFG),
        x0=x0_init,
    )

    model.init(params0)
    model.loss(rho_x0=CFG.rho_x0, rho_th=CFG.rho_th, tau_th=CFG.tau_th)
    model.optimization(
        adam_eta=CFG.adam_eta,
        adam_epochs=CFG.adam_epochs,
        lbfgs_epochs=CFG.lbfgs_epochs,
        iprint=CFG.iprint,
        memory=CFG.lbfgs_memory,
        lbfgs_tol=CFG.lbfgs_tol,
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

    phys = unpack_params_np(model.params, STATE_SPEC, CFG)
    comps = split_components_np(X=Xhat, I=i_id, phys=phys, state_spec=STATE_SPEC, cfg=CFG)
    residual = ytrue - yhat
    rank_phi_raw = matrix_rank_condition_svd_np(comps["Phi"], CFG.svd_tol)
    rank_X_raw = matrix_rank_condition_svd_np(Xhat, CFG.svd_tol)
    lb = ljung_box_test(residual, CFG.ljung_box_lag)

    result = {
        "state_id": CFG.state_id,
        "candidate_id": CFG.candidate_id,
        "model_id": MODEL_ID,
        "truth_state_id": TRUTH_STATE_ID,
        "degree": DEGREE,
        "nx": NX,
        "seed": seed,
        "chunk_index": CFG.chunk_index,
        "Yhat": Yhat,
        "Xhat": Xhat,
        "yhat": yhat,
        "residual": residual,
        "rmse": rmse(ytrue, yhat),
        "mae": mae(ytrue, yhat),
        "r2_percent": r2_percent(ytrue, yhat),
        "bfr_percent": bfr_percent(ytrue, yhat),
        "sse": sse(ytrue, yhat),
        "phys": phys,
        "components": comps,
        "rank_phi_raw": rank_phi_raw,
        "rank_X_raw": rank_X_raw,
        "ljung_box_Q": lb["Q"],
        "ljung_box_p_value": lb["p_value"],
        "raw_params": [np.asarray(p, dtype=np.float64) for p in model.params],
        "x0": np.asarray(model.x0, dtype=np.float64).reshape(-1),
    }

    del model
    gc.collect()
    try:
        jax.clear_caches()
    except Exception:
        pass
    return result


# =====================================================================
# Run multistart chunk
# =====================================================================
all_results = []
fail_rows = []

print("\n" + "=" * 100)
print("STARTING FULLY SYNTHETIC S7/S17 C4 CHUNK")
print("=" * 100)
print("MODEL_ID:", MODEL_ID)
print("Truth state:", TRUTH_STATE_ID)
print("Chunk index:", CFG.chunk_index)
print("Seeds:", CFG.seed0, "to", CFG.seed0 + CFG.n_multistart - 1)
print("=" * 100)

for k in range(CFG.n_multistart):
    seed = CFG.seed0 + k
    print("\n" + "-" * 100)
    print(f"{MODEL_ID} | truth={TRUTH_STATE_ID} | seed {seed} | {k + 1}/{CFG.n_multistart}")
    print("-" * 100)
    try:
        res = fit_one_seed(seed)
        all_results.append(res)
        print(
            f"{MODEL_ID} seed={seed} | "
            f"RMSE={res['rmse']:.6e} | "
            f"MAE={res['mae']:.6e} | "
            f"R2={res['r2_percent']:.6f}% | "
            f"BFR={res['bfr_percent']:.6f}% | "
            f"rankPhi={res['rank_phi_raw']['rank']}/{res['rank_phi_raw']['n_cols']} | "
            f"rankX={res['rank_X_raw']['rank']}/{res['rank_X_raw']['n_cols']}"
        )
    except Exception as exc:
        print(f"[FAIL] {MODEL_ID} seed={seed}: {repr(exc)}")
        fail_rows.append(
            {
                "model_id": MODEL_ID,
                "state_id": CFG.state_id,
                "truth_state_id": TRUTH_STATE_ID,
                "chunk_index": CFG.chunk_index,
                "seed": seed,
                "error": repr(exc),
            }
        )

pd.DataFrame(fail_rows).to_csv(OUT_DIR / "failed_runs.csv", index=False)

if len(all_results) == 0:
    raise RuntimeError("All fits failed in this chunk.")


# =====================================================================
# Build tables
# =====================================================================
truth_param_values: dict[str, float] = {
    "alpha_n_hat": CFG.alpha_n_true,
    "alpha_p_hat": CFG.alpha_p_true,
    "g_n_hat": CFG.g_n_true,
    "g_p_hat": CFG.g_p_true,
    "b_en_hat": CFG.b_en_true,
    "b_ep_hat": CFG.b_ep_true,
    "theta_n0_fixed": CFG.theta_n0_true,
    "theta_p0_fixed": CFG.theta_p0_true,
}

for idx_k, val in enumerate(truth["k_ind_true"], start=1):
    truth_param_values[f"k{idx_k}_hat"] = float(val)
for idx_k, val in enumerate(truth["k_edges_true"], start=1):
    truth_param_values[f"k_edge{idx_k}_hat"] = float(val)
for name, val in zip(beta_names_c4(), truth["beta_true"]):
    safe = name.replace("^", "pow").replace("-", "minus").replace(" ", "_")
    truth_param_values[f"beta_{safe}"] = float(val)

rows = []
beta_rows = []
param_rows = []

for r in all_results:
    phys = r["phys"]
    comps = r["components"]
    beta = phys["beta"]
    beta_names = comps["names"]

    row = {
        "model_id": MODEL_ID,
        "state_id": CFG.state_id,
        "truth_state_id": TRUTH_STATE_ID,
        "candidate_id": CFG.candidate_id,
        "degree": DEGREE,
        "nx": NX,
        "chunk_index": CFG.chunk_index,
        "seed": r["seed"],
        "rmse": r["rmse"],
        "mae": r["mae"],
        "r2_percent": r["r2_percent"],
        "bfr_percent": r["bfr_percent"],
        "sse": r["sse"],
        "alpha_n_hat": phys["alpha_n"],
        "alpha_p_hat": phys["alpha_p"],
        "g_n_hat": phys["g_n"],
        "g_p_hat": phys["g_p"],
        "b_en_hat": phys["b_en"],
        "b_ep_hat": phys["b_ep"],
        "theta_n0_fixed": phys["theta_n0"],
        "theta_p0_fixed": phys["theta_p0"],
        "rank_phi_raw": r["rank_phi_raw"]["rank"],
        "ncols_phi_raw": r["rank_phi_raw"]["n_cols"],
        "cond_phi_raw": r["rank_phi_raw"]["condition_number"],
        "rank_X_raw": r["rank_X_raw"]["rank"],
        "ncols_X_raw": r["rank_X_raw"]["n_cols"],
        "cond_X_raw": r["rank_X_raw"]["condition_number"],
        "ljung_box_Q": r["ljung_box_Q"],
        "ljung_box_p_value": r["ljung_box_p_value"],
    }

    for idx_k, val in enumerate(phys["k_ind"], start=1):
        row[f"k{idx_k}_hat"] = float(val)
    for idx_k, val in enumerate(phys["k_edges"], start=1):
        row[f"k_edge{idx_k}_hat"] = float(val)

    for j, name in enumerate(beta_names):
        safe = name.replace("^", "pow").replace("-", "minus").replace(" ", "_")
        col = f"beta_{safe}"
        row[col] = float(beta[j])
        beta_rows.append(
            {
                "model_id": MODEL_ID,
                "state_id": CFG.state_id,
                "truth_state_id": TRUTH_STATE_ID,
                "chunk_index": CFG.chunk_index,
                "seed": r["seed"],
                "beta_name": name,
                "beta_hat": float(beta[j]),
                "beta_true": truth_param_values.get(col, np.nan),
            }
        )

    # Add error columns where a matching truth exists.
    for p, tv in truth_param_values.items():
        if p in row:
            if p.endswith("_hat"):
                err_col = p.replace("_hat", "_error_percent")
            elif p.endswith("_fixed"):
                err_col = p.replace("_fixed", "_error_percent")
            else:
                err_col = f"{p}_error_percent"
            row[err_col] = relative_error_percent(tv, row[p])
            param_rows.append(
                {
                    "model_id": MODEL_ID,
                    "state_id": CFG.state_id,
                    "truth_state_id": TRUTH_STATE_ID,
                    "chunk_index": CFG.chunk_index,
                    "seed": r["seed"],
                    "parameter": p,
                    "estimate": row[p],
                    "true_value": tv,
                    "error_percent": row[err_col],
                }
            )

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
df_beta = pd.DataFrame(beta_rows)
df_params = pd.DataFrame(param_rows)
df_best = df_all.head(1).copy()
best_result = sorted(all_results, key=lambda r: float(r["rmse"]))[0]

df_all.to_csv(OUT_DIR / "all_runs.csv", index=False)
df_best.to_csv(OUT_DIR / "best_run.csv", index=False)
df_beta.to_csv(OUT_DIR / "beta_coefficients.csv", index=False)
df_params.to_csv(OUT_DIR / "parameter_long.csv", index=False)


def safe_int_or_nan(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return int(x)
    except Exception:
        return np.nan


def safe_float_or_nan(x):
    try:
        x = float(x)
        if not np.isfinite(x):
            return np.nan
        return x
    except Exception:
        return np.nan


summary = {
    "model_id": MODEL_ID,
    "state_id": CFG.state_id,
    "truth_state_id": TRUTH_STATE_ID,
    "chunk_index": CFG.chunk_index,
    "n_success": int(len(df_all)),
    "n_fail": int(len(fail_rows)),
    "best_seed": int(df_best.iloc[0]["seed"]),
    "best_rmse": float(df_best.iloc[0]["rmse"]),
    "best_mae": float(df_best.iloc[0]["mae"]),
    "best_r2_percent": float(df_best.iloc[0]["r2_percent"]),
    "best_bfr_percent": float(df_best.iloc[0]["bfr_percent"]),
    "median_rmse": float(df_all["rmse"].median()),
    "mean_rmse": float(df_all["rmse"].mean()),
    "std_rmse": float(df_all["rmse"].std(ddof=1)) if len(df_all) > 1 else np.nan,
    "truth_phi_rank": safe_int_or_nan(truth_phi_rank["rank"]),
    "truth_phi_ncols": safe_int_or_nan(truth_phi_rank["n_cols"]),
    "truth_phi_condition": safe_float_or_nan(truth_phi_rank["condition_number"]),
    "best_rank_phi_raw": safe_int_or_nan(df_best.iloc[0]["rank_phi_raw"]),
    "best_ncols_phi_raw": safe_int_or_nan(df_best.iloc[0]["ncols_phi_raw"]),
    "best_cond_phi_raw": safe_float_or_nan(df_best.iloc[0]["cond_phi_raw"]),
    "best_rank_X_raw": safe_int_or_nan(df_best.iloc[0]["rank_X_raw"]),
    "best_ncols_X_raw": safe_int_or_nan(df_best.iloc[0]["ncols_X_raw"]),
    "best_cond_X_raw": safe_float_or_nan(df_best.iloc[0]["cond_X_raw"]),
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)

print("\nBest run:")
print(df_best.to_string(index=False))


# =====================================================================
# Save best raw params and best traces
# =====================================================================
best_params_path = OUT_DIR / "best_params_raw.npz"
save_payload = {
    "x0": best_result["x0"],
    "seed": np.array([best_result["seed"]], dtype=np.int64),
    "rmse": np.array([best_result["rmse"]], dtype=np.float64),
    "model_id": np.array([MODEL_ID]),
    "state_id": np.array([CFG.state_id]),
    "truth_state_id": np.array([TRUTH_STATE_ID]),
}
for k, p in enumerate(best_result["raw_params"]):
    save_payload[f"param_{k}"] = np.asarray(p, dtype=np.float64)
np.savez(best_params_path, **save_payload)
print("[saved best raw params]", best_params_path)

t_best = ensure_1d(T_id)
i_best = ensure_1d(i_id)
y_meas = ensure_1d(Y_id)
y_hat = ensure_1d(best_result["yhat"])
residual = ensure_1d(best_result["residual"])
Xhat = np.asarray(best_result["Xhat"], dtype=np.float64)
comps = best_result["components"]
n = min(len(t_best), len(i_best), len(y_meas), len(y_hat), len(residual))

df_response = pd.DataFrame(
    {
        "t_s": t_best[:n],
        "current_A_discharge_positive": i_best[:n],
        "synthetic_voltage_V": y_meas[:n],
        "estimated_voltage_V": y_hat[:n],
        "residual_V": residual[:n],
        "abs_residual_V": np.abs(residual[:n]),
    }
)
response_csv = OUT_DIR / "best_measured_estimated_response.csv"
df_response.to_csv(response_csv, index=False)

state_data = {"t_s": t_best[:n]}
for j in range(Xhat.shape[1]):
    state_data[f"xhat_{j}"] = Xhat[:n, j]
state_csv = OUT_DIR / "best_state_trajectory.csv"
pd.DataFrame(state_data).to_csv(state_csv, index=False)

Phi = np.asarray(comps["Phi"], dtype=np.float64)
phi_data = {"t_s": t_best[:n]}
for j in range(Phi.shape[1]):
    phi_data[f"phi_{j}"] = Phi[:n, j]
phi_csv = OUT_DIR / "best_feature_matrix_phi.csv"
pd.DataFrame(phi_data).to_csv(phi_csv, index=False)

component_data = {
    "t_s": t_best[:n],
    "constant": comps["constant"][:n],
    "ocp_positive": comps["ocp_positive"][:n],
    "ocp_negative": comps["ocp_negative"][:n],
    "ocp_total": comps["ocp_total"][:n],
    "current_branch": comps["current_branch"][:n],
    "electrolyte_branch": comps["electrolyte_branch"][:n],
    "x_n": comps["x_n"][:n],
    "x_p": comps["x_p"][:n],
    "z_e": comps["z_e"][:n],
    "ce_left": comps["ce_left"][:n],
    "ce_right": comps["ce_right"][:n],
}
component_csv = OUT_DIR / "best_voltage_components.csv"
pd.DataFrame(component_data).to_csv(component_csv, index=False)

manifest = {
    "model_id": MODEL_ID,
    "state_id": CFG.state_id,
    "truth_state_id": TRUTH_STATE_ID,
    "chunk_index": CFG.chunk_index,
    "seed": int(best_result["seed"]),
    "rmse": float(best_result["rmse"]),
    "mae": float(best_result["mae"]),
    "r2_percent": float(best_result["r2_percent"]),
    "bfr_percent": float(best_result["bfr_percent"]),
    "best_params_raw": str(best_params_path),
    "response_csv": str(response_csv),
    "state_csv": str(state_csv),
    "phi_csv": str(phi_csv),
    "component_csv": str(component_csv),
}
pd.DataFrame([manifest]).to_csv(OUT_DIR / "best_manifest.csv", index=False)

config_payload = {
    "config": asdict(CFG),
    "run_tag": RUN_TAG,
    "model_id": MODEL_ID,
    "state_id": CFG.state_id,
    "truth_state_id": TRUTH_STATE_ID,
    "state_spec": STATE_SPEC,
    "truth_state_spec": TRUTH_STATE_SPEC,
    "candidate": {
        "candidate_id": CFG.candidate_id,
        "degree": DEGREE,
        "equation": (
            "C + xp + ap2*xp^2 + ap3*xp^3 + ap4*xp^4 "
            "- (xn + an2*xn^2 + an3*xn^3 + an4*xn^4) "
            "+ D1*I + ze + E2*ze^2 + E3*ze^3 + E4*ze^4"
        ),
        "beta_names": beta_names_c4(),
    },
    "truth": {
        "alpha_n_true": CFG.alpha_n_true,
        "alpha_p_true": CFG.alpha_p_true,
        "g_n_true": CFG.g_n_true,
        "g_p_true": CFG.g_p_true,
        "b_en_true": CFG.b_en_true,
        "b_ep_true": CFG.b_ep_true,
        "theta_n0_true": CFG.theta_n0_true,
        "theta_p0_true": CFG.theta_p0_true,
        "k_ind_true": [float(v) for v in truth["k_ind_true"]],
        "k_edges_true": [float(v) for v in truth["k_edges_true"]],
        "beta_true": {name: float(val) for name, val in zip(beta_names_c4(), truth["beta_true"])},
        "parameter_truth_values": truth_param_values,
    },
    "truth_phi_rank": truth_phi_rank,
}
with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config_payload, f, indent=2, default=str)


# =====================================================================
# Plots
# =====================================================================
if CFG.make_plots:
    plt.figure(figsize=(12.5, 6.2))
    plt.plot(t_best[:n], y_meas[:n], linewidth=2.6, label="synthetic truth")
    plt.plot(t_best[:n], y_hat[:n], "--", linewidth=2.4, label="CT-ID fit")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(
        f"{MODEL_ID}: synthetic truth vs estimated\n"
        f"seed={best_result['seed']}, RMSE={best_result['rmse']:.6e}"
    )
    plt.legend(loc="best")
    plt.tight_layout()
    save_or_show(FIG_DIR / "best_synthetic_vs_estimated.png")

    plt.figure(figsize=(12.5, 4.8))
    plt.plot(t_best[:n], residual[:n], linewidth=1.9)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Residual [V]")
    plt.title(f"{MODEL_ID}: residual")
    plt.tight_layout()
    save_or_show(FIG_DIR / "best_residual_vs_time.png")

    plt.figure(figsize=(12.5, 6.2))
    plt.scatter(df_all["seed"], df_all["rmse"], s=35)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Seed")
    plt.ylabel("RMSE [V]")
    plt.title(f"{MODEL_ID}: RMSE across seeds")
    plt.tight_layout()
    save_or_show(FIG_DIR / "rmse_scatter_by_seed.png")

    # Per-chunk count histograms for the main physical parameters.
    plot_params = [
        "alpha_n_hat", "alpha_p_hat", "g_n_hat", "g_p_hat", "b_en_hat", "b_ep_hat",
    ] + [c for c in df_all.columns if c.startswith("k") and c.endswith("_hat")]

    hist_dir = FIG_DIR / "parameter_histograms_chunk"
    hist_dir.mkdir(parents=True, exist_ok=True)
    for p in plot_params:
        if p not in df_all.columns:
            continue
        x = pd.to_numeric(df_all[p], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        plt.figure(figsize=(9.5, 5.6))
        plt.hist(x, bins=CFG.hist_bins, density=False, edgecolor="black", alpha=0.75)
        true_val = truth_param_values.get(p, np.nan)
        if np.isfinite(true_val):
            plt.axvline(float(true_val), linestyle="-", linewidth=2.4, label=f"true={true_val:.6g}")
        plt.axvline(float(np.mean(x)), linestyle="--", linewidth=2.1, label=f"mean={np.mean(x):.6g}")
        plt.axvline(float(np.median(x)), linestyle=":", linewidth=2.4, label=f"median={np.median(x):.6g}")
        plt.grid(True, axis="y", alpha=0.35)
        plt.xlabel(p)
        plt.ylabel("Count of runs")
        plt.title(f"{MODEL_ID}: recovered {p}, chunk {CFG.chunk_index}")
        plt.legend(loc="best")
        plt.tight_layout()
        save_or_show(hist_dir / f"hist_{p}_{CFG.hist_bins}bins.png")

print("\n" + "=" * 100)
print("FULLY SYNTHETIC S7/S17 C4 CHUNK COMPLETE")
print("=" * 100)
print("OUT_DIR:", OUT_DIR)
print("Best params:", best_params_path)
print("=" * 100)
