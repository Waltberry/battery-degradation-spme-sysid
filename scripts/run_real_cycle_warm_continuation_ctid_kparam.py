#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_real_cycle_warm_continuation_ctid_kparam.py

Purpose
-------
Warm-start continuation CT-ID for real discharge cycles using direct
electrolyte coupling parameterization.

This script replaces the old geometry-based electrolyte formulation:

    w = K_e / d^2

with directly identified electrolyte edge couplings.

It also replaces the old electrolyte input source:

    g_e / number_of_region_nodes

with directly identified electrolyte input gains:

    b_e,n and b_e,p.

Supported intended models
-------------------------
S7_C4K:
    S7 state model:
        negative solid: 2 states
        positive solid: 2 states
        electrolyte:    3 states

    Direct electrolyte A parameters:
        k1, k2

    Direct electrolyte B parameters:
        b_e,n, b_e,p

S17_C4K:
    S17 state model:
        negative solid: 4 states
        positive solid: 4 states
        electrolyte:    9 states

    Direct electrolyte A parameters:
        k1, k2, k3, k4, k5

    Expanded edge couplings:
        [k1, k1, k2, k3, k3, k4, k5, k5]

    Direct electrolyte B parameters:
        b_e,n, b_e,p

Output model
------------
Supervisor C4K output:

    Vhat = C
         + xp + ap2*xp^2 + ap3*xp^3 + ap4*xp^4
         - (xn + an2*xn^2 + an3*xn^3 + an4*xn^4)
         + D1*I
         + ze + E2*ze^2 + E3*ze^3 + E4*ze^4

The linear coefficients of xp, xn, and ze are fixed to 1.

Trainable beta vector:
    beta = [
        C,
        ap2, ap3, ap4,
        an2, an3, an4,
        D1,
        E2, E3, E4
    ]

Stoichiometry offsets
---------------------
theta_n0 and theta_p0 are frozen to nominal values.

This removes two weakly identifiable parameters and reduces parameter
compensation.

Outputs
-------
For each cycle/chunk:
    results/real_warm_continuation_ctid/<MODEL_ID>/<RUN_TAG>/

MODEL_ID is:
    S7_C4K or S17_C4K

Each chunk saves:
    all_runs.csv
    best_run.csv
    failed_runs.csv
    beta_coefficients.csv
    parameter_long.csv
    summary.csv
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

from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.io.data_io import load_mpr_as_dataframe
from battery_deg_spme.preprocessing.cycle_detection import (
    find_discharging_cycles_with_meta,
)


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    # Output
    output_root: str = "results/real_warm_continuation_ctid"
    figure_root: str = "results/figures/real_warm_continuation_ctid"

    # Real data
    mpr_path: str = os.environ.get("UN_MPR_PATH", "")
    cycle_index: int = int(os.environ.get("UN_REAL_CYCLE_INDEX", "0"))
    current_sign_mode: str = os.environ.get("UN_CURRENT_SIGN_MODE", "auto")
    smooth_voltage_window: int = int(os.environ.get("UN_SMOOTH_VOLTAGE_WINDOW", "1"))

    enable_id_downsample: bool = True
    id_downsample_dt: float = float(os.environ.get("UN_ID_DOWNSAMPLE_DT", "1.0"))
    id_downsample_use_interp: bool = False
    max_cycle_time: float = float(os.environ.get("UN_MAX_CYCLE_TIME", "0.0"))

    # Model
    state_id: str = os.environ.get("UN_STATE_VARIANT", "S7")
    candidate_id: str = os.environ.get("UN_OUTPUT_CANDIDATE", "C4")

    # Seeds for this chunk
    seed0: int = int(os.environ.get("UN_SEED0", "200"))
    n_multistart: int = int(os.environ.get("UN_N_MULTISTART", "10"))

    # Init mode
    init_mode: str = os.environ.get("UN_INIT_MODE", "cold").lower()
    warm_start_file: str = os.environ.get("UN_WARM_START_FILE", "")

    # Cold jitter
    init_dyn_jitter: float = float(os.environ.get("UN_INIT_DYN_JITTER", "0.30"))
    init_gain_jitter: float = float(os.environ.get("UN_INIT_GAIN_JITTER", "0.30"))
    init_C_jitter: float = float(os.environ.get("UN_INIT_C_JITTER", "0.05"))
    init_beta_scale: float = float(os.environ.get("UN_INIT_BETA_SCALE", "1e-2"))
    init_D1_center: float = float(os.environ.get("UN_INIT_D1_CENTER", "-0.004"))
    init_D1_jitter: float = float(os.environ.get("UN_INIT_D1_JITTER", "0.002"))

    # Warm local jitter
    warm_raw_dyn_jitter: float = float(os.environ.get("UN_WARM_RAW_DYN_JITTER", "0.04"))
    warm_raw_gain_jitter: float = float(os.environ.get("UN_WARM_RAW_GAIN_JITTER", "0.04"))
    warm_beta_rel_jitter: float = float(os.environ.get("UN_WARM_BETA_REL_JITTER", "0.03"))
    warm_beta_abs_jitter: float = float(os.environ.get("UN_WARM_BETA_ABS_JITTER", "2e-4"))
    warm_x0_abs_jitter: float = float(os.environ.get("UN_WARM_X0_ABS_JITTER", "2e-5"))

    # Nominal initial guesses
    alpha_n_nominal: float = float(os.environ.get("UN_ALPHA_N_INIT", "0.0064"))
    alpha_p_nominal: float = float(os.environ.get("UN_ALPHA_P_INIT", "0.0048"))

    # Direct electrolyte A coupling initial guesses.
    # Units are direct 1/s-like coupling rates.
    k_direct_nominal: float = float(os.environ.get("UN_K_DIRECT_INIT", "0.05"))

    # Solid input gains
    g_n_nominal: float = float(os.environ.get("UN_G_N_INIT", "1.20e-4"))
    g_p_nominal: float = float(os.environ.get("UN_G_P_INIT", "1.00e-4"))

    # Direct electrolyte B gains.
    # These replace ge/m and avoid geometry-based B assumptions.
    b_en_nominal: float = float(os.environ.get("UN_B_EN_INIT", "0.03"))
    b_ep_nominal: float = float(os.environ.get("UN_B_EP_INIT", "0.03"))

    # Frozen stoichiometry offsets
    theta_n0_nominal: float = float(os.environ.get("UN_THETA_N0_INIT", "0.50"))
    theta_p0_nominal: float = float(os.environ.get("UN_THETA_P0_INIT", "0.60"))
    theta_min: float = float(os.environ.get("UN_THETA_MIN", "0.02"))
    theta_max: float = float(os.environ.get("UN_THETA_MAX", "0.98"))

    # Electrolyte baseline
    ce0: float = float(os.environ.get("UN_CE0", "1000.0"))
    ce_min: float = float(os.environ.get("UN_CE_MIN", "1.0"))

    # Signs
    solid_input_sign_n: float = +1.0
    solid_input_sign_p: float = +1.0
    electrolyte_input_left_sign: float = +1.0
    electrolyte_input_right_sign: float = -1.0

    # Bounds / floors
    dyn_floor: float = float(os.environ.get("UN_DYN_FLOOR", "1e-12"))
    gain_floor: float = float(os.environ.get("UN_GAIN_FLOOR", "1e-10"))

    # Optimization
    adam_epochs: int = int(os.environ.get("UN_ADAM_EPOCHS", "500"))
    adam_eta: float = float(os.environ.get("UN_ADAM_ETA", "2e-3"))
    lbfgs_epochs: int = int(os.environ.get("UN_LBFGS_EPOCHS", "5000"))
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
    raise ValueError("This k-parameterized script supports only S7 and S17.")

if CFG.candidate_id != "C4":
    raise ValueError("This script is for the supervisor C4K output only. Use UN_OUTPUT_CANDIDATE=C4.")

STATE_SPEC = STATE_VARIANTS[CFG.state_id]
NX = STATE_SPEC["n_solid_n"] + STATE_SPEC["n_solid_p"] + STATE_SPEC["n_electrolyte"]
DEGREE = 4

MODEL_ID = f"{CFG.state_id}_{CFG.candidate_id}K"

RUN_TAG = os.environ.get(
    "UN_RUN_TAG",
    (
        f"warmseq_{MODEL_ID}_cycle_{CFG.cycle_index}_"
        f"{CFG.init_mode}_{CFG.n_multistart}seeds_"
        f"{CFG.seed0}_to_{CFG.seed0 + CFG.n_multistart - 1}_"
        f"dt_{CFG.id_downsample_dt}"
    ),
)

OUT_DIR = Path(CFG.output_root) / MODEL_ID / RUN_TAG
FIG_DIR = Path(CFG.figure_root) / MODEL_ID / RUN_TAG

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("REAL WARM-CONTINUATION CT-ID, DIRECT K/B PARAMETERIZATION")
print("=" * 100)
print("Working directory:", Path.cwd())
print("Threads:", N_THREADS)
print("JAX x64:", jax.config.read("jax_enable_x64"))
print("MODEL_ID:", MODEL_ID)
print("RUN_TAG:", RUN_TAG)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)
print("=" * 100)

with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(CFG), f, indent=2, default=str)


# ============================================================
# Basic helpers
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


def save_or_show(path: Path | None = None) -> None:
    if CFG.save_plots and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=230, bbox_inches="tight")
        print("[saved figure]", path)

    if CFG.show_plots:
        plt.show()

    plt.close()


# ============================================================
# Load selected real discharge cycle
# ============================================================
settings = get_default_settings()

if CFG.mpr_path.strip():
    settings.data.mpr_path = CFG.mpr_path.strip()

print("\nLoading real data from:")
print(" ", settings.data.mpr_path)

df_raw = load_mpr_as_dataframe(
    mpr_path=settings.data.mpr_path,
    time_col=settings.data.time_col,
    i_col=settings.data.i_col,
    v_col=settings.data.v_col,
)

min_len_for_search = (
    int(settings.cycle.min_cycle_len)
    if getattr(settings.cycle, "use_min_cycle_len", False)
    else None
)

cycles, cycle_meta = find_discharging_cycles_with_meta(
    df=df_raw,
    i_col=settings.data.i_col,
    tol_i=1e-9,
    min_len=min_len_for_search,
    include_previous_segment=settings.cycle.include_previous_segment,
    n_prev_points=settings.cycle.n_prev_points,
)

print("Detected discharge cycles:", len(cycles))

if CFG.cycle_index >= len(cycles):
    raise RuntimeError(f"Requested cycle {CFG.cycle_index}, but only {len(cycles)} cycles found.")

pd.DataFrame(cycle_meta).to_csv(OUT_DIR / "detected_cycle_metadata.csv", index=False)

cyc = cycles[CFG.cycle_index].copy()
cyc = cyc.dropna(subset=[settings.data.i_col, settings.data.v_col]).copy()

t_raw = cyc.index.to_numpy(dtype=np.float64)
t_rel = t_raw - t_raw[0]

i_raw = cyc[settings.data.i_col].to_numpy(dtype=np.float64)
v_raw = cyc[settings.data.v_col].to_numpy(dtype=np.float64)

if CFG.max_cycle_time > 0:
    keep = t_rel <= CFG.max_cycle_time
    t_rel = t_rel[keep]
    i_raw = i_raw[keep]
    v_raw = v_raw[keep]

if CFG.smooth_voltage_window > 1:
    w = int(CFG.smooth_voltage_window)
    v_raw = pd.Series(v_raw).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)

if CFG.current_sign_mode == "auto":
    if float(np.nanmean(i_raw)) < 0.0:
        i_fit = -i_raw
        current_flip_applied = True
    else:
        i_fit = i_raw.copy()
        current_flip_applied = False
elif CFG.current_sign_mode == "flip":
    i_fit = -i_raw
    current_flip_applied = True
elif CFG.current_sign_mode == "keep":
    i_fit = i_raw.copy()
    current_flip_applied = False
else:
    raise ValueError("UN_CURRENT_SIGN_MODE must be auto, flip, or keep.")

if CFG.enable_id_downsample and CFG.id_downsample_dt > 0 and len(t_rel) > 2:
    native_dt = float(np.median(np.diff(t_rel)))

    if CFG.id_downsample_dt > native_dt + 1e-12:
        if CFG.id_downsample_use_interp:
            t_new = np.arange(t_rel[0], t_rel[-1] + 0.5 * CFG.id_downsample_dt, CFG.id_downsample_dt)
            i_new = np.interp(t_new, t_rel, i_fit)
            v_new = np.interp(t_new, t_rel, v_raw)
            t_id = t_new
            i_id = i_new
            v_id = v_new
        else:
            step = max(int(round(CFG.id_downsample_dt / native_dt)), 1)
            idx_ds = np.arange(0, len(t_rel), step, dtype=int)
            t_id = t_rel[idx_ds]
            i_id = i_fit[idx_ds]
            v_id = v_raw[idx_ds]
    else:
        t_id = t_rel.copy()
        i_id = i_fit.copy()
        v_id = v_raw.copy()
else:
    t_id = t_rel.copy()
    i_id = i_fit.copy()
    v_id = v_raw.copy()

T_id = t_id - t_id[0]
U_id = ensure_2d_col(i_id, dtype=np.float64)
Y_id = ensure_2d_col(v_id, dtype=np.float64)

if len(T_id) < 2:
    raise RuntimeError("Selected cycle has fewer than two samples.")

Ts = float(np.median(np.diff(T_id)))

print("\nSelected cycle:")
print("  cycle_index:", CFG.cycle_index)
print("  ID points:", len(T_id))
print("  ID Ts:", Ts)
print("  current_flip_applied:", current_flip_applied)
print("  V start/end:", float(v_id[0]), float(v_id[-1]))

pd.DataFrame(
    {
        "t_s": T_id,
        "current_discharge_positive_A": i_id,
        "voltage_V": v_id,
    }
).to_csv(OUT_DIR / "selected_real_cycle_id_data.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(T_id, i_id)
plt.grid(True, alpha=0.35)
plt.ylabel("Current [A]")
plt.title(f"{MODEL_ID}: selected real discharge cycle {CFG.cycle_index}")

plt.subplot(2, 1, 2)
plt.plot(T_id, v_id)
plt.grid(True, alpha=0.35)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.tight_layout()
save_or_show(FIG_DIR / "selected_real_cycle_current_voltage.png")


# ============================================================
# State-space builders
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


def n_k_independent(n_e: int) -> int:
    if n_e == 3:
        return 2
    if n_e == 9:
        return 5
    raise ValueError("This script supports only n_e=3 for S7 and n_e=9 for S17.")


def k_names_for_ne(n_e: int) -> list[str]:
    return [f"k{i}" for i in range(1, n_k_independent(n_e) + 1)]


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
                k_ind[0],
                k_ind[0],
                k_ind[1],
                k_ind[2],
                k_ind[2],
                k_ind[3],
                k_ind[4],
                k_ind[4],
            ],
            dtype=jnp.float64,
        )

    raise ValueError("Unsupported electrolyte size.")


def build_solid_A_general(n: int, alpha: float) -> np.ndarray:
    if n == 2:
        return np.array(
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
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
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
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


def build_electrolyte_B_direct_np(
    n_e: int,
    b_en: float,
    b_ep: float,
    cfg: Config,
) -> np.ndarray:
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


def build_electrolyte_B_direct_jax(
    n_e: int,
    b_en,
    b_ep,
    cfg: Config,
):
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


# ============================================================
# Parameter pack / unpack / initialization
# ============================================================
def beta_names_c4k() -> list[str]:
    return [
        "C",
        "xp^2",
        "xp^3",
        "xp^4",
        "-xn^2",
        "-xn^3",
        "-xn^4",
        "I",
        "ze^2",
        "ze^3",
        "ze^4",
    ]


def make_initial_beta_c4k(seed: int, y0: float, cfg: Config) -> np.ndarray:
    rng = np.random.default_rng(seed)

    names = beta_names_c4k()
    beta = np.zeros(len(names), dtype=np.float64)

    beta[0] = float(y0) + cfg.init_C_jitter * rng.normal()

    # xp^2, xp^3, xp^4
    beta[1:4] = cfg.init_beta_scale * rng.normal(size=3)

    # -xn^2, -xn^3, -xn^4
    beta[4:7] = cfg.init_beta_scale * rng.normal(size=3)

    # I branch
    beta[7] = cfg.init_D1_center + cfg.init_D1_jitter * rng.normal()

    # ze^2, ze^3, ze^4
    beta[8:11] = cfg.init_beta_scale * rng.normal(size=3)

    return beta


def make_cold_initial_params(seed: int, y0: float, cfg: Config) -> tuple[list[np.ndarray], np.ndarray]:
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

    beta0 = make_initial_beta_c4k(seed, y0, cfg)

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


def load_warm_base_params(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Warm-start file does not exist: {path}")

    data = np.load(path, allow_pickle=True)

    params = []
    for k in range(7):
        params.append(np.asarray(data[f"param_{k}"], dtype=np.float64))

    if "x0" in data:
        x0 = np.asarray(data["x0"], dtype=np.float64).reshape(-1)
    else:
        x0 = np.zeros(NX, dtype=np.float64)

    if len(x0) != NX:
        print(f"[warn] warm x0 length {len(x0)} != NX {NX}; using zeros")
        x0 = np.zeros(NX, dtype=np.float64)

    return params, x0


def make_warm_initial_params(seed: int, cfg: Config) -> tuple[list[np.ndarray], np.ndarray]:
    warm_path = Path(cfg.warm_start_file)
    base_params, base_x0 = load_warm_base_params(warm_path)

    rng = np.random.default_rng(seed)
    params = [np.array(p, dtype=np.float64, copy=True) for p in base_params]

    # alpha_n, alpha_p
    for k in [0, 1]:
        params[k] = params[k] + cfg.warm_raw_dyn_jitter * rng.normal(size=params[k].shape)

    # k vector
    params[2] = params[2] + cfg.warm_raw_dyn_jitter * rng.normal(size=params[2].shape)

    # g_n, g_p
    for k in [3, 4]:
        params[k] = params[k] + cfg.warm_raw_gain_jitter * rng.normal(size=params[k].shape)

    # b_en, b_ep
    params[5] = params[5] + cfg.warm_raw_gain_jitter * rng.normal(size=params[5].shape)

    # beta
    beta = params[6].copy()
    beta = beta * (1.0 + cfg.warm_beta_rel_jitter * rng.normal(size=beta.shape))
    beta = beta + cfg.warm_beta_abs_jitter * rng.normal(size=beta.shape)
    params[6] = beta

    x0 = base_x0.copy()
    x0 = x0 + cfg.warm_x0_abs_jitter * rng.normal(size=x0.shape)

    return params, x0


def make_initial_params(seed: int, y0: float, cfg: Config) -> tuple[list[np.ndarray], np.ndarray]:
    if cfg.init_mode == "cold":
        return make_cold_initial_params(seed, y0, cfg)

    if cfg.init_mode == "warm":
        return make_warm_initial_params(seed, cfg)

    raise ValueError("UN_INIT_MODE must be cold or warm.")


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
        "theta_n0": cfg.theta_n0_nominal,
        "theta_p0": cfg.theta_p0_nominal,
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
        "theta_n0": cfg.theta_n0_nominal,
        "theta_p0": cfg.theta_p0_nominal,
        "beta": beta,
    }


# ============================================================
# Feature matrix and output
# ============================================================
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

    names = beta_names_c4k()

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


# ============================================================
# JAX CT model factories
# ============================================================
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


# ============================================================
# Fit one seed
# ============================================================
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

    model.loss(
        rho_x0=CFG.rho_x0,
        rho_th=CFG.rho_th,
        tau_th=CFG.tau_th,
    )

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

    comps = split_components_np(
        X=Xhat,
        I=i_id,
        phys=phys,
        state_spec=STATE_SPEC,
        cfg=CFG,
    )

    residual = ytrue - yhat

    rank_phi_raw = matrix_rank_condition_svd_np(comps["Phi"], CFG.svd_tol)
    rank_X_raw = matrix_rank_condition_svd_np(Xhat, CFG.svd_tol)
    lb = ljung_box_test(residual, CFG.ljung_box_lag)

    result = {
        "state_id": CFG.state_id,
        "candidate_id": CFG.candidate_id,
        "model_id": MODEL_ID,
        "degree": DEGREE,
        "nx": NX,
        "seed": seed,
        "init_mode": CFG.init_mode,
        "cycle_index": CFG.cycle_index,
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


# ============================================================
# Run multistart chunk
# ============================================================
all_results = []
fail_rows = []

print("\n" + "=" * 100)
print("STARTING DIRECT-K/B WARM-CONTINUATION CHUNK")
print("=" * 100)
print("MODEL_ID:", MODEL_ID)
print("Cycle:", CFG.cycle_index)
print("Init mode:", CFG.init_mode)
print("Warm start file:", CFG.warm_start_file)
print("Seeds:", CFG.seed0, "to", CFG.seed0 + CFG.n_multistart - 1)
print("=" * 100)

for k in range(CFG.n_multistart):
    seed = CFG.seed0 + k

    print("\n" + "-" * 100)
    print(f"{MODEL_ID} | cycle {CFG.cycle_index} | seed {seed} | {k + 1}/{CFG.n_multistart}")
    print("-" * 100)

    try:
        res = fit_one_seed(seed)
        all_results.append(res)

        print(
            f"{MODEL_ID} cycle={CFG.cycle_index} seed={seed} | "
            f"RMSE={res['rmse']:.6e} | "
            f"MAE={res['mae']:.6e} | "
            f"R2={res['r2_percent']:.6f}% | "
            f"BFR={res['bfr_percent']:.6f}% | "
            f"rankPhi={res['rank_phi_raw']['rank']}/{res['rank_phi_raw']['n_cols']} | "
            f"rankX={res['rank_X_raw']['rank']}/{res['rank_X_raw']['n_cols']}"
        )

    except Exception as exc:
        print(f"[FAIL] {MODEL_ID} cycle={CFG.cycle_index} seed={seed}: {repr(exc)}")
        fail_rows.append(
            {
                "model_id": MODEL_ID,
                "cycle_index": CFG.cycle_index,
                "seed": seed,
                "error": repr(exc),
            }
        )

pd.DataFrame(fail_rows).to_csv(OUT_DIR / "failed_runs.csv", index=False)

if len(all_results) == 0:
    raise RuntimeError("All fits failed in this chunk.")


# ============================================================
# Build tables
# ============================================================
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
        "candidate_id": CFG.candidate_id,
        "degree": DEGREE,
        "nx": NX,
        "cycle_index": CFG.cycle_index,
        "seed": r["seed"],
        "init_mode": CFG.init_mode,
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
        safe = (
            name.replace("^", "pow")
            .replace("-", "minus")
            .replace(" ", "_")
        )
        col = f"beta_{safe}"
        row[col] = beta[j]

        beta_rows.append(
            {
                "model_id": MODEL_ID,
                "cycle_index": CFG.cycle_index,
                "seed": r["seed"],
                "beta_name": name,
                "beta_hat": beta[j],
            }
        )

        param_rows.append(
            {
                "model_id": MODEL_ID,
                "cycle_index": CFG.cycle_index,
                "seed": r["seed"],
                "parameter": col,
                "estimate": beta[j],
            }
        )

    base_params_for_long = [
        "alpha_n_hat",
        "alpha_p_hat",
        "g_n_hat",
        "g_p_hat",
        "b_en_hat",
        "b_ep_hat",
    ]

    for p in base_params_for_long:
        param_rows.append(
            {
                "model_id": MODEL_ID,
                "cycle_index": CFG.cycle_index,
                "seed": r["seed"],
                "parameter": p,
                "estimate": row[p],
            }
        )

    for idx_k, val in enumerate(phys["k_ind"], start=1):
        param_rows.append(
            {
                "model_id": MODEL_ID,
                "cycle_index": CFG.cycle_index,
                "seed": r["seed"],
                "parameter": f"k{idx_k}_hat",
                "estimate": float(val),
            }
        )

    rows.append(row)

df_all = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
df_beta = pd.DataFrame(beta_rows)
df_params = pd.DataFrame(param_rows)

best_result = sorted(all_results, key=lambda r: float(r["rmse"]))[0]
df_best = df_all.head(1).copy()

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
    "cycle_index": CFG.cycle_index,
    "init_mode": CFG.init_mode,
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


# ============================================================
# Save best raw parameter vector for next cycle
# ============================================================
best_params_path = OUT_DIR / "best_params_raw.npz"

save_payload = {
    "x0": best_result["x0"],
    "cycle_index": np.array([CFG.cycle_index], dtype=np.int64),
    "seed": np.array([best_result["seed"]], dtype=np.int64),
    "rmse": np.array([best_result["rmse"]], dtype=np.float64),
    "model_id": np.array([MODEL_ID]),
}

for k, p in enumerate(best_result["raw_params"]):
    save_payload[f"param_{k}"] = np.asarray(p, dtype=np.float64)

np.savez(best_params_path, **save_payload)

print("[saved best raw params]", best_params_path)


# ============================================================
# Save best measured-vs-estimated response and matrices
# ============================================================
t = ensure_1d(T_id)
i = ensure_1d(i_id)
y_meas = ensure_1d(Y_id)
y_hat = ensure_1d(best_result["yhat"])
residual = ensure_1d(best_result["residual"])
Xhat = np.asarray(best_result["Xhat"], dtype=np.float64)
comps = best_result["components"]

n = min(len(t), len(i), len(y_meas), len(y_hat), len(residual))

df_response = pd.DataFrame(
    {
        "t_s": t[:n],
        "current_A_discharge_positive": i[:n],
        "measured_voltage_V": y_meas[:n],
        "estimated_voltage_V": y_hat[:n],
        "residual_V": residual[:n],
        "abs_residual_V": np.abs(residual[:n]),
    }
)

response_csv = OUT_DIR / "best_measured_estimated_response.csv"
df_response.to_csv(response_csv, index=False)

state_data = {"t_s": t[:n]}
for j in range(Xhat.shape[1]):
    state_data[f"xhat_{j}"] = Xhat[:n, j]

state_csv = OUT_DIR / "best_state_trajectory.csv"
pd.DataFrame(state_data).to_csv(state_csv, index=False)

Phi = np.asarray(comps["Phi"], dtype=np.float64)
phi_data = {"t_s": t[:n]}
for j in range(Phi.shape[1]):
    phi_data[f"phi_{j}"] = Phi[:n, j]

phi_csv = OUT_DIR / "best_feature_matrix_phi.csv"
pd.DataFrame(phi_data).to_csv(phi_csv, index=False)

component_data = {
    "t_s": t[:n],
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
    "cycle_index": CFG.cycle_index,
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

if CFG.make_plots:
    plt.figure(figsize=(12.5, 6.2))
    plt.plot(t[:n], y_meas[:n], linewidth=2.6, label="measured")
    plt.plot(t[:n], y_hat[:n], "--", linewidth=2.4, label="estimated")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(
        f"{MODEL_ID} cycle {CFG.cycle_index}: measured vs estimated\n"
        f"seed={best_result['seed']}, RMSE={best_result['rmse']:.6e}"
    )
    plt.legend(loc="best")
    plt.tight_layout()
    save_or_show(FIG_DIR / "best_measured_vs_estimated.png")

    plt.figure(figsize=(12.5, 4.8))
    plt.plot(t[:n], residual[:n], linewidth=1.9)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Residual [V]")
    plt.title(f"{MODEL_ID} cycle {CFG.cycle_index}: residual")
    plt.tight_layout()
    save_or_show(FIG_DIR / "best_residual_vs_time.png")

    plt.figure(figsize=(12.5, 6.2))
    plt.scatter(df_all["seed"], df_all["rmse"], s=35)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Seed")
    plt.ylabel("RMSE [V]")
    plt.title(f"{MODEL_ID} cycle {CFG.cycle_index}: RMSE across seeds")
    plt.tight_layout()
    save_or_show(FIG_DIR / "rmse_scatter_by_seed.png")


print("\n" + "=" * 100)
print("DIRECT-K/B WARM-CONTINUATION CHUNK COMPLETE")
print("=" * 100)
print("OUT_DIR:", OUT_DIR)
print("Best params:", best_params_path)
print("=" * 100)