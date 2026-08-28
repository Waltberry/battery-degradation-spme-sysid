#!/usr/bin/env python3
# %% =====================================================
# CELL 0 — Imports and environment
# =====================================================
"""
run_real_cycle_state_order_ctid_grid.py

Purpose
-------
Fit the first real-data discharge cycle using several continuous-time
state-space variants and polynomial full-voltage output orders.

State variants
--------------
S7:
    negative solid = 2 states
    positive solid = 2 states
    electrolyte    = 3 states
    total          = 7 states

S12:
    negative solid = 3 states
    positive solid = 3 states
    electrolyte    = 6 states
    total          = 12 states

S14:
    negative solid = 4 states
    positive solid = 4 states
    electrolyte    = 6 states
    total          = 14 states

S17:
    negative solid = 4 states
    positive solid = 4 states
    electrolyte    = 9 states
    total          = 17 states

Output candidates
-----------------
C1:
    Vhat = C + ap1*xp - an1*xn + D1*I + E1*ze

C2:
    Vhat = C
         + ap1*xp + ap2*xp^2
         - (an1*xn + an2*xn^2)
         + D1*I
         + E1*ze + E2*ze^2

C3:
    Vhat = C
         + ap1*xp + ap2*xp^2 + ap3*xp^3
         - (an1*xn + an2*xn^2 + an3*xn^3)
         + D1*I
         + E1*ze + E2*ze^2 + E3*ze^3

C4:
    Vhat = C
         + xp + ap2*xp^2 + ap3*xp^3 + ap4*xp^4
         - (xn + an2*xn^2 + an3*xn^3 + an4*xn^4)
         + D1*I
         + ze + E2*ze^2 + E3*ze^3 + E4*ze^4

Important
---------
This is a real-data fitting script.

There are no known true physical parameter values for the real cycle, so
histograms show:
    - count of runs on the y-axis
    - mean line
    - median line

No true-value line is drawn unless you manually provide one.

No least-squares output coefficient fit is used.
The polynomial coefficients are optimized directly by jax-sysid.
"""

from __future__ import annotations

import os
import gc
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

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

import numpy as np
import pandas as pd
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
    summarize_cycles,
)

print("=" * 100)
print("REAL FIRST-DISCHARGE-CYCLE CT-ID GRID")
print("=" * 100)
print("Working directory:", Path.cwd())
print("Python threads:", N_THREADS)
print("JAX x64 enabled:", jax.config.read("jax_enable_x64"))
print("=" * 100)


# %% =====================================================
# CELL 1 — Configuration
# =====================================================
@dataclass
class Config:
    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------
    output_root: str = "results/real_cycle_ctid_state_order_grid"
    figure_root: str = "results/figures/real_cycle_ctid_state_order_grid"

    # -----------------------------------------------------
    # Real data controls
    # -----------------------------------------------------
    mpr_path: str = os.environ.get("UN_MPR_PATH", "")
    cycle_index: int = int(os.environ.get("UN_REAL_CYCLE_INDEX", "0"))

    # Discharge-current convention.
    # auto:
    #   if mean current in selected cycle is negative, flip sign so discharge is positive.
    current_sign_mode: str = os.environ.get("UN_CURRENT_SIGN_MODE", "auto")

    # Optional voltage/current preprocessing.
    remove_voltage_nan: bool = True
    smooth_voltage_window: int = int(os.environ.get("UN_SMOOTH_VOLTAGE_WINDOW", "1"))

    # Downsampling for ID.
    enable_id_downsample: bool = True
    id_downsample_dt: float = float(os.environ.get("UN_ID_DOWNSAMPLE_DT", "1.0"))
    id_downsample_use_interp: bool = False

    # Optional crop of first cycle after relative time.
    # Set to <=0 to disable.
    max_cycle_time: float = float(os.environ.get("UN_MAX_CYCLE_TIME", "0.0"))

    # -----------------------------------------------------
    # Candidate grid controls
    # -----------------------------------------------------
    state_variants: str = os.environ.get("UN_STATE_VARIANTS", "S7,S12,S14,S17")
    output_candidates: str = os.environ.get("UN_OUTPUT_CANDIDATES", "C1,C2,C3,C4")

    seed0: int = int(os.environ.get("UN_SEED0", "200"))
    n_multistart: int = int(os.environ.get("UN_N_MULTISTART", "10"))

    # -----------------------------------------------------
    # Physical constants
    # -----------------------------------------------------
    R_gas: float = 8.314462618
    F: float = 96485.33212
    T: float = 298.15
    N_series: int = 1

    # -----------------------------------------------------
    # Geometry
    # -----------------------------------------------------
    L_n: float = float(os.environ.get("UN_L_N", "80e-6"))
    L_sep: float = float(os.environ.get("UN_L_SEP", "25e-6"))
    L_p: float = float(os.environ.get("UN_L_P", "75e-6"))

    # -----------------------------------------------------
    # Nominal parameter initialization
    # These are not truth values for real data.
    # They are only initial guesses.
    # -----------------------------------------------------
    alpha_n_nominal: float = float(os.environ.get("UN_ALPHA_N_INIT", "0.0064"))
    alpha_p_nominal: float = float(os.environ.get("UN_ALPHA_P_INIT", "0.0048"))
    K_e_nominal: float = float(os.environ.get("UN_K_E_INIT", "5e-11"))

    g_n_nominal: float = float(os.environ.get("UN_G_N_INIT", "1.20e-4"))
    g_p_nominal: float = float(os.environ.get("UN_G_P_INIT", "1.00e-4"))
    g_e_nominal: float = float(os.environ.get("UN_G_E_INIT", "0.08"))

    theta_n0_nominal: float = float(os.environ.get("UN_THETA_N0_INIT", "0.50"))
    theta_p0_nominal: float = float(os.environ.get("UN_THETA_P0_INIT", "0.60"))

    theta_min: float = float(os.environ.get("UN_THETA_MIN", "0.02"))
    theta_max: float = float(os.environ.get("UN_THETA_MAX", "0.98"))

    ce0: float = float(os.environ.get("UN_CE0", "1000.0"))
    ce_min: float = float(os.environ.get("UN_CE_MIN", "1.0"))

    # -----------------------------------------------------
    # Input signs
    # In this script, I is converted to discharge-positive.
    # -----------------------------------------------------
    solid_input_sign_n: float = +1.0
    solid_input_sign_p: float = +1.0
    electrolyte_input_left_sign: float = +1.0
    electrolyte_input_right_sign: float = -1.0

    # -----------------------------------------------------
    # Parameter floors and bounds
    # -----------------------------------------------------
    dyn_floor: float = 1e-12
    gain_floor: float = 1e-10
    estimate_theta_offsets: bool = True

    # -----------------------------------------------------
    # Optimization controls
    # -----------------------------------------------------
    adam_epochs: int = int(os.environ.get("UN_ADAM_EPOCHS", "500"))
    adam_eta: float = float(os.environ.get("UN_ADAM_ETA", "2e-3"))

    lbfgs_epochs: int = int(os.environ.get("UN_LBFGS_EPOCHS", "5000"))
    lbfgs_tol: float = float(os.environ.get("UN_LBFGS_TOL", "1e-12"))
    lbfgs_memory: int = int(os.environ.get("UN_LBFGS_MEMORY", "30"))
    iprint: int = int(os.environ.get("UN_IPRINT", "0"))

    rho_x0: float = float(os.environ.get("UN_RHO_X0", "1e-8"))
    rho_th: float = float(os.environ.get("UN_RHO_TH", "1e-8"))
    tau_th: float = float(os.environ.get("UN_TAU_TH", "0.0"))

    # -----------------------------------------------------
    # Initialization jitter
    # -----------------------------------------------------
    init_dyn_jitter: float = float(os.environ.get("UN_INIT_DYN_JITTER", "0.20"))
    init_gain_jitter: float = float(os.environ.get("UN_INIT_GAIN_JITTER", "0.20"))
    init_theta_jitter: float = float(os.environ.get("UN_INIT_THETA_JITTER", "0.005"))

    init_C_jitter: float = float(os.environ.get("UN_INIT_C_JITTER", "0.05"))
    init_beta_scale: float = float(os.environ.get("UN_INIT_BETA_SCALE", "1e-2"))
    init_D1_center: float = float(os.environ.get("UN_INIT_D1_CENTER", "-0.004"))
    init_D1_jitter: float = float(os.environ.get("UN_INIT_D1_JITTER", "0.002"))
    init_E_scale: float = float(os.environ.get("UN_INIT_E_SCALE", "1e-2"))

    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------
    svd_tol: float = float(os.environ.get("UN_SVD_TOL", "1e-10"))
    max_corr_lag: int = int(os.environ.get("UN_MAX_CORR_LAG", "40"))
    ljung_box_lag: int = int(os.environ.get("UN_LJUNG_BOX_LAG", "20"))

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    make_plots: bool = os.environ.get("UN_MAKE_PLOTS", "True").lower() == "true"
    show_plots: bool = os.environ.get("UN_SHOW_PLOTS", "False").lower() == "true"
    save_plots: bool = os.environ.get("UN_SAVE_PLOTS", "True").lower() == "true"
    hist_bins: int = int(os.environ.get("UN_HIST_BINS", "100"))


CFG = Config()

RUN_TAG = os.environ.get(
    "UN_RUN_TAG",
    (
        f"real_cycle{CFG.cycle_index}_"
        f"states_{CFG.state_variants.replace(',', '-')}_"
        f"orders_{CFG.output_candidates.replace(',', '-')}_"
        f"{CFG.n_multistart}seeds_{CFG.seed0}_to_{CFG.seed0 + CFG.n_multistart - 1}_"
        f"dt_{CFG.id_downsample_dt}_bins_{CFG.hist_bins}"
    ),
)

OUT_DIR = Path(CFG.output_root) / RUN_TAG
FIG_DIR = Path(CFG.figure_root) / RUN_TAG

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print("\nConfiguration:")
for k, v in asdict(CFG).items():
    print(f"  {k}: {v}")
print("RUN_TAG:", RUN_TAG)
print("OUT_DIR:", OUT_DIR)
print("FIG_DIR:", FIG_DIR)


# %% =====================================================
# CELL 2 — Candidate/state variant definitions
# =====================================================
STATE_VARIANTS: dict[str, dict[str, int]] = {
    "S7": {
        "n_solid_n": 2,
        "n_solid_p": 2,
        "n_electrolyte": 3,
    },
    "S12": {
        "n_solid_n": 3,
        "n_solid_p": 3,
        "n_electrolyte": 6,
    },
    "S14": {
        "n_solid_n": 4,
        "n_solid_p": 4,
        "n_electrolyte": 6,
    },
    "S17": {
        "n_solid_n": 4,
        "n_solid_p": 4,
        "n_electrolyte": 9,
    },
}

OUTPUT_CANDIDATES: dict[str, dict[str, Any]] = {
    "C1": {
        "degree": 1,
        "label": "C1 linear OCP + linear electrolyte",
    },
    "C2": {
        "degree": 2,
        "label": "C2 quadratic OCP + quadratic electrolyte",
    },
    "C3": {
        "degree": 3,
        "label": "C3 cubic OCP + cubic electrolyte",
    },
    "C4": {
        "degree": 4,
        "label": "C4 quartic OCP + quartic electrolyte",
    },
}

SELECTED_STATE_VARIANTS = [s.strip() for s in CFG.state_variants.split(",") if s.strip()]
SELECTED_OUTPUT_CANDIDATES = [s.strip() for s in CFG.output_candidates.split(",") if s.strip()]

for s in SELECTED_STATE_VARIANTS:
    if s not in STATE_VARIANTS:
        raise ValueError(f"Unknown state variant {s}. Use one of {list(STATE_VARIANTS)}.")

for c in SELECTED_OUTPUT_CANDIDATES:
    if c not in OUTPUT_CANDIDATES:
        raise ValueError(f"Unknown output candidate {c}. Use one of {list(OUTPUT_CANDIDATES)}.")

print("\nSelected state variants:", SELECTED_STATE_VARIANTS)
print("Selected output candidates:", SELECTED_OUTPUT_CANDIDATES)


def output_equation_for_degree(degree: int) -> str:
    xp_terms = " + ".join([f"ap{k}*xp^{k}" if k > 1 else "ap1*xp" for k in range(1, degree + 1)])
    xn_terms = " + ".join([f"an{k}*xn^{k}" if k > 1 else "an1*xn" for k in range(1, degree + 1)])
    ze_terms = " + ".join([f"E{k}*ze^{k}" if k > 1 else "E1*ze" for k in range(1, degree + 1)])
    return f"C + {xp_terms} - ({xn_terms}) + D1*I + {ze_terms}"


# %% =====================================================
# CELL 3 — General numerical helpers
# =====================================================
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


def sigmoid_inverse_np(y: float) -> float:
    y = float(np.clip(y, 1e-9, 1.0 - 1e-9))
    return float(np.log(y / (1.0 - y)))


def theta_to_raw(theta: float, cfg: Config) -> float:
    q = (float(theta) - cfg.theta_min) / (cfg.theta_max - cfg.theta_min)
    return sigmoid_inverse_np(q)


def raw_to_theta_np(raw: float, cfg: Config) -> float:
    sig = 1.0 / (1.0 + np.exp(-float(raw)))
    return float(cfg.theta_min + (cfg.theta_max - cfg.theta_min) * sig)


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

    svals = np.linalg.svd(M0, compute_uv=False)

    if len(svals) == 0:
        rank = 0
        cond = np.nan
    else:
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


def save_or_show(fig_path: Path | None = None) -> None:
    if CFG.save_plots and fig_path is not None:
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=230, bbox_inches="tight")
        print("[saved figure]", fig_path)

    if CFG.show_plots:
        plt.show()

    plt.close()


# %% =====================================================
# CELL 4 — Load real data and extract first discharge cycle
# =====================================================
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

print("Raw data shape:", df_raw.shape)
print("Raw data columns:", list(df_raw.columns))

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

if len(cycles) == 0:
    raise RuntimeError("No discharge cycles detected. Check current sign/tolerance/cycle settings.")

if CFG.cycle_index >= len(cycles):
    raise RuntimeError(f"Requested cycle_index={CFG.cycle_index}, but only {len(cycles)} cycles were detected.")

cycle_meta_df = pd.DataFrame(cycle_meta)
cycle_meta_df.to_csv(OUT_DIR / "detected_cycle_metadata.csv", index=False)

cycle_summary = summarize_cycles(cycle_meta) if len(cycle_meta) > 0 else {}
with open(OUT_DIR / "cycle_summary.json", "w", encoding="utf-8") as f:
    json.dump(cycle_summary, f, indent=2, default=str)

cyc = cycles[CFG.cycle_index].copy()

if CFG.remove_voltage_nan:
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

# Smooth voltage only if explicitly requested.
if CFG.smooth_voltage_window > 1:
    w = int(CFG.smooth_voltage_window)
    v_raw = pd.Series(v_raw).rolling(window=w, center=True, min_periods=1).mean().to_numpy(dtype=np.float64)

# Convert current to discharge-positive convention.
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

# Downsample.
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
            idx = np.arange(0, len(t_rel), step, dtype=int)
            t_id = t_rel[idx]
            i_id = i_fit[idx]
            v_id = v_raw[idx]
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

if len(T_id) >= 2:
    Ts = float(np.median(np.diff(T_id)))
else:
    raise RuntimeError("Selected cycle has fewer than two ID samples.")

print("\nSelected real discharge cycle:")
print("  cycle_index:", CFG.cycle_index)
print("  original points:", len(t_rel))
print("  ID points:", len(T_id))
print("  ID Ts:", Ts)
print("  current flip applied:", current_flip_applied)
print("  I raw mean:", float(np.nanmean(i_raw)))
print("  I fit mean:", float(np.nanmean(i_id)))
print("  V start/end:", float(v_id[0]), float(v_id[-1]))
print("  V min/max:", float(np.min(v_id)), float(np.max(v_id)))

pd.DataFrame(
    {
        "t_s": T_id,
        "current_discharge_positive_A": i_id,
        "voltage_V": v_id,
    }
).to_csv(OUT_DIR / "selected_real_cycle_id_data.csv", index=False)

# Raw selected-cycle plot.
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(T_id, i_id)
plt.grid(True, alpha=0.35)
plt.ylabel("Current [A]")
plt.title(f"Selected real discharge cycle {CFG.cycle_index}")

plt.subplot(2, 1, 2)
plt.plot(T_id, v_id)
plt.grid(True, alpha=0.35)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.tight_layout()
save_or_show(FIG_DIR / "selected_real_cycle_current_voltage.png")


# %% =====================================================
# CELL 5 — General state-space builders
# =====================================================
def build_solid_A_general(n: int, alpha: float) -> np.ndarray:
    """
    Build a compact finite-difference-like solid diffusion chain.

    Special cases:
        n=2 matches the reduced 2-state model used earlier.
        n=4 matches the uploaded 4-state builder form:
            [-24, 24]
            [16, -40, 24]
            [0, 16, -40, 24]
            [0, 0, 16, -16]
        times alpha.

    For n=3, the same pattern is extended using:
        lower = 4*n*alpha
        upper = 6*n*alpha

    This gives a consistent sequence:
        surface state is the last solid state.
    """
    if n < 2:
        raise ValueError("solid node count must be >= 2")

    if n == 2:
        A = np.array(
            [
                [-8.0 * alpha, 8.0 * alpha],
                [8.0 * alpha, -8.0 * alpha],
            ],
            dtype=np.float64,
        )
        return A

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


def build_solid_B_general(n: int, gain: float, sign: float) -> np.ndarray:
    B = np.zeros((n, 1), dtype=np.float64)
    B[-1, 0] = sign * float(gain)
    return B


def electrolyte_region_counts(n_e: int) -> tuple[int, int, int]:
    """
    Split electrolyte nodes into negative/separator/positive regions.

    Supported variants:
        3 -> 1, 1, 1
        6 -> 2, 2, 2
        9 -> 3, 3, 3
    """
    if n_e not in (3, 6, 9):
        raise ValueError("Supported electrolyte node counts are 3, 6, and 9.")

    m = n_e // 3
    return m, m, m


def build_electrolyte_A_general(n_e: int, K_e: float, cfg: Config) -> np.ndarray:
    """
    Build an electrolyte diffusion chain across:
        negative electrode | separator | positive electrode.

    The coupling between neighboring electrolyte nodes is based on the
    distance between finite-volume centers.

    Within a region with length L and m nodes:
        center-to-center spacing approximately L/m
        coupling = K_e / spacing^2 = K_e * m^2 / L^2

    Across an interface between region a and b:
        distance = L_a/(2m_a) + L_b/(2m_b)
        coupling = K_e / distance^2

    This exactly reproduces the common 3-node interface form:
        4*K_e/(L_n + L_sep)^2
        4*K_e/(L_sep + L_p)^2

    For 6 nodes with 2 nodes/region, it reproduces:
        4*K_e/L^2 within region
        16*K_e/(L_a + L_b)^2 across region boundaries.
    """
    m_n, m_sep, m_p = electrolyte_region_counts(n_e)

    region_lengths = [cfg.L_n, cfg.L_sep, cfg.L_p]
    region_counts = [m_n, m_sep, m_p]

    # Region id for each electrolyte node.
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


def build_electrolyte_B_general(n_e: int, g_e: float, cfg: Config) -> np.ndarray:
    """
    Electrolyte input source.

    The source is distributed over the negative electrolyte region and the
    positive electrolyte region. It is normalized by the number of nodes in each
    region so that changing electrolyte resolution does not automatically scale
    the total source magnitude.
    """
    m_n, m_sep, m_p = electrolyte_region_counts(n_e)

    B = np.zeros((n_e, 1), dtype=np.float64)

    left_nodes = list(range(0, m_n))
    right_nodes = list(range(m_n + m_sep, m_n + m_sep + m_p))

    for idx in left_nodes:
        B[idx, 0] += cfg.electrolyte_input_left_sign * float(g_e) / len(left_nodes)

    for idx in right_nodes:
        B[idx, 0] += cfg.electrolyte_input_right_sign * float(g_e) / len(right_nodes)

    return B


def assemble_A_B_np(
    state_spec: dict[str, int],
    alpha_n: float,
    alpha_p: float,
    K_e: float,
    g_n: float,
    g_p: float,
    g_e: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_n = int(state_spec["n_solid_n"])
    n_p = int(state_spec["n_solid_p"])
    n_e = int(state_spec["n_electrolyte"])

    A_n = build_solid_A_general(n_n, alpha_n)
    A_p = build_solid_A_general(n_p, alpha_p)
    A_e = build_electrolyte_A_general(n_e, K_e, cfg)

    B_n = build_solid_B_general(n_n, g_n, cfg.solid_input_sign_n)
    B_p = build_solid_B_general(n_p, g_p, cfg.solid_input_sign_p)
    B_e = build_electrolyte_B_general(n_e, g_e, cfg)

    A = block_diag(A_n, A_p, A_e)
    B = np.vstack([B_n, B_p, B_e])

    idx = make_state_indices(state_spec)

    return A, B, idx


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


# %% =====================================================
# CELL 6 — JAX state-space builders
# =====================================================
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


def build_electrolyte_A_general_jax(n_e: int, K_e, cfg: Config):
    m_n, m_sep, m_p = electrolyte_region_counts(n_e)

    region_lengths = [cfg.L_n, cfg.L_sep, cfg.L_p]
    region_counts = [m_n, m_sep, m_p]

    region_ids = []
    for rid, count in enumerate(region_counts):
        region_ids.extend([rid] * count)

    A = jnp.zeros((n_e, n_e), dtype=jnp.float64)

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

        w = K_e / (dist**2)

        A = A.at[j, j].add(-w)
        A = A.at[j, j + 1].add(w)
        A = A.at[j + 1, j].add(w)
        A = A.at[j + 1, j + 1].add(-w)

    return A


# %% =====================================================
# CELL 7 — Parameter pack/unpack and initialization
# =====================================================
def make_initial_beta(seed: int, degree: int, y0: float, cfg: Config) -> np.ndarray:
    """
    Direct JAX optimization initialization.

    No least-squares solve is used.

    Column order:
        C,
        xp^1,...,xp^degree,
        -xn^1,...,-xn^degree,
        I,
        ze^1,...,ze^degree
    """
    rng = np.random.default_rng(seed)

    n_beta = 1 + degree + degree + 1 + degree
    beta = np.zeros(n_beta, dtype=np.float64)

    # C starts near the first measured voltage.
    beta[0] = float(y0) + cfg.init_C_jitter * rng.normal()

    # xp terms
    j = 1
    for _ in range(degree):
        beta[j] = cfg.init_beta_scale * rng.normal()
        j += 1

    # -xn terms
    for _ in range(degree):
        beta[j] = cfg.init_beta_scale * rng.normal()
        j += 1

    # D1 current term.
    beta[j] = cfg.init_D1_center + cfg.init_D1_jitter * rng.normal()
    j += 1

    # electrolyte polynomial terms.
    for _ in range(degree):
        beta[j] = cfg.init_E_scale * rng.normal()
        j += 1

    return beta


def make_initial_params(seed: int, degree: int, y0: float, cfg: Config) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)

    alpha_n0 = cfg.alpha_n_nominal * np.exp(cfg.init_dyn_jitter * rng.normal())
    alpha_p0 = cfg.alpha_p_nominal * np.exp(cfg.init_dyn_jitter * rng.normal())
    K_e0 = cfg.K_e_nominal * np.exp(cfg.init_dyn_jitter * rng.normal())

    g_n0 = cfg.g_n_nominal * np.exp(cfg.init_gain_jitter * rng.normal())
    g_p0 = cfg.g_p_nominal * np.exp(cfg.init_gain_jitter * rng.normal())
    g_e0 = cfg.g_e_nominal * np.exp(cfg.init_gain_jitter * rng.normal())

    theta_n0_guess = cfg.theta_n0_nominal + cfg.init_theta_jitter * rng.normal()
    theta_p0_guess = cfg.theta_p0_nominal + cfg.init_theta_jitter * rng.normal()

    theta_n0_guess = float(np.clip(theta_n0_guess, cfg.theta_min + 1e-6, cfg.theta_max - 1e-6))
    theta_p0_guess = float(np.clip(theta_p0_guess, cfg.theta_min + 1e-6, cfg.theta_max - 1e-6))

    beta0 = make_initial_beta(seed, degree, y0, cfg)

    return [
        np.array([softplus_inverse_np(alpha_n0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(alpha_p0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(K_e0, cfg.dyn_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(g_n0, cfg.gain_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(g_p0, cfg.gain_floor)], dtype=np.float64),
        np.array([softplus_inverse_np(g_e0, cfg.gain_floor)], dtype=np.float64),
        np.array([theta_to_raw(theta_n0_guess, cfg)], dtype=np.float64),
        np.array([theta_to_raw(theta_p0_guess, cfg)], dtype=np.float64),
        np.asarray(beta0, dtype=np.float64),
    ]


def unpack_params_np(params: list[np.ndarray], state_spec: dict[str, int], cfg: Config) -> dict[str, Any]:
    raw_alpha_n = float(np.asarray(params[0]).reshape(-1)[0])
    raw_alpha_p = float(np.asarray(params[1]).reshape(-1)[0])
    raw_K_e = float(np.asarray(params[2]).reshape(-1)[0])

    raw_g_n = float(np.asarray(params[3]).reshape(-1)[0])
    raw_g_p = float(np.asarray(params[4]).reshape(-1)[0])
    raw_g_e = float(np.asarray(params[5]).reshape(-1)[0])

    raw_theta_n0 = float(np.asarray(params[6]).reshape(-1)[0])
    raw_theta_p0 = float(np.asarray(params[7]).reshape(-1)[0])

    beta = np.asarray(params[8], dtype=np.float64).reshape(-1)

    alpha_n = cfg.dyn_floor + float(np.log1p(np.exp(raw_alpha_n)))
    alpha_p = cfg.dyn_floor + float(np.log1p(np.exp(raw_alpha_p)))
    K_e = cfg.dyn_floor + float(np.log1p(np.exp(raw_K_e)))

    g_n = cfg.gain_floor + float(np.log1p(np.exp(raw_g_n)))
    g_p = cfg.gain_floor + float(np.log1p(np.exp(raw_g_p)))
    g_e = cfg.gain_floor + float(np.log1p(np.exp(raw_g_e)))

    if cfg.estimate_theta_offsets:
        theta_n0 = raw_to_theta_np(raw_theta_n0, cfg)
        theta_p0 = raw_to_theta_np(raw_theta_p0, cfg)
    else:
        theta_n0 = cfg.theta_n0_nominal
        theta_p0 = cfg.theta_p0_nominal

    A, B, idx = assemble_A_B_np(
        state_spec=state_spec,
        alpha_n=alpha_n,
        alpha_p=alpha_p,
        K_e=K_e,
        g_n=g_n,
        g_p=g_p,
        g_e=g_e,
        cfg=cfg,
    )

    return {
        "alpha_n": alpha_n,
        "alpha_p": alpha_p,
        "K_e": K_e,
        "g_n": g_n,
        "g_p": g_p,
        "g_e": g_e,
        "theta_n0": theta_n0,
        "theta_p0": theta_p0,
        "beta": beta,
        "A": A,
        "B": B,
        "idx": idx,
        "ct_poles": np.linalg.eigvals(A),
    }


def unpack_params_jax(params, cfg: Config):
    raw_alpha_n = params[0][0]
    raw_alpha_p = params[1][0]
    raw_K_e = params[2][0]

    raw_g_n = params[3][0]
    raw_g_p = params[4][0]
    raw_g_e = params[5][0]

    raw_theta_n0 = params[6][0]
    raw_theta_p0 = params[7][0]

    beta = params[8]

    alpha_n = cfg.dyn_floor + jax.nn.softplus(raw_alpha_n)
    alpha_p = cfg.dyn_floor + jax.nn.softplus(raw_alpha_p)
    K_e = cfg.dyn_floor + jax.nn.softplus(raw_K_e)

    g_n = cfg.gain_floor + jax.nn.softplus(raw_g_n)
    g_p = cfg.gain_floor + jax.nn.softplus(raw_g_p)
    g_e = cfg.gain_floor + jax.nn.softplus(raw_g_e)

    if cfg.estimate_theta_offsets:
        theta_n0 = cfg.theta_min + (cfg.theta_max - cfg.theta_min) * jax.nn.sigmoid(raw_theta_n0)
        theta_p0 = cfg.theta_min + (cfg.theta_max - cfg.theta_min) * jax.nn.sigmoid(raw_theta_p0)
    else:
        theta_n0 = jnp.array(cfg.theta_n0_nominal, dtype=jnp.float64)
        theta_p0 = jnp.array(cfg.theta_p0_nominal, dtype=jnp.float64)

    return {
        "alpha_n": alpha_n,
        "alpha_p": alpha_p,
        "K_e": K_e,
        "g_n": g_n,
        "g_p": g_p,
        "g_e": g_e,
        "theta_n0": theta_n0,
        "theta_p0": theta_p0,
        "beta": beta,
    }


# %% =====================================================
# CELL 8 — Feature matrix and output helpers
# =====================================================
def candidate_feature_matrix_np(
    X: np.ndarray,
    I: np.ndarray,
    state_spec: dict[str, int],
    degree: int,
    theta_n0: float,
    theta_p0: float,
    cfg: Config,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
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

    cols = []
    names = []

    cols.append(np.ones_like(I))
    names.append("C")

    for k in range(1, degree + 1):
        cols.append(x_p**k)
        names.append("xp" if k == 1 else f"xp^{k}")

    for k in range(1, degree + 1):
        cols.append(-(x_n**k))
        names.append("-xn" if k == 1 else f"-xn^{k}")

    cols.append(I)
    names.append("I")

    for k in range(1, degree + 1):
        cols.append(z_e**k)
        names.append("ze" if k == 1 else f"ze^{k}")

    Phi = np.vstack(cols).T.astype(np.float64)

    return Phi, names, {
        "x_n": x_n,
        "x_p": x_p,
        "z_e": z_e,
        "ce_left": ce_left,
        "ce_right": ce_right,
    }


def split_components_np(
    X: np.ndarray,
    I: np.ndarray,
    phys: dict[str, Any],
    state_spec: dict[str, int],
    degree: int,
    cfg: Config,
) -> dict[str, Any]:
    Phi, names, vars_ = candidate_feature_matrix_np(
        X=X,
        I=I,
        state_spec=state_spec,
        degree=degree,
        theta_n0=phys["theta_n0"],
        theta_p0=phys["theta_p0"],
        cfg=cfg,
    )

    beta = phys["beta"]
    yhat = Phi @ beta

    constant = np.zeros_like(yhat)
    ocp_positive = np.zeros_like(yhat)
    ocp_negative = np.zeros_like(yhat)
    ocp_total = np.zeros_like(yhat)
    current_branch = np.zeros_like(yhat)
    electrolyte_branch = np.zeros_like(yhat)

    component_cols = {}

    for j, name in enumerate(names):
        val = Phi[:, j] * beta[j]
        component_cols[name] = val

        if name == "C":
            constant += val
        elif name.startswith("xp"):
            ocp_positive += val
            ocp_total += val
        elif name.startswith("-xn"):
            ocp_negative += val
            ocp_total += val
        elif name == "I":
            current_branch += val
        elif name.startswith("ze"):
            electrolyte_branch += val

    return {
        "Phi": Phi,
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


# %% =====================================================
# CELL 9 — JAX CT model factories
# =====================================================
def make_state_fcn(state_spec: dict[str, int], cfg: Config):
    n_n = int(state_spec["n_solid_n"])
    n_p = int(state_spec["n_solid_p"])
    n_e = int(state_spec["n_electrolyte"])

    idx = make_state_indices(state_spec)

    def state_fcn(x, u, t, params):
        phys = unpack_params_jax(params, cfg)

        alpha_n = phys["alpha_n"]
        alpha_p = phys["alpha_p"]
        K_e = phys["K_e"]

        g_n = phys["g_n"]
        g_p = phys["g_p"]
        g_e = phys["g_e"]

        I_in = u[0]

        xn = x[idx["idx_n_start"]: idx["idx_n_start"] + n_n]
        xp = x[idx["idx_p_start"]: idx["idx_p_start"] + n_p]
        ce = x[idx["idx_e_start"]: idx["idx_e_start"] + n_e]

        A_n = build_solid_A_general_jax(n_n, alpha_n)
        A_p = build_solid_A_general_jax(n_p, alpha_p)
        A_e = build_electrolyte_A_general_jax(n_e, K_e, cfg)

        dxn = A_n @ xn
        dxp = A_p @ xp
        dce = A_e @ ce

        dxn = dxn.at[n_n - 1].add(cfg.solid_input_sign_n * g_n * I_in)
        dxp = dxp.at[n_p - 1].add(cfg.solid_input_sign_p * g_p * I_in)

        m_n, m_sep, m_p = electrolyte_region_counts(n_e)

        for local_idx in range(0, m_n):
            dce = dce.at[local_idx].add(cfg.electrolyte_input_left_sign * g_e * I_in / m_n)

        for local_idx in range(m_n + m_sep, m_n + m_sep + m_p):
            dce = dce.at[local_idx].add(cfg.electrolyte_input_right_sign * g_e * I_in / m_p)

        return jnp.concatenate([dxn, dxp, dce])

    return state_fcn


def make_output_fcn(state_spec: dict[str, int], degree: int, cfg: Config):
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

        feats = []

        feats.append(jnp.array(1.0, dtype=jnp.float64))

        for k in range(1, degree + 1):
            feats.append(x_p**k)

        for k in range(1, degree + 1):
            feats.append(-(x_n**k))

        feats.append(I_in)

        for k in range(1, degree + 1):
            feats.append(z_e**k)

        phi = jnp.stack(feats)
        y = jnp.dot(phi, beta)

        return jnp.array([y], dtype=jnp.float64)

    return output_fcn


# %% =====================================================
# CELL 10 — Fit one model/seed
# =====================================================
def fit_one_real_cycle_model(
    state_id: str,
    candidate_id: str,
    seed: int,
    cfg: Config,
) -> dict[str, Any]:
    state_spec = STATE_VARIANTS[state_id]
    degree = int(OUTPUT_CANDIDATES[candidate_id]["degree"])
    nx = int(state_spec["n_solid_n"] + state_spec["n_solid_p"] + state_spec["n_electrolyte"])

    y0 = float(Y_id.reshape(-1)[0])

    params0 = make_initial_params(seed, degree, y0, cfg)

    model = CTModel(
        nx,
        1,
        1,
        state_fcn=make_state_fcn(state_spec, cfg),
        output_fcn=make_output_fcn(state_spec, degree, cfg),
        x0=np.zeros(nx, dtype=np.float64),
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

    phys = unpack_params_np(model.params, state_spec, cfg)

    comps = split_components_np(
        X=Xhat,
        I=i_id,
        phys=phys,
        state_spec=state_spec,
        degree=degree,
        cfg=cfg,
    )

    residual = ytrue - yhat

    rank_phi_raw = matrix_rank_condition_svd_np(comps["Phi"], cfg.svd_tol)
    rank_X_raw = matrix_rank_condition_svd_np(Xhat, cfg.svd_tol)
    lb = ljung_box_test(residual, cfg.ljung_box_lag)

    result = {
        "state_id": state_id,
        "candidate_id": candidate_id,
        "model_id": f"{state_id}_{candidate_id}",
        "state_spec": state_spec,
        "degree": degree,
        "seed": seed,
        "nx": nx,
        "Yhat": Yhat,
        "Xhat": Xhat,
        "yhat": yhat,
        "residual": residual,
        "rmse": rmse(ytrue, yhat),
        "mae": mae(ytrue, yhat),
        "r2": r2_percent(ytrue, yhat),
        "bfr": bfr_percent(ytrue, yhat),
        "sse": sse(ytrue, yhat),
        "phys": phys,
        "components": comps,
        "rank_phi_raw": rank_phi_raw,
        "rank_X_raw": rank_X_raw,
        "ljung_box_Q": lb["Q"],
        "ljung_box_p_value": lb["p_value"],
    }

    del model
    gc.collect()

    try:
        jax.clear_caches()
    except Exception:
        pass

    return result


# %% =====================================================
# CELL 11 — Run full real-data grid
# =====================================================
all_results: list[dict[str, Any]] = []

print("\n" + "=" * 100)
print("STARTING REAL-CYCLE STATE/ORDER CT-ID GRID")
print("=" * 100)
print("State variants:", SELECTED_STATE_VARIANTS)
print("Output candidates:", SELECTED_OUTPUT_CANDIDATES)
print("Seeds:", CFG.seed0, "to", CFG.seed0 + CFG.n_multistart - 1)
print("Total fits:", len(SELECTED_STATE_VARIANTS) * len(SELECTED_OUTPUT_CANDIDATES) * CFG.n_multistart)
print("=" * 100)

for state_id in SELECTED_STATE_VARIANTS:
    for candidate_id in SELECTED_OUTPUT_CANDIDATES:
        print("\n" + "#" * 100)
        print(f"FITTING MODEL {state_id}_{candidate_id}")
        print("State spec:", STATE_VARIANTS[state_id])
        print("Output:", output_equation_for_degree(OUTPUT_CANDIDATES[candidate_id]["degree"]))
        print("#" * 100)

        for k in range(CFG.n_multistart):
            seed = CFG.seed0 + k

            print("\n" + "-" * 100)
            print(f"{state_id}_{candidate_id} | seed {seed} | {k + 1}/{CFG.n_multistart}")
            print("-" * 100)

            try:
                res = fit_one_real_cycle_model(
                    state_id=state_id,
                    candidate_id=candidate_id,
                    seed=seed,
                    cfg=CFG,
                )
                all_results.append(res)

                print(
                    f"{state_id}_{candidate_id} seed={seed} | "
                    f"RMSE={res['rmse']:.6e} | "
                    f"MAE={res['mae']:.6e} | "
                    f"R2={res['r2']:.6f}% | "
                    f"BFR={res['bfr']:.6f}% | "
                    f"alpha_n={res['phys']['alpha_n']:.8g} | "
                    f"alpha_p={res['phys']['alpha_p']:.8g} | "
                    f"K_e={res['phys']['K_e']:.8g} | "
                    f"g_n={res['phys']['g_n']:.8g} | "
                    f"g_p={res['phys']['g_p']:.8g} | "
                    f"g_e={res['phys']['g_e']:.8g} | "
                    f"rankPhi={res['rank_phi_raw']['rank']}/{res['rank_phi_raw']['n_cols']} | "
                    f"rankX={res['rank_X_raw']['rank']}/{res['rank_X_raw']['n_cols']}"
                )

            except Exception as exc:
                print(f"[FAIL] model={state_id}_{candidate_id}, seed={seed}: {repr(exc)}")

if len(all_results) == 0:
    raise RuntimeError("All real-cycle CT-ID fits failed.")


# %% =====================================================
# CELL 12 — Build result tables
# =====================================================
rows = []
beta_rows = []
param_rows = []

for r in all_results:
    phys = r["phys"]
    comps = r["components"]
    beta = phys["beta"]
    names = comps["names"]

    row = {
        "state_id": r["state_id"],
        "candidate_id": r["candidate_id"],
        "model_id": r["model_id"],
        "degree": r["degree"],
        "nx": r["nx"],
        "n_solid_n": r["state_spec"]["n_solid_n"],
        "n_solid_p": r["state_spec"]["n_solid_p"],
        "n_electrolyte": r["state_spec"]["n_electrolyte"],
        "seed": r["seed"],
        "rmse": r["rmse"],
        "mae": r["mae"],
        "r2_percent": r["r2"],
        "bfr_percent": r["bfr"],
        "sse": r["sse"],
        "alpha_n_hat": phys["alpha_n"],
        "alpha_p_hat": phys["alpha_p"],
        "K_e_hat": phys["K_e"],
        "g_n_hat": phys["g_n"],
        "g_p_hat": phys["g_p"],
        "g_e_hat": phys["g_e"],
        "theta_n0_hat": phys["theta_n0"],
        "theta_p0_hat": phys["theta_p0"],
        "rank_phi_raw": r["rank_phi_raw"]["rank"],
        "ncols_phi_raw": r["rank_phi_raw"]["n_cols"],
        "cond_phi_raw": r["rank_phi_raw"]["condition_number"],
        "rank_X_raw": r["rank_X_raw"]["rank"],
        "ncols_X_raw": r["rank_X_raw"]["n_cols"],
        "cond_X_raw": r["rank_X_raw"]["condition_number"],
        "ljung_box_Q": r["ljung_box_Q"],
        "ljung_box_p_value": r["ljung_box_p_value"],
    }

    for j, name in enumerate(names):
        safe_name = (
            name.replace("^", "pow")
            .replace("-", "minus")
            .replace(" ", "_")
        )
        col = f"beta_{safe_name}"
        row[col] = beta[j]

        beta_rows.append(
            {
                "state_id": r["state_id"],
                "candidate_id": r["candidate_id"],
                "model_id": r["model_id"],
                "degree": r["degree"],
                "nx": r["nx"],
                "seed": r["seed"],
                "beta_name": name,
                "beta_hat": beta[j],
            }
        )

        param_rows.append(
            {
                "state_id": r["state_id"],
                "candidate_id": r["candidate_id"],
                "model_id": r["model_id"],
                "degree": r["degree"],
                "nx": r["nx"],
                "seed": r["seed"],
                "parameter": col,
                "estimate": beta[j],
            }
        )

    for p in [
        "alpha_n_hat",
        "alpha_p_hat",
        "K_e_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
        "theta_n0_hat",
        "theta_p0_hat",
    ]:
        param_rows.append(
            {
                "state_id": r["state_id"],
                "candidate_id": r["candidate_id"],
                "model_id": r["model_id"],
                "degree": r["degree"],
                "nx": r["nx"],
                "seed": r["seed"],
                "parameter": p,
                "estimate": row[p],
            }
        )

    rows.append(row)

df_all = pd.DataFrame(rows)
df_beta = pd.DataFrame(beta_rows)
df_params = pd.DataFrame(param_rows)

df_all = df_all.sort_values(["model_id", "rmse"]).reset_index(drop=True)

best_rows = []
summary_rows = []

for model_id, g in df_all.groupby("model_id"):
    g = g.sort_values("rmse").reset_index(drop=True)
    best = g.iloc[0].copy()
    best_rows.append(best)

    summary_rows.append(
        {
            "model_id": model_id,
            "state_id": best["state_id"],
            "candidate_id": best["candidate_id"],
            "degree": int(best["degree"]),
            "nx": int(best["nx"]),
            "n_runs": int(len(g)),
            "best_seed": int(best["seed"]),
            "best_rmse": float(best["rmse"]),
            "best_mae": float(best["mae"]),
            "best_r2_percent": float(best["r2_percent"]),
            "best_bfr_percent": float(best["bfr_percent"]),
            "mean_rmse": float(g["rmse"].mean()),
            "median_rmse": float(g["rmse"].median()),
            "std_rmse": float(g["rmse"].std(ddof=1)),
            "q05_rmse": float(g["rmse"].quantile(0.05)),
            "q95_rmse": float(g["rmse"].quantile(0.95)),
            "best_rank_phi_raw": int(best["rank_phi_raw"]),
            "best_ncols_phi_raw": int(best["ncols_phi_raw"]),
            "best_cond_phi_raw": float(best["cond_phi_raw"]),
            "best_rank_X_raw": int(best["rank_X_raw"]),
            "best_ncols_X_raw": int(best["ncols_X_raw"]),
            "best_cond_X_raw": float(best["cond_X_raw"]),
        }
    )

df_best = pd.DataFrame(best_rows).sort_values("rmse").reset_index(drop=True)

# %% =====================================================
# EXTRA CELL — Save measured-vs-estimated step-response data
# =====================================================
"""
Save full measured-vs-estimated response for the best seed of each fitted model.

This is needed for visual inspection:

    measured step response:     V_meas(t)
    estimated step response:    V_hat(t)
    residual:                   e(t) = V_meas(t) - V_hat(t)

RMSE is only a scalar. This time-series export lets us inspect whether the
model follows the transient shape, long-time decay, and residual structure.

This cell saves:
    1. best_step_response_manifest.csv
    2. measured/estimated/residual time-series CSV
    3. fitted state trajectory CSV
    4. voltage components CSV when available
    5. feature matrix Phi CSV when available
    6. measured-vs-estimated plot
    7. residual plot
    8. current + voltage plot
"""

BEST_RESPONSE_DIR = OUT_DIR / "best_step_response_timeseries"
BEST_RESPONSE_FIG_DIR = FIG_DIR / "best_step_response_timeseries"

BEST_RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
BEST_RESPONSE_FIG_DIR.mkdir(parents=True, exist_ok=True)


def _as_1d_float_array(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _safe_file_text(text):
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(";", "_")
        .replace(",", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def _save_best_response_one_result(result):
    """
    Save measured-vs-estimated response for one fitted result dictionary.

    Expected result keys:
        model_id, state_id, candidate_id, seed, degree, nx,
        yhat, residual, Xhat,
        rmse, mae, r2, bfr,
        optionally components
    """
    model_id = str(result["model_id"])
    state_id = str(result["state_id"])
    candidate_id = str(result["candidate_id"])
    seed = int(result["seed"])

    t = _as_1d_float_array(T_id)
    i = _as_1d_float_array(i_id)
    y_meas = _as_1d_float_array(Y_id)
    y_hat = _as_1d_float_array(result["yhat"])
    residual = _as_1d_float_array(result["residual"])

    n = min(len(t), len(i), len(y_meas), len(y_hat), len(residual))

    t = t[:n]
    i = i[:n]
    y_meas = y_meas[:n]
    y_hat = y_hat[:n]
    residual = residual[:n]

    rmse_value = rmse(y_meas, y_hat)
    mae_value = mae(y_meas, y_hat)
    sse_value = sse(y_meas, y_hat)
    r2_value = r2_percent(y_meas, y_hat)
    bfr_value = bfr_percent(y_meas, y_hat)

    residual_bias = float(np.mean(residual))
    residual_std = float(np.std(residual, ddof=1)) if n > 1 else np.nan
    max_abs_residual = float(np.max(np.abs(residual)))

    # -----------------------------------------------------
    # Save measured/estimated time-series
    # -----------------------------------------------------
    df_response = pd.DataFrame(
        {
            "t_s": t,
            "current_A_discharge_positive": i,
            "measured_voltage_V": y_meas,
            "estimated_voltage_V": y_hat,
            "residual_V": residual,
            "abs_residual_V": np.abs(residual),
            "squared_residual_V2": residual**2,
        }
    )

    response_csv = BEST_RESPONSE_DIR / f"{model_id}_best_seed_{seed}_measured_estimated_step_response.csv"
    df_response.to_csv(response_csv, index=False)

    # -----------------------------------------------------
    # Save fitted state trajectory
    # -----------------------------------------------------
    Xhat = np.asarray(result["Xhat"], dtype=np.float64)

    if Xhat.ndim == 2 and Xhat.shape[0] >= n:
        state_data = {"t_s": t}

        for j in range(Xhat.shape[1]):
            state_data[f"xhat_{j}"] = Xhat[:n, j]

        df_states = pd.DataFrame(state_data)
    else:
        df_states = pd.DataFrame({"t_s": t})

    state_csv = BEST_RESPONSE_DIR / f"{model_id}_best_seed_{seed}_state_trajectory.csv"
    df_states.to_csv(state_csv, index=False)

    # -----------------------------------------------------
    # Save voltage components and Phi if available
    # -----------------------------------------------------
    components_csv = ""
    phi_csv = ""

    components = result.get("components", {})

    if isinstance(components, dict) and len(components) > 0:
        component_data = {"t_s": t}

        for key, value in components.items():
            arr = np.asarray(value)

            if arr.ndim == 1 and len(arr) >= n:
                component_data[str(key)] = arr[:n].astype(np.float64)

        df_components = pd.DataFrame(component_data)

        components_csv_path = BEST_RESPONSE_DIR / f"{model_id}_best_seed_{seed}_voltage_components.csv"
        df_components.to_csv(components_csv_path, index=False)
        components_csv = str(components_csv_path)

        if "Phi" in components:
            Phi = np.asarray(components["Phi"], dtype=np.float64)

            if Phi.ndim == 2 and Phi.shape[0] >= n:
                phi_data = {"t_s": t}

                for j in range(Phi.shape[1]):
                    phi_data[f"phi_{j}"] = Phi[:n, j]

                df_phi = pd.DataFrame(phi_data)

                phi_csv_path = BEST_RESPONSE_DIR / f"{model_id}_best_seed_{seed}_feature_matrix_phi.csv"
                df_phi.to_csv(phi_csv_path, index=False)
                phi_csv = str(phi_csv_path)

    # -----------------------------------------------------
    # Plot 1: measured vs estimated step response
    # -----------------------------------------------------
    plt.figure(figsize=(12.5, 6.2))
    plt.plot(
        t,
        y_meas,
        linewidth=2.8,
        label="measured step response",
    )
    plt.plot(
        t,
        y_hat,
        "--",
        linewidth=2.5,
        label="estimated step response",
    )
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(
        f"{model_id}: measured vs estimated step response\n"
        f"seed={seed}, RMSE={rmse_value:.6e}, MAE={mae_value:.6e}, "
        f"R2={r2_value:.6f}%, BFR={bfr_value:.6f}%"
    )
    plt.legend(loc="best")
    plt.tight_layout()

    voltage_fig = BEST_RESPONSE_FIG_DIR / f"{model_id}_best_seed_{seed}_measured_vs_estimated_step_response.png"
    save_or_show(voltage_fig)

    # -----------------------------------------------------
    # Plot 2: residual vs time
    # -----------------------------------------------------
    plt.figure(figsize=(12.5, 4.8))
    plt.plot(
        t,
        residual,
        linewidth=1.9,
        label=r"$e(t)=V_{\mathrm{meas}}(t)-\widehat{V}(t)$",
    )
    plt.axhline(0.0, linestyle="--", linewidth=1.3)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Residual [V]")
    plt.title(
        f"{model_id}: residual over time\n"
        f"bias={residual_bias:.6e} V, std={residual_std:.6e} V, "
        f"max abs={max_abs_residual:.6e} V"
    )
    plt.legend(loc="best")
    plt.tight_layout()

    residual_fig = BEST_RESPONSE_FIG_DIR / f"{model_id}_best_seed_{seed}_residual_vs_time.png"
    save_or_show(residual_fig)

    # -----------------------------------------------------
    # Plot 3: input current and voltage response
    # -----------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(12.5, 6.2))

    ax1.plot(t, y_meas, linewidth=2.6, label="measured voltage")
    ax1.plot(t, y_hat, "--", linewidth=2.4, label="estimated voltage")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Voltage [V]")
    ax1.grid(True, alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(t, i, ":", linewidth=2.0, label="current input")
    ax2.set_ylabel("Current [A], discharge positive")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(f"{model_id}: input current and measured/estimated voltage")
    fig.tight_layout()

    input_voltage_fig = BEST_RESPONSE_FIG_DIR / f"{model_id}_best_seed_{seed}_input_and_voltage_fit.png"
    save_or_show(input_voltage_fig)

    # -----------------------------------------------------
    # Optional component plot
    # -----------------------------------------------------
    component_fig = ""

    if components_csv:
        try:
            df_comp = pd.read_csv(components_csv)

            plot_cols = [
                c for c in df_comp.columns
                if c != "t_s"
                and not c.startswith("phi_")
                and np.issubdtype(df_comp[c].dtype, np.number)
            ]

            if len(plot_cols) > 0:
                plt.figure(figsize=(12.5, 6.2))

                for c in plot_cols:
                    plt.plot(df_comp["t_s"], df_comp[c], linewidth=1.8, label=c)

                plt.grid(True, alpha=0.35)
                plt.xlabel("Time [s]")
                plt.ylabel("Component value")
                plt.title(f"{model_id}: voltage-related component traces")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()

                component_fig_path = BEST_RESPONSE_FIG_DIR / f"{model_id}_best_seed_{seed}_voltage_components.png"
                save_or_show(component_fig_path)
                component_fig = str(component_fig_path)

        except Exception as exc:
            print(f"[warn] component plot failed for {model_id}: {repr(exc)}")

    return {
        "model_id": model_id,
        "state_id": state_id,
        "candidate_id": candidate_id,
        "seed": seed,
        "degree": int(result["degree"]),
        "nx": int(result["nx"]),
        "n_points": int(n),
        "rmse": float(rmse_value),
        "mae": float(mae_value),
        "sse": float(sse_value),
        "r2_percent": float(r2_value),
        "bfr_percent": float(bfr_value),
        "residual_bias_V": residual_bias,
        "residual_std_V": residual_std,
        "max_abs_residual_V": max_abs_residual,
        "response_csv": str(response_csv),
        "state_csv": str(state_csv),
        "components_csv": components_csv,
        "phi_csv": phi_csv,
        "voltage_fit_figure": str(voltage_fig),
        "residual_figure": str(residual_fig),
        "input_voltage_figure": str(input_voltage_fig),
        "component_figure": component_fig,
    }


# ---------------------------------------------------------
# Save the best response for each model in this process
# ---------------------------------------------------------
best_response_rows = []

if "all_results" not in globals() or len(all_results) == 0:
    print("[warn] all_results not found or empty; no step-response files saved.")
else:
    model_ids_here = sorted(set(str(r["model_id"]) for r in all_results))

    for model_id in model_ids_here:
        model_results = [r for r in all_results if str(r["model_id"]) == model_id]

        if len(model_results) == 0:
            continue

        best_result = sorted(model_results, key=lambda r: float(r["rmse"]))[0]

        try:
            row = _save_best_response_one_result(best_result)
            best_response_rows.append(row)
        except Exception as exc:
            print(f"[warn] failed to save measured-vs-estimated response for {model_id}: {repr(exc)}")

df_best_response_manifest = pd.DataFrame(best_response_rows)

manifest_path = OUT_DIR / "best_step_response_manifest.csv"
df_best_response_manifest.to_csv(manifest_path, index=False)

print("\nSaved measured-vs-estimated step-response files:")
print("  manifest:", manifest_path)
print("  CSV folder:", BEST_RESPONSE_DIR)
print("  figure folder:", BEST_RESPONSE_FIG_DIR)

if len(df_best_response_manifest) > 0:
    display_cols = [
        "model_id",
        "seed",
        "rmse",
        "mae",
        "r2_percent",
        "bfr_percent",
        "residual_bias_V",
        "residual_std_V",
        "max_abs_residual_V",
        "response_csv",
        "voltage_fit_figure",
    ]
    display_cols = [c for c in display_cols if c in df_best_response_manifest.columns]
    print(df_best_response_manifest[display_cols].to_string(index=False))

df_best.insert(0, "overall_rank", np.arange(1, len(df_best) + 1))

df_summary = pd.DataFrame(summary_rows).sort_values("best_rmse").reset_index(drop=True)
df_summary.insert(0, "summary_rank", np.arange(1, len(df_summary) + 1))

df_all.to_csv(OUT_DIR / "real_cycle_all_runs.csv", index=False)
df_best.to_csv(OUT_DIR / "real_cycle_best_runs.csv", index=False)
df_summary.to_csv(OUT_DIR / "real_cycle_model_summary.csv", index=False)
df_beta.to_csv(OUT_DIR / "real_cycle_beta_coefficients.csv", index=False)
df_params.to_csv(OUT_DIR / "real_cycle_parameter_long.csv", index=False)

config_payload = {
    "config": asdict(CFG),
    "run_tag": RUN_TAG,
    "selected_state_variants": SELECTED_STATE_VARIANTS,
    "selected_output_candidates": SELECTED_OUTPUT_CANDIDATES,
    "state_variants": STATE_VARIANTS,
    "output_candidates": OUTPUT_CANDIDATES,
    "real_cycle": {
        "cycle_index": CFG.cycle_index,
        "id_samples": int(len(T_id)),
        "id_Ts": float(Ts),
        "current_flip_applied": bool(current_flip_applied),
        "current_mean_discharge_positive": float(np.mean(i_id)),
        "voltage_start": float(v_id[0]),
        "voltage_end": float(v_id[-1]),
        "voltage_min": float(np.min(v_id)),
        "voltage_max": float(np.max(v_id)),
    },
}

with open(OUT_DIR / "real_cycle_config.json", "w", encoding="utf-8") as f:
    json.dump(config_payload, f, indent=2, default=str)

print("\nSaved tables:")
print(" ", OUT_DIR / "real_cycle_all_runs.csv")
print(" ", OUT_DIR / "real_cycle_best_runs.csv")
print(" ", OUT_DIR / "real_cycle_model_summary.csv")
print(" ", OUT_DIR / "real_cycle_beta_coefficients.csv")
print(" ", OUT_DIR / "real_cycle_parameter_long.csv")
print(" ", OUT_DIR / "real_cycle_config.json")

print("\nBest models:")
print(
    df_best[
        [
            "overall_rank",
            "model_id",
            "seed",
            "rmse",
            "mae",
            "r2_percent",
            "bfr_percent",
            "rank_phi_raw",
            "ncols_phi_raw",
            "cond_phi_raw",
            "rank_X_raw",
            "ncols_X_raw",
        ]
    ].to_string(index=False)
)


# %% =====================================================
# CELL 13 — Plot helpers
# =====================================================
def histogram_count_with_lines(
    data,
    title,
    xlabel,
    out_path,
    bins=100,
    true_value=None,
    mean_value=None,
    median_value=None,
    best_value=None,
):
    """
    Robust histogram helper.

    Handles:
        - one-point data
        - constant data
        - too many bins for tiny data range
        - NaN/inf values

    y-axis is count of simulations.
    """
    data = np.asarray(data, dtype=float).reshape(-1)
    data = data[np.isfinite(data)]

    if len(data) == 0:
        print(f"[warn] no finite data for histogram: {title}")
        return

    if mean_value is None:
        mean_value = float(np.mean(data))

    if median_value is None:
        median_value = float(np.median(data))

    data_min = float(np.min(data))
    data_max = float(np.max(data))

    # Critical fix:
    # If there is only one value or all values are identical, force a small range.
    if len(data) < 2 or data_min == data_max:
        center = data_min
        pad = max(abs(center) * 1e-3, 1e-12)
        hist_range = (center - pad, center + pad)
        safe_bins = 1
    else:
        hist_range = None

        # If bins is too large relative to unique values, reduce it safely.
        unique_count = len(np.unique(data))
        safe_bins = int(min(bins, max(1, unique_count * 5)))

        # Still allow high resolution when the data genuinely has spread.
        if unique_count > 20:
            safe_bins = int(bins)

    plt.figure(figsize=(9.5, 5.8))

    plt.hist(
        data,
        bins=safe_bins,
        range=hist_range,
        density=False,
        edgecolor="black",
        alpha=0.75,
    )

    if true_value is not None and np.isfinite(true_value):
        plt.axvline(
            float(true_value),
            linestyle="-",
            linewidth=2.8,
            label=f"true = {float(true_value):.6g}",
        )

    if mean_value is not None and np.isfinite(mean_value):
        plt.axvline(
            float(mean_value),
            linestyle="--",
            linewidth=2.4,
            label=f"mean = {float(mean_value):.6g}",
        )

    if median_value is not None and np.isfinite(median_value):
        plt.axvline(
            float(median_value),
            linestyle=":",
            linewidth=2.8,
            label=f"median = {float(median_value):.6g}",
        )

    if best_value is not None and np.isfinite(best_value):
        plt.axvline(
            float(best_value),
            linestyle="-.",
            linewidth=2.5,
            label=f"best = {float(best_value):.6g}",
        )

    plt.grid(True, axis="y", alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel("Count of simulations")
    plt.title(title)

    if true_value is not None or mean_value is not None or median_value is not None or best_value is not None:
        plt.legend(loc="best", fontsize=8)

    plt.tight_layout()
    save_or_show(out_path)


def plot_best_fit_for_result(result: dict[str, Any]) -> None:
    model_id = result["model_id"]
    seed = int(result["seed"])

    ytrue = Y_id.reshape(-1)
    yhat = result["yhat"]
    residual = result["residual"]
    comps = result["components"]

    model_fig_dir = FIG_DIR / model_id
    model_fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5.8))
    plt.plot(T_id, ytrue, linewidth=2.4, label="measured voltage")
    plt.plot(T_id, yhat, "--", linewidth=2.1, label="CT-ID fit")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"{model_id}: best voltage fit, seed={seed}, RMSE={result['rmse']:.6e}")
    plt.legend(loc="best")
    plt.tight_layout()
    save_or_show(model_fig_dir / f"{model_id}_best_seed_{seed}_voltage_fit.png")

    plt.figure(figsize=(12, 4.8))
    plt.plot(T_id, residual, linewidth=1.8)
    plt.axhline(0.0, linestyle="--", linewidth=1.2)
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Residual [V]")
    plt.title(f"{model_id}: residual time plot, seed={seed}")
    plt.tight_layout()
    save_or_show(model_fig_dir / f"{model_id}_best_seed_{seed}_residual_time.png")

    plt.figure(figsize=(12, 6.4))
    plt.plot(T_id, comps["constant"], linewidth=2.0, label="constant")
    plt.plot(T_id, comps["ocp_positive"], linewidth=2.0, label="positive OCP polynomial")
    plt.plot(T_id, comps["ocp_negative"], linewidth=2.0, label="negative OCP polynomial")
    plt.plot(T_id, comps["ocp_total"], linewidth=2.0, label="total OCP branch")
    plt.plot(T_id, comps["current_branch"], linewidth=2.0, label="current branch")
    plt.plot(T_id, comps["electrolyte_branch"], linewidth=2.0, label="electrolyte branch")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage contribution [V]")
    plt.title(f"{model_id}: best-fit component groups, seed={seed}")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    save_or_show(model_fig_dir / f"{model_id}_best_seed_{seed}_component_groups.png")

    plt.figure(figsize=(12, 5.8))
    plt.plot(T_id, comps["x_n"], linewidth=2.0, label="x_n")
    plt.plot(T_id, comps["x_p"], linewidth=2.0, label="x_p")
    plt.plot(T_id, comps["z_e"], linewidth=2.0, label="z_e")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.title(f"{model_id}: best-fit physical variables, seed={seed}")
    plt.legend(loc="best")
    plt.tight_layout()
    save_or_show(model_fig_dir / f"{model_id}_best_seed_{seed}_physical_variables.png")

    plt.figure(figsize=(12, 5.8))
    Xhat = result["Xhat"]
    for j in range(Xhat.shape[1]):
        plt.plot(T_id, Xhat[:, j], linewidth=1.2, alpha=0.75, label=f"x{j}")
    plt.grid(True, alpha=0.35)
    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.title(f"{model_id}: fitted state trajectories, seed={seed}")
    if Xhat.shape[1] <= 14:
        plt.legend(loc="best", fontsize=7, ncol=4)
    plt.tight_layout()
    save_or_show(model_fig_dir / f"{model_id}_best_seed_{seed}_all_states.png")


# %% =====================================================
# CELL 14 — Main comparison figures
# =====================================================
if CFG.make_plots:
    # Best RMSE by model.
    plt.figure(figsize=(13, 6))
    plt.bar(df_summary["model_id"], df_summary["best_rmse"])
    plt.grid(True, axis="y", alpha=0.35)
    plt.xlabel("Model")
    plt.ylabel("Best RMSE [V]")
    plt.title("Best RMSE by state variant and output polynomial order")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_or_show(FIG_DIR / "comparison_best_rmse_by_model.png")

    # Scatter RMSE by model.
    plt.figure(figsize=(14, 6.2))
    rng = np.random.default_rng(123)
    model_ids = list(df_summary["model_id"].values)
    model_to_x = {m: i + 1 for i, m in enumerate(model_ids)}

    for model_id in model_ids:
        g = df_all[df_all["model_id"] == model_id]
        x0 = model_to_x[model_id]
        jitter = rng.normal(0.0, 0.04, size=len(g))
        plt.scatter(np.full(len(g), x0) + jitter, g["rmse"].values, s=18, alpha=0.65)

    plt.xticks(np.arange(1, len(model_ids) + 1), model_ids, rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.35)
    plt.xlabel("Model")
    plt.ylabel("RMSE [V]")
    plt.title("RMSE scatter across seeds for each state/order model")
    plt.tight_layout()
    save_or_show(FIG_DIR / "comparison_rmse_scatter_by_model.png")

    # Rank evolution: output degree vs best Phi rank for each state variant.
    plt.figure(figsize=(10.5, 6))
    for state_id in SELECTED_STATE_VARIANTS:
        sub = df_summary[df_summary["state_id"] == state_id].sort_values("degree")
        if len(sub) == 0:
            continue
        plt.plot(sub["degree"], sub["best_rank_phi_raw"], marker="o", linewidth=2.2, label=f"{state_id} rank")
        plt.plot(sub["degree"], sub["best_ncols_phi_raw"], marker="s", linestyle="--", linewidth=1.8, label=f"{state_id} columns")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Output polynomial degree")
    plt.ylabel("Rank / number of columns")
    plt.title("Feature-matrix rank evolution with polynomial order")
    plt.xticks([1, 2, 3, 4], ["C1", "C2", "C3", "C4"])
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    save_or_show(FIG_DIR / "rank_evolution_phi_by_state_and_order.png")

    # State trajectory rank evolution.
    plt.figure(figsize=(10.5, 6))
    for state_id in SELECTED_STATE_VARIANTS:
        sub = df_summary[df_summary["state_id"] == state_id].sort_values("degree")
        if len(sub) == 0:
            continue
        plt.plot(sub["degree"], sub["best_rank_X_raw"], marker="o", linewidth=2.2, label=f"{state_id} state rank")
        plt.plot(sub["degree"], sub["best_ncols_X_raw"], marker="s", linestyle="--", linewidth=1.8, label=f"{state_id} states")

    plt.grid(True, alpha=0.35)
    plt.xlabel("Output polynomial degree")
    plt.ylabel("State trajectory rank / state dimension")
    plt.title("Fitted state-trajectory rank evolution")
    plt.xticks([1, 2, 3, 4], ["C1", "C2", "C3", "C4"])
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    save_or_show(FIG_DIR / "rank_evolution_state_trajectory_by_state_and_order.png")

    # Condition number evolution.
    plt.figure(figsize=(10.5, 6))
    for state_id in SELECTED_STATE_VARIANTS:
        sub = df_summary[df_summary["state_id"] == state_id].sort_values("degree")
        if len(sub) == 0:
            continue
        plt.plot(sub["degree"], sub["best_cond_phi_raw"], marker="o", linewidth=2.2, label=f"{state_id}")

    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.35)
    plt.xlabel("Output polynomial degree")
    plt.ylabel("Best fitted Phi condition number")
    plt.title("Feature-matrix conditioning evolution")
    plt.xticks([1, 2, 3, 4], ["C1", "C2", "C3", "C4"])
    plt.legend(loc="best")
    plt.tight_layout()
    save_or_show(FIG_DIR / "condition_evolution_phi_by_state_and_order.png")


# %% =====================================================
# CELL 15 — Best-fit figures and histograms per model
# =====================================================
if CFG.make_plots:
    # Map best results for easy lookup.
    best_keys = set((row["model_id"], int(row["seed"])) for _, row in df_best.iterrows())

    for r in all_results:
        if (r["model_id"], int(r["seed"])) in best_keys:
            plot_best_fit_for_result(r)

    # Histograms per model.
    hist_params = [
        "rmse",
        "mae",
        "r2_percent",
        "bfr_percent",
        "alpha_n_hat",
        "alpha_p_hat",
        "K_e_hat",
        "g_n_hat",
        "g_p_hat",
        "g_e_hat",
        "theta_n0_hat",
        "theta_p0_hat",
        "rank_phi_raw",
        "cond_phi_raw",
        "rank_X_raw",
        "cond_X_raw",
    ]

    beta_cols = [c for c in df_all.columns if c.startswith("beta_")]

    for model_id, g in df_all.groupby("model_id"):
        model_fig_dir = FIG_DIR / model_id
        model_fig_dir.mkdir(parents=True, exist_ok=True)

        for param in hist_params:
            if param not in g.columns:
                continue
            histogram_count_with_lines(
                data=g[param].values,
                title=f"{model_id}: {param} distribution over {len(g)} runs",
                xlabel=param,
                out_path=model_fig_dir / f"{model_id}_hist_{param}_count.png",
                bins=CFG.hist_bins,
            )

        for param in beta_cols:
            if param not in g.columns:
                continue

            values = g[param].values
            if np.sum(np.isfinite(values)) == 0:
                continue

            safe_param = param.replace("^", "pow").replace("-", "minus").replace(" ", "_")

            histogram_count_with_lines(
                data=values,
                title=f"{model_id}: {param} distribution over {len(g)} runs",
                xlabel=param,
                out_path=model_fig_dir / f"{model_id}_hist_{safe_param}_count.png",
                bins=CFG.hist_bins,
            )


# %% =====================================================
# CELL 16 — Final printout
# =====================================================
print("\n" + "=" * 100)
print("REAL FIRST-DISCHARGE-CYCLE CT-ID GRID COMPLETE")
print("=" * 100)
print("Output directory:", OUT_DIR)
print("Figure directory:", FIG_DIR)
print("\nTop models:")
print(
    df_best[
        [
            "overall_rank",
            "model_id",
            "seed",
            "nx",
            "degree",
            "rmse",
            "mae",
            "r2_percent",
            "bfr_percent",
            "rank_phi_raw",
            "ncols_phi_raw",
            "cond_phi_raw",
            "rank_X_raw",
            "ncols_X_raw",
        ]
    ].head(20).to_string(index=False)
)
print("=" * 100)