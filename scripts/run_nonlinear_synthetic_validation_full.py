#!/usr/bin/env python3
# ============================================================
# AUTO-GENERATED STANDALONE SCRIPT
# Source notebook:
#   /home/onyero.ofuzim/projects/battery-degradation-spme-sysid/notebooks/nonlinear_synthetic_validation_thesis_clean.ipynb
#
# Do not edit this generated script manually unless needed.
# Regenerate with:
#   python scripts/export_notebook_to_standalone.py
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid")
os.chdir(PROJECT_DIR)

# Make package imports work from project root.
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Non-interactive plotting for Slurm.
os.environ.setdefault("MPLBACKEND", "Agg")

# Thread settings. Slurm can override SLURM_CPUS_PER_TASK.
N_THREADS_EXPORT = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_THREADS", "8")))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS_EXPORT)
os.environ["XLA_FLAGS"] = (
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={N_THREADS_EXPORT}"
)

# Create output directories.
for _p in [
    PROJECT_DIR / "results" / "logs",
    PROJECT_DIR / "results" / "figures",
    PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic",
    PROJECT_DIR / "results" / "metrics",
    PROJECT_DIR / "results" / "tables",
    PROJECT_DIR / "results" / "outputs",
]:
    _p.mkdir(parents=True, exist_ok=True)

# Batch-safe display replacement.
def display(obj=None, *args, **kwargs):
    try:
        print(obj)
    except Exception:
        print(repr(obj))

# Batch-safe matplotlib setup.
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

_FIG_COUNTER = {"n": 0}
_FIG_DIR = PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic"

def _batch_show(*args, **kwargs):
    """
    Replace plt.show() in notebook-exported code.
    Saves each open figure instead of trying to display interactively.
    """
    figs = list(map(plt.figure, plt.get_fignums()))
    if not figs:
        return

    for fig in figs:
        _FIG_COUNTER["n"] += 1
        out = _FIG_DIR / f"notebook_fig_{_FIG_COUNTER['n']:04d}.png"
        try:
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"[saved figure] {out}")
        except Exception as exc:
            print(f"[warn] could not save figure {out}: {exc}")
    plt.close("all")

plt.show = _batch_show

print("=" * 72)
print("STANDALONE NONLINEAR SYNTHETIC VALIDATION SCRIPT")
print("=" * 72)
print("Project dir:", PROJECT_DIR)
print("Source notebook:", Path("/home/onyero.ofuzim/projects/battery-degradation-spme-sysid/notebooks/nonlinear_synthetic_validation_thesis_clean.ipynb"))
print("Python:", sys.executable)
print("Threads:", N_THREADS_EXPORT)
print("=" * 72)



# ========================================================================
# EXPORTED NOTEBOOK CODE CELL 0
# ========================================================================
# nonlinear_synthetic_validation_thesis_clean
# controlled synthetic validation data
# %% =====================================================
# CELL 0 — Imports / environment
# Thesis notebook setup
#
# Purpose of this notebook
# ------------------------
# This notebook implements the Thesis workflow for
# validating a reduced SPMe-based synthetic battery model
# and identifying an Output-Error (OE) model from the
# generated voltage response.
#
# Final thesis workflow
# ---------------------
#   1) configure a reduced 7-state SPMe-inspired truth model
#      and the corresponding nonlinear terminal-voltage map
#
#   2) generate a synthetic voltage/current dataset, or load
#      a real discharge segment for comparison
#
#   3) build a "comparison synthetic" dataset whose duration,
#      excitation timing, and voltage shape are tuned to
#      resemble the selected real discharge segment
#
#   4) choose the actual identification source:
#         - default synthetic truth, or
#         - tuned comparison synthetic
#
#   5) optionally downsample only the identification signal
#      (without changing the underlying synthetic truth
#      simulation) to study how sampling affects OE fitting
#
#   6) construct the reporting target and the internal
#      modeling signal, then apply scaling for numerically
#      stable OE estimation
#
#   7) sweep OE model orders across pole/zero combinations,
#      subject to the thesis rule that the final reported
#      model should use 7 poles to match the reduced 7-state
#      truth generator
#
#   8) evaluate fit quality using RMSE / R2 / BFR, inspect
#      residual behavior, and compare candidate structures
#
#   9) compute the continuous-time poles of the SPMe truth
#      model and the discrete-time poles of the discretized
#      truth system at the actual identification sample time
#
#  10) compare the poles of the final identified OE model
#      against the poles of the SPMe data-generating system
#      using printed matches and clean pole plots
#
# Active thesis direction
# -----------------------
#   - primary source mode: synthetic
#   - primary identification target: absolute voltage
#   - primary synthetic output: nonlinear terminal voltage
#   - tuned comparison synthetic used for ID
#   - ID-only downsampling used as a sensitivity study
#   - final thesis choice: 5 s identification downsampling
#   - final thesis comparison: 7-pole OE model vs reduced_7 SPMe truth poles
#
# Notes
# -----
#   - The active executable truth model is the reduced 7-state
#     variant used for the thesis pole-comparison study.
#   - A linear-output option may still exist in the notebook
#     as a diagnostic path, but the main thesis path uses the
#     nonlinear terminal-voltage output.
#   - BJ-related scaffolding may remain in the notebook for
#     completeness, but the final thesis workflow is centered
#     on OE identification and pole comparison.
# =====================================================
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

N_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS)
os.environ["XLA_FLAGS"] = (
    f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={N_THREADS}"
)
# from IPython.display import display  # replaced by batch-safe display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import scipy.signal as sig
from scipy.linalg import block_diag, expm
from scipy.optimize import least_squares
import control as ct

try:
    from galvani import BioLogic
except ImportError:
    BioLogic = None

warnings.filterwarnings("ignore", category=UserWarning)
print("Threads:", N_THREADS)


# %% =====================================================
# CELL 1 — Nonlinear ID settings: authoritative active config
# Purpose:
#   This notebook is now treated as a nonlinear CT-ID notebook.
#   OE/BJ variables are defined only as harmless compatibility
#   placeholders because older downstream cells may still reference them.
# =====================================================

# ---------------------------------------------------------
# Main notebook mode
# ---------------------------------------------------------
SOURCE_MODE = "synthetic"          # "synthetic" | "real"
SYSID_MODEL = "nonlinear_ct"

CURRENT_EXPERIMENT = "exp_step_absolute_full_voltage"

EXPERIMENT_LIBRARY = {
    "exp_step_absolute_full_voltage": {
        "identification_target": "absolute_voltage",
        "synthetic_mode": "step",
        "truth_weights": {
            "ocv": 0.95,
            "eta": 1.75,
            "electrolyte": 1.60,
            "ohmic": 1.15,
        },
        "truth_params": {
            "bv_scale": 1.15,
            "R_ohm": 0.0014,
            "Rf": 0.00055,
            "kappa_s_eff": 0.35,
        },
    },
}

if CURRENT_EXPERIMENT not in EXPERIMENT_LIBRARY:
    raise ValueError(f"Unsupported CURRENT_EXPERIMENT={CURRENT_EXPERIMENT!r}")

EXP_CFG = EXPERIMENT_LIBRARY[CURRENT_EXPERIMENT]

# ---------------------------------------------------------
# Truth / nonlinear model setup
# ---------------------------------------------------------
STATE_SPACE_VARIANT = "full_14"
IDENTIFICATION_TARGET = EXP_CFG["identification_target"]
SYNTHETIC_MODE = EXP_CFG["synthetic_mode"]
SYNTHETIC_OUTPUT_MODE = "nonlinear_voltage"

RUN_NONLINEAR_SYNTHETIC_ID = True
NONLINEAR_ID_TARGET = "absolute_voltage"
NONLINEAR_USE_ID_SIGNAL = True
NONLINEAR_USE_POST_STEP_ONLY = False
NONLINEAR_CENTER_OUTPUT = True

NONLINEAR_T_MIN = None
NONLINEAR_T_MAX = None

NONLINEAR_SURFACE_GRID_N = 60
NONLINEAR_SURFACE_GUARD = 1e-4

# ---------------------------------------------------------
# Identification source
# ---------------------------------------------------------
USE_COMPARISON_SYNTHETIC_FOR_ID = True
COMPARISON_SYNTHETIC_ID_VOLTAGE = "shifted"  # "raw" | "shifted" | "affine_aligned"
COMPARISON_SYNTHETIC_ID_USE_FULL_TRACE = True

CURRENT_SIGN_MODE = "flip"

# Legacy name used downstream. In this nonlinear notebook it simply means:
# center the internal model output when building the frozen ID trace.
CENTER_OUTPUT_FOR_OE = True

# ---------------------------------------------------------
# ID downsampling
# ---------------------------------------------------------
ENABLE_ID_DOWNSAMPLE = True
ID_DOWNSAMPLE_DT = 5.0
ID_DOWNSAMPLE_USE_INTERP = False
PRINT_ID_DOWNSAMPLE_SUMMARY = True

FIT_WINDOW_MODE = "all"            # "all" | "post_step_only" | "manual"
FIT_START_TIME = 0.0
FIT_END_TIME = 690.0
EXCLUDE_POST_STEP_SAMPLES = 0

# ---------------------------------------------------------
# Synthetic truth simulation
# ---------------------------------------------------------
SIM_DT = 0.1
SIM_T_END = 250.0

USE_PRECONDITION = False
T_PRE = 4000.0
I_PRE = 2.0

STEP_PROFILE_STYLE = "deviation_id"
I_BEFORE = 0.0
I_AFTER = 2.0
T_STEP = 90.0

I_CONST = 2.0
MULTISTEP_BREAKS = [0.0, 10.0, 25.0, 45.0, 65.0]
MULTISTEP_LEVELS = [2.0, 2.5, 1.5, 2.7, 2.2]

ADD_VOLTAGE_NOISE = False
V_NOISE_STD = 1e-4

THETA_N0 = 0.80
THETA_P0 = 0.40
CE0_DEV = 0.0

N_SERIES = 3
DISCHARGE_POSITIVE = True

THETA_MIN_CUTOFF = 0.02
THETA_MAX_CUTOFF = 0.98
V_MIN_CUTOFF = 2.70 * 3.0
V_MAX_CUTOFF = 4.25 * 3.0

RUN_DURATION_ESTIMATE = False
DURATION_TEST_DT = 1.0
DURATION_TEST_MAX_TIME = 40000.0
DURATION_TEST_CURRENTS = [2.0, -2.0]

# Optional x0 override used by some old helper paths.
X0_OVERRIDE = None

# ---------------------------------------------------------
# Truth voltage model
# ---------------------------------------------------------
OCV_MODEL_MODE = "electrode_ocp"

USE_OCV_SCALE = False
OCV_SCALE = 1.0

USE_SOLID_STOICH_RATE_SCALE = True
SOLID_STOICH_RATE_SCALE = 14.0

FULL_CELL_OCV_MIN = 3.00
FULL_CELL_OCV_MAX = 4.20

USE_KINETIC_TERM = True
USE_ELECTROLYTE_TERM = True
USE_OHMIC_TERM = True

USE_ETA_P = True
USE_ETA_N = True

USE_ELECTROLYTE_LOG_TERM = True
USE_ELECTROLYTE_OHMIC_RESISTANCE = True

USE_PURE_R_OHM = True
USE_FILM_RESISTANCE = True

ELECTROLYTE_LOG_BLEND = 1.80
KINETIC_BLEND = 0.14
OHMIC_BLEND = 1.55

EXCHANGE_CURRENT_SCALE_P = 0.12
EXCHANGE_CURRENT_SCALE_N = 0.32

ELECTROLYTE_STATE_ORIENTATION = "flipped"
ELECTROLYTE_LOG_ORIENTATION = "left_over_right"
USE_DISPLAY_ELECTROLYTE_FLIP = False

# ---------------------------------------------------------
# Truth preset mapping
# ---------------------------------------------------------
EXPERIMENT_PRESET = "pole_id"

POLE_ID_OCV_WEIGHT = EXP_CFG["truth_weights"]["ocv"]
POLE_ID_ETA_WEIGHT = EXP_CFG["truth_weights"]["eta"]
POLE_ID_ELECTROLYTE_WEIGHT = EXP_CFG["truth_weights"]["electrolyte"]
POLE_ID_OHMIC_WEIGHT = EXP_CFG["truth_weights"]["ohmic"]

POLE_ID_BV_SCALE = EXP_CFG["truth_params"]["bv_scale"]
POLE_ID_R_OHM_VALUE = EXP_CFG["truth_params"]["R_ohm"]
POLE_ID_RF_VALUE = EXP_CFG["truth_params"]["Rf"]
POLE_ID_KAPPA_S_EFF_VALUE = EXP_CFG["truth_params"]["kappa_s_eff"]

# ---------------------------------------------------------
# Chebyshev OCV fallback controls
# These are not the active OCV model, but older helper cells
# may still reference them.
# ---------------------------------------------------------
USE_CHEBYSHEV_OCV_CLIP = True
CHEBYSHEV_OCV_PRESET = "manual"

CHEBYSHEV_OCV_COEFFS_DEFAULT_SOFT_SLOPE = [
    3.62, 0.52, 0.06, -0.03, 0.025, -0.010,
]

CHEBYSHEV_OCV_COEFFS_STRONGER_MID_SLOPE = [
    3.62, 0.62, 0.08, -0.05, 0.020, -0.015,
]

CHEBYSHEV_OCV_COEFFS_MORE_END_CURVATURE = [
    3.62, 0.50, 0.03, -0.02, 0.050, -0.030,
]

CHEBYSHEV_OCV_COEFFS_MANUAL = [
    3.58, 0.46, 0.10, -0.08, 0.060, -0.028,
]

CHEBYSHEV_OCV_GAIN = 1.0
CHEBYSHEV_OCV_BIAS = 0.0

CHEBYSHEV_USE_SOC_REMAP = True
CHEBYSHEV_SOC_REMAP_CENTER = 0.58
CHEBYSHEV_SOC_REMAP_GAIN = 0.24

# ---------------------------------------------------------
# Comparison synthetic controls
# ---------------------------------------------------------
COMPARE_REAL_AND_SYNTHETIC = True
COMPARE_SIDE_BY_SIDE = True
COMPARE_NORMALIZED_SHAPES = True
COMPARE_DELTA_SHAPES = True

MATCH_SYNTHETIC_TO_REAL_DURATION = True
MATCH_SYNTHETIC_STEP_TO_REAL_START = True
AUTO_MATCH_SYNTHETIC_CURRENT_TO_REAL = False
MANUAL_SYNTHETIC_CURRENT_SCALE = 2.5
MANUAL_SYNTHETIC_DURATION = None

ENABLE_COMPARISON_ONLY_TUNING = True
COMPARE_SOLID_STOICH_RATE_SCALE = SOLID_STOICH_RATE_SCALE
COMPARE_KINETIC_BLEND = KINETIC_BLEND
COMPARE_OHMIC_BLEND = OHMIC_BLEND
COMPARE_ELECTROLYTE_LOG_BLEND = ELECTROLYTE_LOG_BLEND
COMPARE_EXCHANGE_CURRENT_SCALE_P = EXCHANGE_CURRENT_SCALE_P
COMPARE_EXCHANGE_CURRENT_SCALE_N = EXCHANGE_CURRENT_SCALE_N

# Caches used by old comparison cells.
REAL_COMPARE_CACHE = None
SYNTH_COMPARE_CACHE = None
REAL_COMPARE_META = None
REAL_META_MAIN = None
TRUE_AVAILABLE = False

# ---------------------------------------------------------
# Real data settings used only for comparison synthetic
# ---------------------------------------------------------
MPR_PATH = str(PROJECT_DIR / "data" / "12to1-25%CNC-3%GQDs _C01.mpr")
TIME_COL = "time/s"
I_COL = "I/mA"
V_COL = "Ewe/V"

ALT_I_COL_CANDIDATES = ["control/mA", "I/mA"]

REAL_COMPARE_INPUT_COLUMN_MODE = "control"
REAL_CURRENT_UNITS = "mA"
RAW_DISCHARGE_SIGN = "negative"

CYCLE_INDEX = 0
COMPARE_REAL_INDEX = 0

MIN_CYCLE_LEN = 20
RESAMPLE_REAL = True
REAL_DT = 0.1

# Old real-cycle extraction compatibility flags.
REAL_OE_CYCLE_TYPE = "discharge"
REAL_OE_PREVIOUS_POINTS = 10
REAL_OE_MIN_ACTIVE_CURRENT = 1e-9
REAL_OE_MIN_CYCLE_LEN = 20
REAL_OE_NORMALIZE_TIME_TO_ZERO = True
REAL_OE_DROP_NAN = True
REAL_OE_USE_TIME_BASED_TS = True
REAL_OE_TS_FALLBACK = 10.0

# ---------------------------------------------------------
# Plot / print controls
# Set most old plots to False to prevent massive output.
# Turn on only what you need.
# ---------------------------------------------------------
PLOT_VOLTAGE_COMPONENTS = True
PLOT_COMPONENTS_SEPARATELY = True
PLOT_DETAILED_COMPONENTS = True
PLOT_OCV_DECOMPOSITION = True

PLOT_OCV_SUBTERMS_SEPARATELY = True
PLOT_ELECTROLYTE_SUBTERMS_SEPARATELY = True
PLOT_OHMIC_SUBTERMS_SEPARATELY = True
PLOT_KINETIC_SUBTERMS_SEPARATELY = True

PLOT_GROUPED_OCV_DECOMPOSITION = True
PLOT_GROUPED_ELECTROLYTE_DECOMPOSITION = True
PLOT_GROUPED_OHMIC_DECOMPOSITION = True
PLOT_GROUPED_KINETIC_DECOMPOSITION = True

PLOT_COMPARISON_WITH_SHIFTED_VOLTAGE = True
PLOT_COMPARISON_WITH_NORMALIZED_VOLTAGE = True

PLOT_NONLINEAR_STAGE_FITS = True
PLOT_NONLINEAR_STAGE_RESIDUALS = True
PLOT_NONLINEAR_STAGE_COMPARISON = True
PLOT_NONLINEAR_SURFACES = True

PLOT_STAGE_FITS_POST_STEP_ONLY = True

PRINT_COMPONENT_SUMMARY = True
PRINT_DETAILED_COMPONENT_SUMMARY = False
PRINT_COMPARISON_SUMMARY = True
PRINT_REAL_CURRENT_COLUMN_DIAGNOSTICS = True
PRINT_REAL_RAW_DT_DIAGNOSTICS = True

SHIFT_REAL_COMPARE_BY_TSTEP_FOR_PLOTS = False

# ---------------------------------------------------------
# Nonlinear staged optimizer defaults
# Actual detailed overrides still happen later in Cell 12.
# These are here only so earlier cells never fail.
# ---------------------------------------------------------
RUN_STAGE3 = True
RUN_STAGE3A = True
RUN_STAGE3B = True

# ---------------------------------------------------------
# Legacy linear-ID compatibility only
# These should not drive this nonlinear notebook.
# ---------------------------------------------------------
STABILITY_RADIUS = 1.0
RUN_BJ = False

OE_POLE_ORDERS = []
NB_MAX_FOR_SWEEP = 0
INPUT_DELAY_NK = 0
FIXED_POLES = 0
ZERO_ORDERS = []

BJ_ZERO_ORDERS = []
BJ_NC = 0
BJ_ND = 0
BJ_HIGH_ORDER_INIT = 0

USE_OE_BIAS = False
OE_BIAS_BOUND = 0.0
USE_MONOTONE_REAL_POLES = False
POLE_RADIUS_MIN = 0.0
POLE_RADIUS_MAX = 0.9995
USE_WEIGHTED_OE_RESIDUAL = False

STEP_WEIGHT_BOOST = 1.0
MID_WEIGHT_BOOST = 1.0
TAIL_WEIGHT_BOOST = 1.0
EARLY_STEP_WINDOW = (10.0, 25.0)
MID_STEP_WINDOW = (25.0, 80.0)
TAIL_STEP_WINDOW = (80.0, 120.0)

LS_MAX_NFEV = 0
TRY_RANDOM_RESTARTS = False
N_RANDOM_RESTARTS = 0
RANDOM_PERTURB_SCALE = 0.0
WARMUP_SAMPLES = 0
RMSE_TIE_TOL = 1e-6
COST_TIE_REL_TOL = 1e-3
NUMERATOR_BOUND = 0.0

PLOT_ALL_CANDIDATE_FITS = False
TOP_OE_CANDIDATES_TO_PLOT = 0
ZOOM_AROUND_STEP = False
ZOOM_WINDOW = (0.0, 40.0)
ENABLE_RESIDUAL_TESTS = False
MAX_CORR_LAG = 60
PRINT_ZERO_DETAILS = False
PLOT_ZERO_MAGNITUDE_SUMMARY = False

# ---------------------------------------------------------
# Required prints
# ---------------------------------------------------------
print("=" * 72)
print("ACTIVE NONLINEAR ID CONFIG")
print("=" * 72)
print("SOURCE_MODE:", SOURCE_MODE)
print("SYSID_MODEL:", SYSID_MODEL)
print("CURRENT_EXPERIMENT:", CURRENT_EXPERIMENT)
print("STATE_SPACE_VARIANT:", STATE_SPACE_VARIANT)
print("IDENTIFICATION_TARGET:", IDENTIFICATION_TARGET)
print("SYNTHETIC_MODE:", SYNTHETIC_MODE)
print("USE_PRECONDITION:", USE_PRECONDITION)
print("USE_COMPARISON_SYNTHETIC_FOR_ID:", USE_COMPARISON_SYNTHETIC_FOR_ID)
print("COMPARISON_SYNTHETIC_ID_VOLTAGE:", COMPARISON_SYNTHETIC_ID_VOLTAGE)
print("ENABLE_ID_DOWNSAMPLE:", ENABLE_ID_DOWNSAMPLE)
print("ID_DOWNSAMPLE_DT:", ID_DOWNSAMPLE_DT)
print("CURRENT_SIGN_MODE:", CURRENT_SIGN_MODE)
print("RUN_NONLINEAR_SYNTHETIC_ID:", RUN_NONLINEAR_SYNTHETIC_ID)
print("NONLINEAR_ID_TARGET:", NONLINEAR_ID_TARGET)
print("=" * 72)


# %% =====================================================
# CELL 2 — Generic helpers
# =====================================================
def col(x):
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _ensure_1d_array(x: Any, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def series_span(x: np.ndarray) -> float:
    x = _ensure_1d_array(x, dtype=np.float64)
    if len(x) == 0:
        return 0.0
    return float(np.max(x) - np.min(x))


def normalize_series_signed(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a signed series by its largest absolute value.
    Keeps sign information and returns zeros safely if the
    series is flat.
    """
    x = _ensure_1d_array(x, dtype=np.float64)
    if len(x) == 0:
        return x.copy()
    denom = float(np.max(np.abs(x)))
    if not np.isfinite(denom) or denom < eps:
        return np.zeros_like(x, dtype=np.float64)
    return x / denom


def normalize_for_alignment_plot(y: np.ndarray) -> np.ndarray:
    """
    Shift to zero at the first sample, then normalize by span.
    Used for shape-only comparison plots.
    """
    y = _ensure_1d_array(y, dtype=np.float64)
    if len(y) == 0:
        return y.copy()
    y_shift = y - y[0]
    span = float(np.max(y_shift) - np.min(y_shift))
    if abs(span) < 1e-12:
        denom = max(float(np.max(np.abs(y_shift))), 1.0)
    else:
        denom = span
    return y_shift / denom


def finite_diff_stats(t: np.ndarray) -> dict[str, float]:
    """
    Return basic finite-difference time-step statistics as a dict.
    Used in real_meta for raw and resampled time vectors.
    """
    t = _ensure_1d_array(t, dtype=np.float64)
    if len(t) < 2:
        return {
            "n": int(len(t)),
            "dt_min": np.nan,
            "dt_max": np.nan,
            "dt_mean": np.nan,
            "dt_median": np.nan,
            "dt_std": np.nan,
        }

    dt = np.diff(t)
    return {
        "n": int(len(t)),
        "dt_min": float(np.min(dt)),
        "dt_max": float(np.max(dt)),
        "dt_mean": float(np.mean(dt)),
        "dt_median": float(np.median(dt)),
        "dt_std": float(np.std(dt)),
    }


def print_dt_stats(name: str, t: np.ndarray) -> None:
    """
    Print simple time-step diagnostics for a time vector.
    """
    s = finite_diff_stats(t)
    print(f"{name} dt stats:")
    print("  n         =", s["n"])
    print("  dt_min    =", s["dt_min"])
    print("  dt_max    =", s["dt_max"])
    print("  dt_mean   =", s["dt_mean"])
    print("  dt_median =", s["dt_median"])
    print("  dt_std    =", s["dt_std"])


def rmse(y, yh):
    y = _ensure_1d_array(y)
    yh = _ensure_1d_array(yh)
    return float(np.sqrt(np.mean((yh - y) ** 2)))


def r2_percent(y, yh):
    y = _ensure_1d_array(y)
    yh = _ensure_1d_array(yh)
    denom = np.sum((y - np.mean(y)) ** 2) + 1e-12
    return float(100.0 * (1.0 - np.sum((yh - y) ** 2) / denom))


def bfr_percent(y, yh):
    y = _ensure_1d_array(y)
    yh = _ensure_1d_array(yh)
    denom = np.linalg.norm(y - np.mean(y)) + 1e-12
    return float(100.0 * (1.0 - np.linalg.norm(yh - y) / denom))


def report_fit(name, y, yh):
    print(
        f"{name:35s} | RMSE = {rmse(y, yh):.6g} | "
        f"R2% = {r2_percent(y, yh):.3f} | BFR% = {bfr_percent(y, yh):.3f}"
    )


def score_fit(y_true: np.ndarray, y_hat: np.ndarray, warmup: int = 0) -> dict[str, float]:
    y_true = _ensure_1d_array(y_true)
    y_hat = _ensure_1d_array(y_hat)
    n = min(len(y_true), len(y_hat))
    y_true = y_true[:n]
    y_hat = y_hat[:n]
    if 0 < warmup < n:
        y_true = y_true[warmup:]
        y_hat = y_hat[warmup:]
    if len(y_true) == 0:
        return {"rmse": np.nan, "r2": np.nan, "bfr_percent": np.nan, "sse": np.nan}
    err = y_true - y_hat
    sse = float(np.sum(err ** 2))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - sse / denom) if denom > 0 else np.nan,
        "bfr_percent": float(
            max(
                0.0,
                100.0 * (1.0 - np.linalg.norm(err) / np.linalg.norm(y_true - np.mean(y_true)))
            )
        ) if denom > 0 else np.nan,
        "sse": sse,
    }


def stable_discrete_poles(poles, radius: float | None = None):
    if radius is None:
        radius = float(globals().get("STABILITY_RADIUS", 1.0))
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)
    return bool(np.all(np.abs(poles) < radius))


def nearest_pole_distance(reference_poles, estimated_poles):
    refp = list(np.asarray(reference_poles, dtype=np.complex128).reshape(-1))
    estp = list(np.asarray(estimated_poles, dtype=np.complex128).reshape(-1))
    if len(refp) == 0 or len(estp) == 0:
        return np.nan, []
    remaining = estp.copy()
    matches = []
    for rp in refp:
        if len(remaining) == 0:
            break
        dists = [abs(rp - ep) for ep in remaining]
        j = int(np.argmin(dists))
        matches.append((rp, remaining[j], dists[j]))
        remaining.pop(j)
    mean_dist = float(np.mean([m[2] for m in matches])) if matches else np.nan
    return mean_dist, matches


def resample_uniform(t, u, y, dt):
    t = _ensure_1d_array(t)
    u = _ensure_1d_array(u)
    y = _ensure_1d_array(y)
    tg = np.arange(float(t[0]), float(t[-1]) + dt, dt, dtype=np.float64)
    ug = np.interp(tg, t, u)
    yg = np.interp(tg, t, y)
    return tg, col(ug), col(yg)


def resolve_mpr_path(file_name: str) -> Path:
    p = Path(file_name).expanduser()
    if p.exists():
        return p.resolve()

    cwd = Path.cwd().resolve()
    requested_name = p.name
    requested_stem = Path(requested_name).stem.lower()

    direct_candidates = [
        cwd / requested_name,
        cwd / "data" / requested_name,
        cwd / "Data" / requested_name,
        cwd / "datasets" / requested_name,
        cwd / "Datasets" / requested_name,
        cwd.parent / requested_name,
        cwd.parent / "data" / requested_name,
        cwd.parent / "Data" / requested_name,
        cwd.parent / "datasets" / requested_name,
        cwd.parent / "Datasets" / requested_name,
    ]
    for c in direct_candidates:
        if c.exists():
            return c.resolve()

    def _score_candidate(path_obj: Path) -> tuple[int, int, int, str]:
        name_l = path_obj.name.lower()
        stem_l = path_obj.stem.lower()
        exact_name = 0 if name_l == requested_name.lower() else 1
        exact_stem = 0 if stem_l == requested_stem else 1
        try:
            depth = len(path_obj.relative_to(cwd).parts)
        except Exception:
            depth = 9999
        return (exact_name, exact_stem, depth, str(path_obj))

    recursive_hits = []
    for root in [cwd, cwd.parent]:
        if root.exists():
            try:
                recursive_hits.extend(root.rglob(requested_name))
            except Exception:
                pass

    recursive_hits = [x for x in recursive_hits if x.is_file()]
    if recursive_hits:
        recursive_hits = sorted(set([x.resolve() for x in recursive_hits]), key=_score_candidate)
        chosen = recursive_hits[0]
        print(f"[resolve_mpr_path] Found exact filename recursively: {chosen}")
        return chosen

    fuzzy_hits = []
    for root in [cwd, cwd.parent]:
        if root.exists():
            try:
                for x in root.rglob("*.mpr"):
                    if x.is_file():
                        name_l = x.name.lower()
                        stem_l = x.stem.lower()
                        if requested_stem in stem_l or stem_l in requested_stem or requested_name.lower() in name_l:
                            fuzzy_hits.append(x.resolve())
            except Exception:
                pass

    if fuzzy_hits:
        fuzzy_hits = sorted(set(fuzzy_hits), key=_score_candidate)
        chosen = fuzzy_hits[0]
        print(f"[resolve_mpr_path] Found fuzzy .mpr match recursively: {chosen}")
        return chosen

    search_roots = [str(cwd), str(cwd.parent)]
    raise FileNotFoundError(
        "Could not find MPR file.\n"
        f"  requested: {file_name}\n"
        f"  cwd: {cwd}\n"
        f"  searched direct candidates: {[str(x) for x in direct_candidates]}\n"
        f"  searched recursively under: {search_roots}"
    )


def autocorr(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = _ensure_1d_array(x) - np.mean(_ensure_1d_array(x))
    denom = np.dot(x, x) + 1e-12
    vals = []
    lags = np.arange(0, max_lag + 1)
    for lag in lags:
        vals.append(float(np.dot(x[lag:], x[:len(x) - lag]) / denom))
    return lags, np.asarray(vals)


def crosscorr(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = _ensure_1d_array(x) - np.mean(_ensure_1d_array(x))
    y = _ensure_1d_array(y) - np.mean(_ensure_1d_array(y))
    denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    lags = np.arange(-max_lag, max_lag + 1)
    vals = []
    for lag in lags:
        if lag < 0:
            xs = x[:lag]
            ys = y[-lag:]
        elif lag > 0:
            xs = x[lag:]
            ys = y[:-lag]
        else:
            xs = x
            ys = y
        vals.append(float(np.dot(xs, ys) / denom))
    return lags, np.asarray(vals)


def summarize_residual_tests(name: str, e: np.ndarray, u: np.ndarray, max_lag: int = 40):
    lags_e, acf_e = autocorr(e, max_lag)
    lags_eu, ccf_eu = crosscorr(e, u, max_lag)
    conf = 1.96 / np.sqrt(max(len(e), 1))
    peak_acf = float(np.max(np.abs(acf_e[1:]))) if len(acf_e) > 1 else np.nan
    peak_ccf = float(np.max(np.abs(ccf_eu))) if len(ccf_eu) > 0 else np.nan
    print(f"\nResidual tests for {name}:")
    print(f"  95% confidence approx: ±{conf:.4f}")
    print(f"  max |residual ACF| for lag>=1: {peak_acf:.4f}")
    print(f"  max |corr(residual, input)|: {peak_ccf:.4f}")
    return {
        "lags_acf": lags_e,
        "acf": acf_e,
        "lags_ccf": lags_eu,
        "ccf": ccf_eu,
        "conf": conf,
    }


def safe_scale_from_signal(x: np.ndarray, floor: float = 1e-12) -> float:
    x = _ensure_1d_array(x)
    x_std = float(np.std(x))
    if np.isfinite(x_std) and x_std > 1e-10:
        return x_std

    x_span = float(np.max(x) - np.min(x)) if len(x) else 0.0
    if np.isfinite(x_span) and x_span > 1e-10:
        return x_span

    x_abs = float(np.max(np.abs(x))) if len(x) else 0.0
    if np.isfinite(x_abs) and x_abs > 1e-10:
        return x_abs

    return floor


def build_fit_mask_from_time(t: np.ndarray) -> np.ndarray:
    t = _ensure_1d_array(t)

    if FIT_WINDOW_MODE == "all":
        mask = np.ones_like(t, dtype=bool)

    elif FIT_WINDOW_MODE == "post_step_only":
        mask = t >= float(T_STEP)

    elif FIT_WINDOW_MODE == "manual":
        t_start = float(FIT_START_TIME) if FIT_START_TIME is not None else float(t[0])
        t_end = float(FIT_END_TIME) if FIT_END_TIME is not None else float(t[-1])
        mask = (t >= t_start) & (t <= t_end)

    else:
        raise ValueError(f"Unsupported FIT_WINDOW_MODE={FIT_WINDOW_MODE!r}")

    if SYNTHETIC_MODE == "step" and EXCLUDE_POST_STEP_SAMPLES > 0:
        post_idx = np.where(t >= float(T_STEP))[0]
        if len(post_idx) > 0:
            skip_idx = post_idx[:EXCLUDE_POST_STEP_SAMPLES]
            mask[skip_idx] = False

    return mask


def build_step_residual_weights(t: np.ndarray) -> np.ndarray:
    t = _ensure_1d_array(t)
    w = np.ones_like(t, dtype=np.float64)

    if SYNTHETIC_MODE != "step":
        return w

    early_mask = (t >= EARLY_STEP_WINDOW[0]) & (t <= EARLY_STEP_WINDOW[1])
    mid_mask = (t > MID_STEP_WINDOW[0]) & (t <= MID_STEP_WINDOW[1])
    tail_mask = (t > TAIL_STEP_WINDOW[0]) & (t <= TAIL_STEP_WINDOW[1])

    w[early_mask] *= STEP_WEIGHT_BOOST
    w[mid_mask] *= MID_WEIGHT_BOOST
    w[tail_mask] *= TAIL_WEIGHT_BOOST

    pre_mask = t < T_STEP
    w[pre_mask] *= 1.0

    w = w / max(float(np.mean(w)), 1e-12)
    return w


def downsample_id_source(
    t: np.ndarray,
    u: np.ndarray,
    y_abs: np.ndarray,
    y_modeling: np.ndarray,
    target_dt: float,
    use_interp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample an identification source while preserving a strictly
    uniform time grid when target_dt > native_dt.

    IMPORTANT:
    - The nonlinear pipeline later calls control.forced_response(...),
      which requires equally spaced time samples.
    - Therefore we must NOT append an off-grid final time point.
    """
    t = _ensure_1d_array(t, dtype=np.float64)
    u = _ensure_1d_array(u, dtype=np.float64)
    y_abs = _ensure_1d_array(y_abs, dtype=np.float64)
    y_modeling = _ensure_1d_array(y_modeling, dtype=np.float64)

    if len(t) < 2:
        return t.copy(), u.copy(), y_abs.copy(), y_modeling.copy()

    dt_all = np.diff(t)
    native_dt = float(np.median(dt_all))

    if not np.isfinite(target_dt) or target_dt <= native_dt + 1e-12:
        return t.copy(), u.copy(), y_abs.copy(), y_modeling.copy()

    # -----------------------------------------------------
    # Option 1: interpolation to an exactly uniform coarse grid
    # -----------------------------------------------------
    if use_interp:
        t0 = float(t[0])
        tf = float(t[-1])

        n_steps = int(np.floor((tf - t0) / target_dt))
        t_new = t0 + target_dt * np.arange(n_steps + 1, dtype=np.float64)

        u_new = np.interp(t_new, t, u)
        y_abs_new = np.interp(t_new, t, y_abs)
        y_modeling_new = np.interp(t_new, t, y_modeling)

        return t_new, u_new, y_abs_new, y_modeling_new

    # -----------------------------------------------------
    # Option 2: decimation by integer stride, but keep only
    # strictly uniform samples on the stride grid
    # -----------------------------------------------------
    step = max(int(round(target_dt / native_dt)), 1)
    idx = np.arange(0, len(t), step, dtype=int)

    # DO NOT append the final point if it breaks uniform spacing
    t_new = t[idx].copy()
    u_new = u[idx].copy()
    y_abs_new = y_abs[idx].copy()
    y_modeling_new = y_modeling[idx].copy()

    # If rounding/noise in the original time vector makes the decimated
    # grid slightly imperfect, rebuild an exact uniform grid using interp.
    if len(t_new) >= 3:
        dt_new = np.diff(t_new)
        if not np.allclose(dt_new, dt_new[0], rtol=0.0, atol=1e-10):
            t0 = float(t_new[0])
            tf = float(t_new[-1])
            target_dt_eff = float(np.median(dt_new))

            n_steps = int(np.floor((tf - t0) / target_dt_eff))
            t_uniform = t0 + target_dt_eff * np.arange(n_steps + 1, dtype=np.float64)

            u_uniform = np.interp(t_uniform, t, u)
            y_abs_uniform = np.interp(t_uniform, t, y_abs)
            y_model_uniform = np.interp(t_uniform, t, y_modeling)

            return t_uniform, u_uniform, y_abs_uniform, y_model_uniform

    return t_new, u_new, y_abs_new, y_modeling_new

# %% =====================================================
# CELL 3 — Plotting helpers (merged old + new)
# =====================================================
def plot_voltage(t, y, yh=None, title="Voltage"):
    t = _ensure_1d_array(t)
    y = _ensure_1d_array(y)

    plt.figure(figsize=(11, 4))
    plt.plot(t, y, linewidth=2, label="Truth / measured")
    if yh is not None:
        plt.plot(t, _ensure_1d_array(yh), "--", linewidth=2, label="Model")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_current_and_voltage(t, u, y, title="Current and voltage"):
    t = _ensure_1d_array(t)
    u = _ensure_1d_array(u)
    y = _ensure_1d_array(y)

    plt.figure(figsize=(11, 6))

    plt.subplot(2, 1, 1)
    plt.plot(t, u, linewidth=2)
    plt.grid(True)
    plt.ylabel("Current [A]")
    plt.title(title)

    plt.subplot(2, 1, 2)
    plt.plot(t, y, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")

    plt.tight_layout()
    plt.show()


def plot_residuals(t, y, yh, title="Residuals"):
    t = _ensure_1d_array(t)
    e = _ensure_1d_array(yh) - _ensure_1d_array(y)

    plt.figure(figsize=(11, 4))
    plt.plot(t, e, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Pred - truth [V]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_zoom_voltage(t, y, yh=None, xlim=(0.0, 50.0), title="Voltage zoom"):
    t = _ensure_1d_array(t)
    y = _ensure_1d_array(y)
    mask = (t >= xlim[0]) & (t <= xlim[1])

    plt.figure(figsize=(11, 4))
    plt.plot(t[mask], y[mask], label="Truth / measured", linewidth=2)
    if yh is not None:
        plt.plot(t[mask], _ensure_1d_array(yh)[mask], "--", label="Model")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_component_series(t, series, title, ylabel="Voltage [V]"):
    t = _ensure_1d_array(t)
    s = _ensure_1d_array(series)

    plt.figure(figsize=(11, 4))
    plt.plot(t, s, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_multi_series(t, series_dict: Dict[str, np.ndarray], title: str, ylabel: str = "Value"):
    plt.figure(figsize=(11, 5))
    for label, values in series_dict.items():
        plt.plot(_ensure_1d_array(t), _ensure_1d_array(values), label=label, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_two_panel_series(
    t,
    top_series: Dict[str, np.ndarray],
    bottom_series: Dict[str, np.ndarray],
    top_title: str,
    bottom_title: str,
    top_ylabel: str = "Value",
    bottom_ylabel: str = "Value",
):
    t = _ensure_1d_array(t)

    plt.figure(figsize=(11, 8))

    plt.subplot(2, 1, 1)
    for label, values in top_series.items():
        plt.plot(t, _ensure_1d_array(values), label=label, linewidth=2)
    plt.grid(True)
    plt.ylabel(top_ylabel)
    plt.title(top_title)
    plt.legend(loc="upper right")

    plt.subplot(2, 1, 2)
    for label, values in bottom_series.items():
        plt.plot(t, _ensure_1d_array(values), label=label, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel(bottom_ylabel)
    plt.title(bottom_title)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_voltage_component_decomposition(
    t,
    v_ocv,
    v_eta,
    v_elyte,
    v_ohmic,
    title="Voltage-component decomposition"
):
    t = _ensure_1d_array(t)
    v_ocv = _ensure_1d_array(v_ocv)
    v_eta = _ensure_1d_array(v_eta)
    v_elyte = _ensure_1d_array(v_elyte)
    v_ohmic = _ensure_1d_array(v_ohmic)

    plt.figure(figsize=(11, 5))
    plt.plot(t, v_ocv, label="OCV")
    plt.plot(t, v_eta, label="Kinetic overpotential")
    plt.plot(t, v_elyte, label="Electrolyte term")
    plt.plot(t, v_ohmic, label="Ohmic + film")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage contribution [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_voltage_component_decomposition_centered(
    t,
    v_ocv,
    v_eta,
    v_elyte,
    v_ohmic,
    title="Voltage components [centered]"
):
    t = _ensure_1d_array(t)
    v_ocv = _ensure_1d_array(v_ocv)
    v_eta = _ensure_1d_array(v_eta)
    v_elyte = _ensure_1d_array(v_elyte)
    v_ohmic = _ensure_1d_array(v_ohmic)

    v_ocv_c = v_ocv - v_ocv[0]
    v_eta_c = v_eta - v_eta[0]
    v_elyte_c = v_elyte - v_elyte[0]
    v_ohmic_c = v_ohmic - v_ohmic[0]

    plt.figure(figsize=(11, 5))
    plt.plot(t, v_ocv_c, linewidth=2, label="OCV")
    plt.plot(t, v_eta_c, linewidth=2, label="Kinetic")
    plt.plot(t, v_elyte_c, linewidth=2, label="Electrolyte")
    plt.plot(t, v_ohmic_c, linewidth=2, label="Ohmic")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Centered contribution [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_voltage_component_decomposition_normalized(
    t,
    v_ocv,
    v_eta,
    v_elyte,
    v_ohmic,
    title="Voltage components [normalized]"
):
    t = _ensure_1d_array(t)
    v_ocv = _ensure_1d_array(v_ocv)
    v_eta = _ensure_1d_array(v_eta)
    v_elyte = _ensure_1d_array(v_elyte)
    v_ohmic = _ensure_1d_array(v_ohmic)

    v_ocv_c = v_ocv - v_ocv[0]
    v_eta_c = v_eta - v_eta[0]
    v_elyte_c = v_elyte - v_elyte[0]
    v_ohmic_c = v_ohmic - v_ohmic[0]

    plt.figure(figsize=(11, 5))
    plt.plot(t, normalize_series_signed(v_ocv_c), linewidth=2, label="OCV")
    plt.plot(t, normalize_series_signed(v_eta_c), linewidth=2, label="Kinetic")
    plt.plot(t, normalize_series_signed(v_elyte_c), linewidth=2, label="Electrolyte")
    plt.plot(t, normalize_series_signed(v_ohmic_c), linewidth=2, label="Ohmic")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized contribution")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_pole_zero_map(
    pole_sets: Dict[str, np.ndarray],
    zero_sets: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Pole-zero map"
):
    plt.figure(figsize=(7.0, 7.0))
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    plt.plot(np.cos(th), np.sin(th), "k:", alpha=0.6, label="Unit circle")

    real_vals = []
    imag_vals = []

    for label, poles in pole_sets.items():
        p = np.asarray(poles, dtype=np.complex128).reshape(-1)
        p = p[np.isfinite(np.real(p)) & np.isfinite(np.imag(p))]
        if p.size:
            real_vals.extend(np.real(p).tolist())
            imag_vals.extend(np.imag(p).tolist())
            plt.scatter(
                np.real(p),
                np.imag(p),
                marker="x",
                s=80,
                linewidths=2.0,
                label=f"{label} poles",
            )

    if zero_sets is not None:
        for label, zeros in zero_sets.items():
            z = np.asarray(zeros, dtype=np.complex128).reshape(-1)
            z = z[np.isfinite(np.real(z)) & np.isfinite(np.imag(z))]
            if z.size:
                real_vals.extend(np.real(z).tolist())
                imag_vals.extend(np.imag(z).tolist())
                plt.scatter(
                    np.real(z),
                    np.imag(z),
                    facecolors="none",
                    edgecolors="C1",
                    marker="o",
                    s=80,
                    linewidths=2.0,
                    label=f"{label} zeros",
                )

    if real_vals and imag_vals:
        lim = 1.15 * max(1.0, np.max(np.abs(real_vals)), np.max(np.abs(imag_vals)))
    else:
        lim = 1.2

    plt.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
    plt.axvline(0.0, color="k", linewidth=0.8, alpha=0.6)
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_zero_magnitudes(zeros: np.ndarray, title="Zero magnitudes"):
    z = np.asarray(zeros, dtype=np.complex128).reshape(-1)
    z = z[np.isfinite(np.real(z)) & np.isfinite(np.imag(z))]
    if z.size == 0:
        print(f"{title}: no finite zeros to plot.")
        return

    idx = np.arange(len(z))
    plt.figure(figsize=(8, 4))
    plt.plot(idx, np.abs(z), marker="o")
    plt.grid(True)
    plt.xlabel("Zero index")
    plt.ylabel("|z|")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pole_magnitudes(poles, title="Pole magnitudes"):
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)

    if poles.size == 0:
        print(f"{title}: no poles to plot.")
        return

    mags = np.abs(poles)
    idx = np.arange(len(mags))

    plt.figure(figsize=(8, 4.5))
    plt.plot(idx, mags, "o")
    plt.axhline(1.0, linestyle="--", linewidth=1, label="Unit circle")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Pole index")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_matched_poles(truth_poles, est_poles, title="Matched poles"):
    truth_poles = np.asarray(truth_poles, dtype=np.complex128).reshape(-1)
    est_poles = np.asarray(est_poles, dtype=np.complex128).reshape(-1)

    matches = []
    used = set()
    for tp in truth_poles:
        best_j = None
        best_d = np.inf
        for j, ep in enumerate(est_poles):
            if j in used:
                continue
            d = abs(tp - ep)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is not None:
            used.add(best_j)
            matches.append((tp, est_poles[best_j]))

    if not matches:
        print(f"{title}: no matched poles to plot.")
        return

    truth_abs = np.array([abs(tp) for tp, _ in matches], dtype=np.float64)
    est_abs = np.array([abs(ep) for _, ep in matches], dtype=np.float64)
    idx = np.arange(len(matches))

    plt.figure(figsize=(8, 4))
    plt.plot(idx, truth_abs, marker="o", label="Truth |p|")
    plt.plot(idx, est_abs, marker="o", label="Estimated |p|")
    plt.xlabel("Matched pole index")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_candidate_overlay(t, y, candidates: list[dict], title: str, max_models: int = 5):
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, y, linewidth=2, label="Truth / measured")
    for cand in candidates[:max_models]:
        name = f"nb={cand['n'][0]}"
        if "yhat_unscaled" in cand:
            plt.plot(t, cand["yhat_unscaled"], "--", alpha=0.85, label=name)
        elif "yhat_abs_full" in cand:
            plt.plot(t, cand["yhat_abs_full"], "--", alpha=0.85, label=name)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_residual_diagnostics(name: str, residual: np.ndarray, input_signal: np.ndarray, max_lag: int = 50):
    tests = summarize_residual_tests(name, residual, input_signal, max_lag=max_lag)

    plt.figure(figsize=(11, 4.5))

    plt.subplot(1, 2, 1)
    plt.stem(
        tests["lags_acf"][1:],
        tests["acf"][1:],
        linefmt="C0-",
        markerfmt="C0o",
        basefmt="k-"
    )
    plt.axhline(tests["conf"], linestyle=":")
    plt.axhline(-tests["conf"], linestyle=":")
    plt.xlabel("Lag")
    plt.ylabel("Residual ACF")
    plt.title(f"{name} residual autocorrelation")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.stem(
        tests["lags_ccf"],
        tests["ccf"],
        linefmt="C1-",
        markerfmt="C1o",
        basefmt="k-"
    )
    plt.axhline(tests["conf"], linestyle=":")
    plt.axhline(-tests["conf"], linestyle=":")
    plt.xlabel("Lag")
    plt.ylabel("Corr(residual, input)")
    plt.title(f"{name} residual-input correlation")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_side_by_side_current_voltage(
    t_left,
    u_left,
    y_left,
    t_right,
    u_right,
    y_right,
    left_title="Real",
    right_title="Synthetic",
    overall_title="Real vs synthetic"
):
    t_left = _ensure_1d_array(t_left)
    u_left = _ensure_1d_array(u_left)
    y_left = _ensure_1d_array(y_left)

    t_right = _ensure_1d_array(t_right)
    u_right = _ensure_1d_array(u_right)
    y_right = _ensure_1d_array(y_right)

    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex="col")

    axs[0, 0].plot(t_left, u_left, linewidth=2)
    axs[0, 0].grid(True)
    axs[0, 0].set_title(f"{left_title} current")
    axs[0, 0].set_ylabel("Current [A]")

    axs[1, 0].plot(t_left, y_left, linewidth=2)
    axs[1, 0].grid(True)
    axs[1, 0].set_title(f"{left_title} voltage")
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel("Voltage [V]")

    axs[0, 1].plot(t_right, u_right, linewidth=2)
    axs[0, 1].grid(True)
    axs[0, 1].set_title(f"{right_title} current")
    axs[0, 1].set_ylabel("Current [A]")

    axs[1, 1].plot(t_right, y_right, linewidth=2)
    axs[1, 1].grid(True)
    axs[1, 1].set_title(f"{right_title} voltage")
    axs[1, 1].set_xlabel("Time [s]")
    axs[1, 1].set_ylabel("Voltage [V]")

    fig.suptitle(overall_title)
    fig.tight_layout()
    plt.show()


def plot_overlay_real_vs_synthetic(
    t_real,
    y_real,
    t_syn,
    y_syn,
    title="Real vs synthetic overlay",
    label_real="Real",
    label_syn="Synthetic"
):
    t_real = _ensure_1d_array(t_real)
    y_real = _ensure_1d_array(y_real)
    t_syn = _ensure_1d_array(t_syn)
    y_syn = _ensure_1d_array(y_syn)

    plt.figure(figsize=(11, 4))
    plt.plot(t_real, y_real, linewidth=2, label=label_real)
    plt.plot(t_syn, y_syn, linewidth=2, label=label_syn)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_overlay_normalized_real_vs_synthetic(
    t_real,
    y_real,
    t_syn,
    y_syn,
    title="Normalized real vs synthetic overlay"
):
    yr = normalize_series_signed(y_real)
    ys = normalize_series_signed(y_syn)

    plt.figure(figsize=(11, 4))
    plt.plot(_ensure_1d_array(t_real), yr, linewidth=2, label="Real normalized")
    plt.plot(_ensure_1d_array(t_syn), ys, linewidth=2, label="Synthetic normalized")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized voltage")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def shift_series_to_match_start(y_src: np.ndarray, y_ref: np.ndarray) -> np.ndarray:
    y_src = _ensure_1d_array(y_src).astype(np.float64)
    y_ref = _ensure_1d_array(y_ref).astype(np.float64)
    if len(y_src) == 0 or len(y_ref) == 0:
        return y_src
    return y_src + (float(y_ref[0]) - float(y_src[0]))


def affine_align_series_to_reference(y_src: np.ndarray, y_ref: np.ndarray) -> tuple[np.ndarray, float, float]:
    y_src = _ensure_1d_array(y_src).astype(np.float64)
    y_ref = _ensure_1d_array(y_ref).astype(np.float64)

    if len(y_src) == 0 or len(y_ref) == 0:
        return y_src.copy(), 1.0, 0.0

    src0 = float(y_src[0])
    ref0 = float(y_ref[0])

    src_dev = y_src - src0
    ref_dev = y_ref - ref0

    src_span = float(np.max(src_dev) - np.min(src_dev))
    ref_span = float(np.max(ref_dev) - np.min(ref_dev))

    if abs(src_span) < 1e-15:
        a = 1.0
    else:
        a = ref_span / src_span

    y_aligned = a * src_dev + ref0
    b = ref0 - a * src0
    return y_aligned, float(a), float(b)


def plot_alignment_comparison(
    t_real,
    y_real,
    t_syn,
    y_syn_raw,
    y_syn_shifted,
    y_syn_affine,
    title="Real vs synthetic voltage alignment"
):
    t_real = _ensure_1d_array(t_real)
    y_real = _ensure_1d_array(y_real)
    t_syn = _ensure_1d_array(t_syn)
    y_syn_raw = _ensure_1d_array(y_syn_raw)
    y_syn_shifted = _ensure_1d_array(y_syn_shifted)
    y_syn_affine = _ensure_1d_array(y_syn_affine)

    plt.figure(figsize=(12, 5))
    plt.plot(t_real, y_real, linewidth=2.5, label="Real")
    plt.plot(t_syn, y_syn_raw, linewidth=1.5, label="Synthetic raw")
    plt.plot(t_syn, y_syn_shifted, linewidth=2.0, label="Synthetic shifted-to-start")
    plt.plot(t_syn, y_syn_affine, linewidth=2.0, label="Synthetic affine-aligned")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_pole_magnitudes(poles, title="Pole magnitudes"):
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)

    if poles.size == 0:
        print(f"{title}: no poles to plot.")
        return

    mags = np.abs(poles)
    idx = np.arange(len(mags))

    plt.figure(figsize=(8, 4.5))
    plt.plot(idx, mags, "o")
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Pole index")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# Normalized real-vs-synthetic voltage alignment plot
# Place this immediately after plot_alignment_comparison(...)
# and before if PLOT_COMPARISON_WITH_SHIFTED_VOLTAGE:
# ---------------------------------------------------------
def normalize_for_alignment_plot(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y) == 0:
        return y.copy()
    y_shift = y - y[0]
    span = float(np.max(y_shift) - np.min(y_shift))
    if abs(span) < 1e-12:
        denom = max(float(np.max(np.abs(y_shift))), 1.0)
    else:
        denom = span
    return y_shift / denom

# %% =====================================================
# CELL 4 — State-space truth variants
# Supported variants:
#   - reduced_7
#   - full_14
# =====================================================
@dataclass
class Config:
    R: float = 8.314462618
    F: float = 96485.33212
    T: float = 298.15

    L1: float = 25e-6
    L2: float = 20e-6
    L3: float = 25e-6

    Rn: float = 5e-6
    Rp: float = 5e-6
    A: float = 1.0

    Dn: float = 1e-14
    Dp: float = 1e-14
    De: float = 7.23e-10
    eps: float = 0.30

    kappa_n_eff: float = 1.0
    kappa_s_eff: float = 1.0
    kappa_p_eff: float = 1.0

    a_s_n: float = 1.0e6
    a_s_p: float = 1.0e6

    k_n0: float = 3.0e-11
    k_p0: float = 1.4e-11

    csn_max: float = 3.1e4
    csp_max: float = 3.1e4
    ce0: float = 1000.0
    t_plus: float = 0.38

    k_f: float = 1.0

    R_ohm: float = 0.0
    Rf: float = 0.0

    ce_is_deviation: bool = True
    discharge_positive: bool = DISCHARGE_POSITIVE
    eta_mode: str = "sum"

    theta_guard: float = 1e-3
    ce_guard: float = 1e-6

    I0_floor_p: float = 3.0
    I0_floor_n: float = 2.2

    bv_scale: float = 0.7
    N_series: int = N_SERIES

    use_solid_stoich_rate_scale: bool = USE_SOLID_STOICH_RATE_SCALE
    solid_stoich_rate_scale: float = SOLID_STOICH_RATE_SCALE

    use_ocv_scale: bool = USE_OCV_SCALE
    ocv_scale: float = OCV_SCALE
    ocv_model_mode: str = OCV_MODEL_MODE

    ocv_weight: float = 1.0
    eta_weight: float = 1.0
    electrolyte_weight: float = 1.0
    ohmic_weight: float = 1.0

    use_kinetic_term: bool = USE_KINETIC_TERM
    use_electrolyte_term: bool = USE_ELECTROLYTE_TERM
    use_ohmic_term: bool = USE_OHMIC_TERM

    use_eta_p: bool = USE_ETA_P
    use_eta_n: bool = USE_ETA_N

    use_electrolyte_log_term: bool = USE_ELECTROLYTE_LOG_TERM
    use_electrolyte_ohmic_resistance: bool = USE_ELECTROLYTE_OHMIC_RESISTANCE

    use_pure_R_ohm: bool = USE_PURE_R_OHM
    use_film_resistance: bool = USE_FILM_RESISTANCE

    electrolyte_log_blend: float = ELECTROLYTE_LOG_BLEND
    kinetic_blend: float = KINETIC_BLEND
    ohmic_blend: float = OHMIC_BLEND

    exchange_current_scale_p: float = EXCHANGE_CURRENT_SCALE_P
    exchange_current_scale_n: float = EXCHANGE_CURRENT_SCALE_N

    electrolyte_state_orientation: str = ELECTROLYTE_STATE_ORIENTATION
    ln_orientation: str = ELECTROLYTE_LOG_ORIENTATION
    use_display_electrolyte_flip: bool = USE_DISPLAY_ELECTROLYTE_FLIP


cfg = Config()

if cfg.electrolyte_state_orientation not in ("default", "flipped"):
    raise ValueError(
        f"Unsupported ELECTROLYTE_STATE_ORIENTATION={cfg.electrolyte_state_orientation!r}. "
        f"Use 'default' or 'flipped'."
    )

if cfg.ln_orientation not in ("right_over_left", "left_over_right"):
    raise ValueError(
        f"Unsupported ELECTROLYTE_LOG_ORIENTATION={cfg.ln_orientation!r}. "
        f"Use 'right_over_left' or 'left_over_right'."
    )

if EXPERIMENT_PRESET == "visual_ocv":
    TRUTH_VOLTAGE_MODE = "custom"
    cfg.ocv_weight = 1.00
    cfg.eta_weight = 0.05
    cfg.electrolyte_weight = 0.05
    cfg.ohmic_weight = 0.05
    cfg.bv_scale = 0.08
    cfg.R_ohm = 0.0001
    cfg.Rf = 0.00005
    cfg.kappa_s_eff = 1.0

elif EXPERIMENT_PRESET == "pole_id":
    TRUTH_VOLTAGE_MODE = "balanced"
    cfg.ocv_weight = POLE_ID_OCV_WEIGHT
    cfg.eta_weight = POLE_ID_ETA_WEIGHT
    cfg.electrolyte_weight = POLE_ID_ELECTROLYTE_WEIGHT
    cfg.ohmic_weight = POLE_ID_OHMIC_WEIGHT
    cfg.bv_scale = POLE_ID_BV_SCALE
    cfg.R_ohm = POLE_ID_R_OHM_VALUE
    cfg.Rf = POLE_ID_RF_VALUE
    cfg.kappa_s_eff = POLE_ID_KAPPA_S_EFF_VALUE

else:
    raise ValueError(f"Unsupported EXPERIMENT_PRESET={EXPERIMENT_PRESET!r}")

print("Truth variant:", STATE_SPACE_VARIANT)
print("Voltage balance mode:", TRUTH_VOLTAGE_MODE)
print(
    "Weights:",
    {
        "ocv": cfg.ocv_weight,
        "eta": cfg.eta_weight,
        "electrolyte": cfg.electrolyte_weight,
        "ohmic": cfg.ohmic_weight,
    },
)
print(
    "R_ohm:", cfg.R_ohm,
    "Rf:", cfg.Rf,
    "kappa_s_eff:", cfg.kappa_s_eff,
    "bv_scale:", cfg.bv_scale,
)
print("ocv_model_mode:", cfg.ocv_model_mode)
print("use_ocv_scale:", cfg.use_ocv_scale, "ocv_scale:", cfg.ocv_scale)
print(
    "use_solid_stoich_rate_scale:", cfg.use_solid_stoich_rate_scale,
    "solid_stoich_rate_scale:", cfg.solid_stoich_rate_scale,
)

VARIANT_IDXS: dict[str, dict[str, Any]] = {
    "reduced_7": {
        "cn": slice(0, 2),
        "cn_surf": 1,
        "cp": slice(2, 4),
        "cp_surf": 3,
        "ce_left": 4,
        "ce_mid": 5,
        "ce_right": 6,
        "n_states": 7,
    },
    "full_14": {
        "cn": slice(0, 4),
        "cn_surf": 3,
        "cp": slice(4, 8),
        "cp_surf": 7,
        "ce": slice(8, 14),
        "ce_left": 8,
        "ce_mid": 10,
        "ce_right": 13,
        "n_states": 14,
    },
}


def get_variant_idx(variant: str) -> dict[str, Any]:
    if variant not in VARIANT_IDXS:
        raise ValueError(
            f"Unsupported STATE_SPACE_VARIANT={variant!r}. "
            f"Supported: {list(VARIANT_IDXS.keys())}"
        )
    return VARIANT_IDXS[variant]


IDX = get_variant_idx(STATE_SPACE_VARIANT)


def _solid_scale(cfg: Config) -> float:
    return float(cfg.solid_stoich_rate_scale) if cfg.use_solid_stoich_rate_scale else 1.0


# ---------------------------------------------------------
# reduced_7 builders
# ---------------------------------------------------------
def build_An_reduced7(cfg: Config) -> np.ndarray:
    s = cfg.Dn / (cfg.Rn ** 2)
    return np.array(
        [[-8.0 * s, 8.0 * s],
         [8.0 * s, -8.0 * s]],
        dtype=np.float64,
    )


def build_Bn_reduced7(cfg: Config) -> np.ndarray:
    sign = -1.0 if cfg.discharge_positive else +1.0
    g = sign * (1.0 / cfg.Rn) * (1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1))
    B = np.zeros((2, 1), dtype=np.float64)
    B[1, 0] = _solid_scale(cfg) * 4.0 * g
    return B


def build_Ap_reduced7(cfg: Config) -> np.ndarray:
    s = cfg.Dp / (cfg.Rp ** 2)
    return np.array(
        [[-8.0 * s, 8.0 * s],
         [8.0 * s, -8.0 * s]],
        dtype=np.float64,
    )


def build_Bp_reduced7(cfg: Config) -> np.ndarray:
    sign = +1.0 if cfg.discharge_positive else -1.0
    g = sign * (1.0 / cfg.Rp) * (1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3))
    B = np.zeros((2, 1), dtype=np.float64)
    B[1, 0] = _solid_scale(cfg) * 4.0 * g
    return B


def build_Ae_reduced7(cfg: Config) -> np.ndarray:
    K = cfg.De / cfg.eps
    k12 = K * 4.0 / ((cfg.L1 + cfg.L2) ** 2)
    k23 = K * 4.0 / ((cfg.L2 + cfg.L3) ** 2)

    A = np.zeros((3, 3), dtype=np.float64)
    A[0, 0] = -k12
    A[0, 1] = +k12
    A[1, 0] = +k12
    A[1, 1] = -(k12 + k23)
    A[1, 2] = +k23
    A[2, 1] = +k23
    A[2, 2] = -k23
    return A


def build_Be_reduced7(cfg: Config) -> np.ndarray:
    if cfg.electrolyte_state_orientation == "default":
        sign_left = -1.0 if cfg.discharge_positive else +1.0
        sign_right = +1.0 if cfg.discharge_positive else -1.0
    elif cfg.electrolyte_state_orientation == "flipped":
        sign_left = +1.0 if cfg.discharge_positive else -1.0
        sign_right = -1.0 if cfg.discharge_positive else +1.0
    else:
        raise ValueError(
            f"Unsupported electrolyte_state_orientation={cfg.electrolyte_state_orientation!r}"
        )

    B = np.zeros((3, 1), dtype=np.float64)

    s1 = sign_left * (1.0 - cfg.t_plus) / (
        cfg.eps * 0.5 * cfg.F * cfg.A * (cfg.L1 + cfg.L2)
    )
    s3 = sign_right * (1.0 - cfg.t_plus) / (
        cfg.eps * 0.5 * cfg.F * cfg.A * (cfg.L2 + cfg.L3)
    )

    B[0, 0] = s1
    B[2, 0] = s3
    return B


def assemble_reduced7_system(cfg: Config):
    A = block_diag(
        build_An_reduced7(cfg),
        build_Ap_reduced7(cfg),
        build_Ae_reduced7(cfg),
    )
    B = np.vstack([
        build_Bn_reduced7(cfg),
        build_Bp_reduced7(cfg),
        build_Be_reduced7(cfg),
    ])
    C = np.eye(7)
    D = np.zeros((7, 1))
    S = ct.ss(A, B, C, D)
    return S, A, B


def make_x0_reduced7(cfg: Config, theta_n0=0.8, theta_p0=0.4, ce0=0.0):
    x0 = np.zeros(7, dtype=np.float64)
    x0[VARIANT_IDXS["reduced_7"]["cn"]] = float(theta_n0) * cfg.csn_max
    x0[VARIANT_IDXS["reduced_7"]["cp"]] = float(theta_p0) * cfg.csp_max
    x0[VARIANT_IDXS["reduced_7"]["ce_left"]] = float(ce0)
    x0[VARIANT_IDXS["reduced_7"]["ce_mid"]] = float(ce0)
    x0[VARIANT_IDXS["reduced_7"]["ce_right"]] = float(ce0)
    return x0


# ---------------------------------------------------------
# full_14 builders
# ---------------------------------------------------------
def build_An_full14(cfg: Config) -> np.ndarray:
    s = cfg.Dn / (cfg.Rn ** 2)
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0], A[0, 1] = -24.0 * s, 24.0 * s
    A[1, 0], A[1, 1], A[1, 2] = 16.0 * s, -40.0 * s, 24.0 * s
    A[2, 1], A[2, 2], A[2, 3] = 16.0 * s, -40.0 * s, 24.0 * s
    A[3, 2], A[3, 3] = 16.0 * s, -16.0 * s
    return A


def build_Bn_full14(cfg: Config) -> np.ndarray:
    sign = -1.0 if cfg.discharge_positive else +1.0
    b = np.zeros((4, 1), dtype=np.float64)
    b[-1, 0] = (
        _solid_scale(cfg)
        * sign
        * (6.0 / cfg.Rn)
        * (1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1))
    )
    return b


def build_Ap_full14(cfg: Config) -> np.ndarray:
    s = cfg.Dp / (cfg.Rp ** 2)
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0], A[0, 1] = -24.0 * s, 24.0 * s
    A[1, 0], A[1, 1], A[1, 2] = 16.0 * s, -40.0 * s, 24.0 * s
    A[2, 1], A[2, 2], A[2, 3] = 16.0 * s, -40.0 * s, 24.0 * s
    A[3, 2], A[3, 3] = 16.0 * s, -16.0 * s
    return A


def build_Bp_full14(cfg: Config) -> np.ndarray:
    sign = +1.0 if cfg.discharge_positive else -1.0
    b = np.zeros((4, 1), dtype=np.float64)
    b[-1, 0] = (
        _solid_scale(cfg)
        * sign
        * (6.0 / cfg.Rp)
        * (1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3))
    )
    return b


def build_Ae_full14(cfg: Config) -> np.ndarray:
    K = cfg.De / cfg.eps
    Ae = np.zeros((6, 6), dtype=np.float64)

    def w_in(L):
        return K * 4.0 / (L ** 2)

    def w_intf(La, Lb):
        return K * 16.0 / ((La + Lb) ** 2)

    w11 = w_in(cfg.L1)
    w12 = w_intf(cfg.L1, cfg.L2)
    w23 = w_in(cfg.L2)
    w34 = w_intf(cfg.L2, cfg.L3)
    w45 = w_in(cfg.L3)

    Ae[0, 0] = -w11
    Ae[0, 1] = +w11

    Ae[1, 0] = +w11
    Ae[1, 1] = -(w11 + w12)
    Ae[1, 2] = +w12

    Ae[2, 1] = +w12
    Ae[2, 2] = -(w12 + w23)
    Ae[2, 3] = +w23

    Ae[3, 2] = +w23
    Ae[3, 3] = -(w23 + w34)
    Ae[3, 4] = +w34

    Ae[4, 3] = +w34
    Ae[4, 4] = -(w34 + w45)
    Ae[4, 5] = +w45

    Ae[5, 4] = +w45
    Ae[5, 5] = -w45

    return Ae


def build_Be_full14(cfg: Config) -> np.ndarray:
    b = np.zeros((6, 1), dtype=np.float64)

    if cfg.electrolyte_state_orientation == "default":
        sign_left = -1.0 if cfg.discharge_positive else +1.0
        sign_right = +1.0 if cfg.discharge_positive else -1.0
    elif cfg.electrolyte_state_orientation == "flipped":
        sign_left = +1.0 if cfg.discharge_positive else -1.0
        sign_right = -1.0 if cfg.discharge_positive else +1.0
    else:
        raise ValueError(
            f"Unsupported electrolyte_state_orientation={cfg.electrolyte_state_orientation!r}"
        )

    s1 = sign_left * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L1 * cfg.eps)
    s3 = sign_right * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L3 * cfg.eps)

    b[0, 0] = s1
    b[1, 0] = s1
    b[4, 0] = s3
    b[5, 0] = s3

    return b


def assemble_full14_system(cfg: Config):
    A = block_diag(
        build_An_full14(cfg),
        build_Ap_full14(cfg),
        build_Ae_full14(cfg),
    )
    B = np.vstack([
        build_Bn_full14(cfg),
        build_Bp_full14(cfg),
        build_Be_full14(cfg),
    ])
    C = np.eye(14)
    D = np.zeros((14, 1))
    S = ct.ss(A, B, C, D)
    return S, A, B


def make_x0_full14(cfg: Config, theta_n0=0.8, theta_p0=0.4, ce0=0.0):
    x0 = np.zeros(14, dtype=np.float64)
    x0[VARIANT_IDXS["full_14"]["cn"]] = float(theta_n0) * cfg.csn_max
    x0[VARIANT_IDXS["full_14"]["cp"]] = float(theta_p0) * cfg.csp_max
    x0[VARIANT_IDXS["full_14"]["ce"]] = float(ce0)
    return x0


def assemble_truth_system(cfg: Config, variant: str):
    if variant == "reduced_7":
        return assemble_reduced7_system(cfg)
    if variant == "full_14":
        return assemble_full14_system(cfg)
    raise ValueError(
        f"Unsupported STATE_SPACE_VARIANT={variant!r}. "
        f"Supported: ['reduced_7', 'full_14']"
    )


def make_x0_truth(cfg: Config, variant: str, theta_n0=0.8, theta_p0=0.4, ce0=0.0):
    if variant == "reduced_7":
        return make_x0_reduced7(cfg, theta_n0=theta_n0, theta_p0=theta_p0, ce0=ce0)
    if variant == "full_14":
        return make_x0_full14(cfg, theta_n0=theta_n0, theta_p0=theta_p0, ce0=ce0)
    raise ValueError(
        f"Unsupported STATE_SPACE_VARIANT={variant!r}. "
        f"Supported: ['reduced_7', 'full_14']"
    )


S_truth, A_truth, B_truth = assemble_truth_system(cfg, STATE_SPACE_VARIANT)

print("Truth variant:", STATE_SPACE_VARIANT)
print("A_truth shape:", A_truth.shape)
print("B_truth shape:", B_truth.shape)
print("Continuous-time eigenvalues of A_truth:")
print(np.linalg.eigvals(A_truth))
print("Voltage balance mode:", TRUTH_VOLTAGE_MODE)
print("Weights:", {
    "ocv": cfg.ocv_weight,
    "eta": cfg.eta_weight,
    "electrolyte": cfg.electrolyte_weight,
    "ohmic": cfg.ohmic_weight,
})
print("R_ohm:", cfg.R_ohm, "Rf:", cfg.Rf, "kappa_s_eff:", cfg.kappa_s_eff, "bv_scale:", cfg.bv_scale)
print("ocv_model_mode:", cfg.ocv_model_mode)
print("use_ocv_scale:", cfg.use_ocv_scale, "ocv_scale:", cfg.ocv_scale)
print("use_solid_stoich_rate_scale:", cfg.use_solid_stoich_rate_scale, "solid_stoich_rate_scale:", cfg.solid_stoich_rate_scale)


# %% =====================================================
# CELL 5 — Truth voltage model and synthetic data generator
# =====================================================
def ocp_p(xp: float) -> float:
    x = float(np.clip(xp, 1e-9, 1.0 - 1e-9))
    return float(4.15 - 0.12 * np.tanh((x - 0.60) / 0.08))


def ocp_n(xn: float) -> float:
    x = float(np.clip(xn, 1e-9, 1.0 - 1e-9))
    return float(0.10 + 0.80 * (1.0 / (1.0 + np.exp(-(x - 0.50) / 0.04))))


def solid_stoich_from_state(x: np.ndarray, cfg: Config) -> tuple[float, float]:
    xp = float(np.clip(x[IDX["cp_surf"]] / cfg.csp_max, 1e-9, 1.0 - 1e-9))
    xn = float(np.clip(x[IDX["cn_surf"]] / cfg.csn_max, 1e-9, 1.0 - 1e-9))
    return xn, xp


def extract_electrolyte_edges(x: np.ndarray, cfg: Config) -> tuple[float, float]:
    ceL_raw = float(x[IDX["ce_left"]])
    ceR_raw = float(x[IDX["ce_right"]])

    ceL = cfg.ce0 + ceL_raw if cfg.ce_is_deviation else ceL_raw
    ceR = cfg.ce0 + ceR_raw if cfg.ce_is_deviation else ceR_raw

    if cfg.use_display_electrolyte_flip:
        ceL, ceR = ceR, ceL

    return ceL, ceR


# def full_cell_soc_proxy(x: np.ndarray, cfg: Config) -> float:
#     xn, xp = solid_stoich_from_state(x, cfg)
#     soc = 0.5 * (xn + (1.0 - xp))
#     return float(np.clip(soc, 1e-9, 1.0 - 1e-9))


# def _clip_soc_01(s: float) -> float:
#     return float(np.clip(s, 1e-9, 1.0 - 1e-9))


# def _chebyshev_coeffs_from_settings() -> np.ndarray:
#     preset = str(CHEBYSHEV_OCV_PRESET).lower()

#     if preset == "default_soft_slope":
#         coeffs = CHEBYSHEV_OCV_COEFFS_DEFAULT_SOFT_SLOPE
#     elif preset == "stronger_mid_slope":
#         coeffs = CHEBYSHEV_OCV_COEFFS_STRONGER_MID_SLOPE
#     elif preset == "more_end_curvature":
#         coeffs = CHEBYSHEV_OCV_COEFFS_MORE_END_CURVATURE
#     elif preset == "manual":
#         coeffs = CHEBYSHEV_OCV_COEFFS_MANUAL
#     else:
#         raise ValueError(
#             f"Unsupported CHEBYSHEV_OCV_PRESET={CHEBYSHEV_OCV_PRESET!r}. "
#             "Use 'default_soft_slope', 'stronger_mid_slope', "
#             "'more_end_curvature', or 'manual'."
#         )

#     coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
#     if len(coeffs) == 0:
#         raise ValueError("Chebyshev coefficient list must not be empty.")
#     return coeffs


# def _chebyshev_T_series(xc: float, n_terms: int) -> np.ndarray:
#     """
#     Return [T0(xc), T1(xc), ..., T_{n_terms-1}(xc)] using recurrence.
#     xc must lie in [-1, 1] for standard Chebyshev evaluation.
#     """
#     xc = float(np.clip(xc, -1.0, 1.0))
#     if n_terms <= 0:
#         return np.zeros(0, dtype=np.float64)

#     T = np.zeros(n_terms, dtype=np.float64)
#     T[0] = 1.0
#     if n_terms >= 2:
#         T[1] = xc
#     for k in range(2, n_terms):
#         T[k] = 2.0 * xc * T[k - 1] - T[k - 2]
#     return T


# def remap_soc_for_chebyshev(soc: float) -> float:
#     """
#     Optional SOC remap to concentrate slope around a chosen SOC region.
#     When disabled, returns the original SOC.
#     """
#     s = _clip_soc_01(soc)

#     if not CHEBYSHEV_USE_SOC_REMAP:
#         return s

#     c = float(np.clip(CHEBYSHEV_SOC_REMAP_CENTER, 1e-6, 1.0 - 1e-6))
#     g = float(max(CHEBYSHEV_SOC_REMAP_GAIN, 1e-6))

#     # smooth logistic-like remap around chosen center
#     z = (s - c) / g
#     s_map = 1.0 / (1.0 + np.exp(-z))

#     # normalize so endpoints remain close to 0 and 1 over practical range
#     return _clip_soc_01(s_map)


# def full_cell_ocv_from_soc(soc: float) -> float:
#     """
#     Existing shaped SOC->OCV option retained exactly as a non-Chebyshev baseline.
#     """
#     s = _clip_soc_01(soc)
#     v = FULL_CELL_OCV_MIN + (FULL_CELL_OCV_MAX - FULL_CELL_OCV_MIN) * s
#     v += 0.05 * np.tanh((s - 0.12) / 0.05)
#     v -= 0.03 * np.tanh((s - 0.55) / 0.14)
#     v += 0.06 * np.tanh((s - 0.88) / 0.06)
#     return float(np.clip(v, FULL_CELL_OCV_MIN, FULL_CELL_OCV_MAX))


# def full_cell_ocv_from_soc_chebyshev(soc: float) -> float:
#     """
#     NEW:
#     Full-cell OCV surrogate using Chebyshev polynomials of the first kind.

#     Steps:
#       1) get SOC in [0,1]
#       2) optionally remap SOC for stronger local slope shaping
#       3) map to x_c in [-1,1]
#       4) evaluate sum c_k T_k(x_c)
#       5) optionally clip to [FULL_CELL_OCV_MIN, FULL_CELL_OCV_MAX]
#     """
#     s_raw = _clip_soc_01(soc)
#     s = remap_soc_for_chebyshev(s_raw)
#     xc = 2.0 * s - 1.0

#     coeffs = _chebyshev_coeffs_from_settings()
#     T = _chebyshev_T_series(xc, len(coeffs))
#     v = float(np.dot(coeffs, T))

#     v = float(CHEBYSHEV_OCV_GAIN * v + CHEBYSHEV_OCV_BIAS)

#     if USE_CHEBYSHEV_OCV_CLIP:
#         v = float(np.clip(v, FULL_CELL_OCV_MIN, FULL_CELL_OCV_MAX))

#     return v


# def ocv_decomposition_from_state(x: np.ndarray, cfg: Config) -> dict[str, float]:
#     xn, xp = solid_stoich_from_state(x, cfg)
#     mode = str(OCV_MODEL_MODE).lower()

#     if mode == "electrode_ocp":
#         Up_cell = ocp_p(xp)
#         Un_cell = ocp_n(xn)
#         ocv_cell = Up_cell - Un_cell

#         if USE_OCV_SCALE:
#             ocv_cell *= OCV_SCALE
#             Up_cell *= OCV_SCALE
#             Un_cell *= OCV_SCALE

#         Up_pack = cfg.N_series * cfg.ocv_weight * Up_cell
#         minus_Un_pack = -cfg.N_series * cfg.ocv_weight * Un_cell
#         ocv_pack = Up_pack + minus_Un_pack

#         return {
#             "xp": float(xp),
#             "xn": float(xn),
#             "soc": float(full_cell_soc_proxy(x, cfg)),
#             "Up_cell": float(Up_cell),
#             "Un_cell": float(Un_cell),
#             "Up_pack": float(Up_pack),
#             "minus_Un_pack": float(minus_Un_pack),
#             "ocv_cell": float(ocv_cell),
#             "ocv_pack": float(ocv_pack),
#             "ocv_mode": "electrode_ocp",
#         }

#     elif mode == "soc_proxy_shaped":
#         soc = full_cell_soc_proxy(x, cfg)
#         ocv_cell = full_cell_ocv_from_soc(soc)

#         if USE_OCV_SCALE:
#             ocv_cell *= OCV_SCALE

#         ocv_pack = cfg.N_series * cfg.ocv_weight * ocv_cell

#         return {
#             "xp": float(xp),
#             "xn": float(xn),
#             "soc": float(soc),
#             "Up_cell": np.nan,
#             "Un_cell": np.nan,
#             "Up_pack": np.nan,
#             "minus_Un_pack": np.nan,
#             "ocv_cell": float(ocv_cell),
#             "ocv_pack": float(ocv_pack),
#             "ocv_mode": "soc_proxy_shaped",
#         }

#     elif mode == "chebyshev_soc":
#         soc = full_cell_soc_proxy(x, cfg)
#         ocv_cell = full_cell_ocv_from_soc_chebyshev(soc)

#         if USE_OCV_SCALE:
#             ocv_cell *= OCV_SCALE

#         ocv_pack = cfg.N_series * cfg.ocv_weight * ocv_cell

#         return {
#             "xp": float(xp),
#             "xn": float(xn),
#             "soc": float(soc),
#             "Up_cell": np.nan,
#             "Un_cell": np.nan,
#             "Up_pack": np.nan,
#             "minus_Un_pack": np.nan,
#             "ocv_cell": float(ocv_cell),
#             "ocv_pack": float(ocv_pack),
#             "ocv_mode": "chebyshev_soc",
#         }

#     else:
#         raise ValueError(
#             f"Unsupported OCV_MODEL_MODE={OCV_MODEL_MODE!r}. "
#             "Use 'electrode_ocp', 'soc_proxy_shaped', or 'chebyshev_soc'."
#         )


# def i0_current_scales(xn: float, xp: float, ceL: float, ceR: float, cfg: Config):
#     ce_avg = float(np.clip(0.5 * (ceL + ceR), 1e-12, 10.0 * cfg.ce0))
#     xp_eff = float(np.clip(xp, cfg.theta_guard, 1.0 - cfg.theta_guard))
#     xn_eff = float(np.clip(xn, cfg.theta_guard, 1.0 - cfg.theta_guard))

#     Sp = cfg.a_s_p * cfg.A * cfg.L3
#     Sn = cfg.a_s_n * cfg.A * cfg.L1

#     i0p = (
#         cfg.exchange_current_scale_p
#         * cfg.F * cfg.k_p0 * cfg.csp_max
#         * np.sqrt(ce_avg) * np.sqrt(xp_eff * (1.0 - xp_eff))
#     )
#     i0n = (
#         cfg.exchange_current_scale_n
#         * cfg.F * cfg.k_n0 * cfg.csn_max
#         * np.sqrt(ce_avg) * np.sqrt(xn_eff * (1.0 - xn_eff))
#     )

#     return max(float(Sp * i0p), cfg.I0_floor_p), max(float(Sn * i0n), cfg.I0_floor_n)


# def electrolyte_log_term_details(ceL: float, ceR: float, cfg: Config) -> dict[str, float]:
#     ceL = max(float(ceL), 1e-12)
#     ceR = max(float(ceR), 1e-12)

#     if cfg.ln_orientation == "right_over_left":
#         arg = ceR / ceL
#     elif cfg.ln_orientation == "left_over_right":
#         arg = ceL / ceR
#     else:
#         raise ValueError(f"Unsupported ln_orientation={cfg.ln_orientation!r}")

#     ln_ratio = np.log(arg)
#     prefactor = (2.0 * cfg.R * cfg.T / cfg.F) * (1.0 - cfg.t_plus) * cfg.k_f
#     value = prefactor * ln_ratio

#     if not cfg.use_electrolyte_log_term:
#         value = 0.0

#     value *= cfg.electrolyte_log_blend

#     return {
#         "ceL": float(ceL),
#         "ceR": float(ceR),
#         "arg": float(arg),
#         "ln_ratio": float(ln_ratio),
#         "prefactor": float(prefactor),
#         "value": float(value),
#     }


# def electrolyte_resistance_parts(cfg: Config) -> dict[str, float]:
#     r_n = cfg.L1 / cfg.kappa_n_eff
#     r_s = 2.0 * cfg.L2 / cfg.kappa_s_eff
#     r_p = cfg.L3 / cfg.kappa_p_eff

#     return {
#         "negative_electrolyte_res": float(r_n / (2.0 * cfg.A)),
#         "separator_electrolyte_res": float(r_s / (2.0 * cfg.A)),
#         "positive_electrolyte_res": float(r_p / (2.0 * cfg.A)),
#         "total_electrolyte_res": float((r_n + r_s + r_p) / (2.0 * cfg.A)),
#     }


# def film_resistance(cfg: Config):
#     return float(cfg.Rf if cfg.use_film_resistance else 0.0)


# def pure_ohmic_resistance(cfg: Config):
#     return float(cfg.R_ohm if cfg.use_pure_R_ohm else 0.0)


# def voltage_components_truth(x: np.ndarray, cfg: Config, I: float) -> dict[str, float]:
#     xn, xp = solid_stoich_from_state(x, cfg)
#     ceL, ceR = extract_electrolyte_edges(x, cfg)

#     ocv_info = ocv_decomposition_from_state(x, cfg)
#     ocv_pack = ocv_info["ocv_pack"]

#     I0p, I0n = i0_current_scales(xn, xp, ceL, ceR, cfg)

#     eta_p_raw = (2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(I / (2.0 * max(I0p, 1e-20)))
#     eta_n_raw = (2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(I / (2.0 * max(I0n, 1e-20)))

#     eta_p = cfg.bv_scale * eta_p_raw if cfg.use_eta_p else 0.0
#     eta_n = cfg.bv_scale * eta_n_raw if cfg.use_eta_n else 0.0

#     eta_combo_cell = (eta_p - eta_n) if cfg.eta_mode == "diff" else (eta_p + eta_n)
#     eta_combo_cell *= cfg.kinetic_blend
#     if not cfg.use_kinetic_term:
#         eta_combo_cell = 0.0

#     elyte_info = electrolyte_log_term_details(ceL, ceR, cfg)
#     dphi_e_cell = elyte_info["value"]
#     if not cfg.use_electrolyte_term:
#         dphi_e_cell = 0.0

#     r_parts = electrolyte_resistance_parts(cfg)
#     r_elyte = r_parts["total_electrolyte_res"] if cfg.use_electrolyte_ohmic_resistance else 0.0
#     r_pure = pure_ohmic_resistance(cfg)
#     r_film = film_resistance(cfg)

#     ohmic_pure_cell = -I * r_pure
#     ohmic_film_cell = -I * r_film
#     ohmic_elyte_res_cell = -I * r_elyte

#     ohmic_cell = ohmic_pure_cell + ohmic_film_cell + ohmic_elyte_res_cell
#     ohmic_cell *= cfg.ohmic_blend
#     if not cfg.use_ohmic_term:
#         ohmic_cell = 0.0

#     eta_pack = cfg.N_series * cfg.eta_weight * eta_combo_cell
#     dphi_e_pack = cfg.N_series * cfg.electrolyte_weight * dphi_e_cell
#     ohmic_pack = cfg.N_series * cfg.ohmic_weight * ohmic_cell

#     total = ocv_pack + eta_pack + dphi_e_pack + ohmic_pack

#     return {
#         "xp": float(xp),
#         "xn": float(xn),

#         "Up_cell": float(ocv_info["Up_cell"]) if np.isfinite(ocv_info["Up_cell"]) else np.nan,
#         "Un_cell": float(ocv_info["Un_cell"]) if np.isfinite(ocv_info["Un_cell"]) else np.nan,
#         "Up_pack": float(ocv_info["Up_pack"]) if np.isfinite(ocv_info["Up_pack"]) else np.nan,
#         "minus_Un_pack": float(ocv_info["minus_Un_pack"]) if np.isfinite(ocv_info["minus_Un_pack"]) else np.nan,
#         "ocv_cell": float(ocv_info["ocv_cell"]),
#         "ocv": float(ocv_pack),

#         "eta_p_raw": float(eta_p_raw),
#         "eta_n_raw": float(eta_n_raw),
#         "eta_p": float(cfg.N_series * cfg.eta_weight * eta_p),
#         "minus_eta_n": float(-cfg.N_series * cfg.eta_weight * eta_n),
#         "eta": float(eta_pack),
#         "I0p": float(I0p),
#         "I0n": float(I0n),

#         "ceL": float(elyte_info["ceL"]),
#         "ceR": float(elyte_info["ceR"]),
#         "elyte_ln_ratio": float(elyte_info["ln_ratio"]),
#         "elyte_prefactor": float(elyte_info["prefactor"]),
#         "electrolyte": float(dphi_e_pack),
#         "elyte_orientation_state": cfg.electrolyte_state_orientation,
#         "elyte_orientation_log": cfg.ln_orientation,

#         "r_pure": float(r_pure),
#         "r_film": float(r_film),
#         "r_elyte": float(r_elyte),
#         "R_elyte_neg": float(r_parts["negative_electrolyte_res"]),
#         "R_elyte_sep": float(r_parts["separator_electrolyte_res"]),
#         "R_elyte_pos": float(r_parts["positive_electrolyte_res"]),
#         "ohmic_pure": float(cfg.N_series * cfg.ohmic_weight * ohmic_pure_cell * cfg.ohmic_blend),
#         "ohmic_film": float(cfg.N_series * cfg.ohmic_weight * ohmic_film_cell * cfg.ohmic_blend),
#         "ohmic_elyte_res": float(cfg.N_series * cfg.ohmic_weight * ohmic_elyte_res_cell * cfg.ohmic_blend),
#         "ohmic": float(ohmic_pack),

#         "total": float(total),
#     }


# def truth_z_from_state(x: np.ndarray, cfg: Config, I: float) -> float:
#     comps = voltage_components_truth(x, cfg, I)
#     return float(comps["ocv"] + comps["eta"] + comps["electrolyte"])


# def terminal_voltage_truth(x: np.ndarray, cfg: Config, I: float):
#     comps = voltage_components_truth(x, cfg, I)
#     return float(comps["total"])


# def dynamic_voltage_only(x: np.ndarray, cfg: Config, I: float):
#     comps = voltage_components_truth(x, cfg, I)
#     return float(comps["eta"] + comps["electrolyte"] + comps["ohmic"])


# def output_measurement_truth(x: np.ndarray, cfg: Config, I: float):
#     return terminal_voltage_truth(x, cfg, I)

def full_cell_soc_proxy(x: np.ndarray, cfg: Config) -> float:
    xn, xp = solid_stoich_from_state(x, cfg)
    soc = 0.5 * (xn + (1.0 - xp))
    return float(np.clip(soc, 1e-9, 1.0 - 1e-9))


def _clip_soc_01(s: float) -> float:
    return float(np.clip(s, 1e-9, 1.0 - 1e-9))


def _chebyshev_coeffs_from_settings() -> np.ndarray:
    preset = str(CHEBYSHEV_OCV_PRESET).lower()

    if preset == "default_soft_slope":
        coeffs = CHEBYSHEV_OCV_COEFFS_DEFAULT_SOFT_SLOPE
    elif preset == "stronger_mid_slope":
        coeffs = CHEBYSHEV_OCV_COEFFS_STRONGER_MID_SLOPE
    elif preset == "more_end_curvature":
        coeffs = CHEBYSHEV_OCV_COEFFS_MORE_END_CURVATURE
    elif preset == "manual":
        coeffs = CHEBYSHEV_OCV_COEFFS_MANUAL
    else:
        raise ValueError(
            f"Unsupported CHEBYSHEV_OCV_PRESET={CHEBYSHEV_OCV_PRESET!r}. "
            "Use 'default_soft_slope', 'stronger_mid_slope', "
            "'more_end_curvature', or 'manual'."
        )

    coeffs = np.asarray(coeffs, dtype=np.float64).reshape(-1)
    if len(coeffs) == 0:
        raise ValueError("Chebyshev coefficient list must not be empty.")
    return coeffs


def _chebyshev_T_series(xc: float, n_terms: int) -> np.ndarray:
    xc = float(np.clip(xc, -1.0, 1.0))
    if n_terms <= 0:
        return np.zeros(0, dtype=np.float64)

    T = np.zeros(n_terms, dtype=np.float64)
    T[0] = 1.0
    if n_terms >= 2:
        T[1] = xc
    for k in range(2, n_terms):
        T[k] = 2.0 * xc * T[k - 1] - T[k - 2]
    return T


def remap_soc_for_chebyshev(soc: float) -> float:
    s = _clip_soc_01(soc)

    if not CHEBYSHEV_USE_SOC_REMAP:
        return s

    c = float(np.clip(CHEBYSHEV_SOC_REMAP_CENTER, 1e-6, 1.0 - 1e-6))
    g = float(max(CHEBYSHEV_SOC_REMAP_GAIN, 1e-6))

    z = (s - c) / g
    s_map = 1.0 / (1.0 + np.exp(-z))
    return _clip_soc_01(s_map)


def full_cell_ocv_from_soc(soc: float) -> float:
    s = _clip_soc_01(soc)
    v = FULL_CELL_OCV_MIN + (FULL_CELL_OCV_MAX - FULL_CELL_OCV_MIN) * s
    v += 0.05 * np.tanh((s - 0.12) / 0.05)
    v -= 0.03 * np.tanh((s - 0.55) / 0.14)
    v += 0.06 * np.tanh((s - 0.88) / 0.06)
    return float(np.clip(v, FULL_CELL_OCV_MIN, FULL_CELL_OCV_MAX))


def full_cell_ocv_from_soc_chebyshev(soc: float) -> float:
    s_raw = _clip_soc_01(soc)
    s = remap_soc_for_chebyshev(s_raw)
    xc = 2.0 * s - 1.0

    coeffs = _chebyshev_coeffs_from_settings()
    T = _chebyshev_T_series(xc, len(coeffs))
    v = float(np.dot(coeffs, T))

    v = float(CHEBYSHEV_OCV_GAIN * v + CHEBYSHEV_OCV_BIAS)

    if USE_CHEBYSHEV_OCV_CLIP:
        v = float(np.clip(v, FULL_CELL_OCV_MIN, FULL_CELL_OCV_MAX))

    return v


def ocv_decomposition_from_state(x: np.ndarray, cfg: Config) -> dict[str, float]:
    xn, xp = solid_stoich_from_state(x, cfg)
    mode = str(cfg.ocv_model_mode).lower()

    if mode == "electrode_ocp":
        Up_cell = ocp_p(xp)
        Un_cell = ocp_n(xn)
        ocv_cell = Up_cell - Un_cell

        if cfg.use_ocv_scale:
            ocv_cell *= cfg.ocv_scale
            Up_cell *= cfg.ocv_scale
            Un_cell *= cfg.ocv_scale

        Up_pack = cfg.N_series * cfg.ocv_weight * Up_cell
        minus_Un_pack = -cfg.N_series * cfg.ocv_weight * Un_cell
        ocv_pack = Up_pack + minus_Un_pack

        return {
            "xp": float(xp),
            "xn": float(xn),
            "soc": float(full_cell_soc_proxy(x, cfg)),
            "Up_cell": float(Up_cell),
            "Un_cell": float(Un_cell),
            "Up_pack": float(Up_pack),
            "minus_Un_pack": float(minus_Un_pack),
            "ocv_cell": float(ocv_cell),
            "ocv_pack": float(ocv_pack),
            "ocv_mode": "electrode_ocp",
        }

    elif mode == "soc_proxy_shaped":
        soc = full_cell_soc_proxy(x, cfg)
        ocv_cell = full_cell_ocv_from_soc(soc)

        if cfg.use_ocv_scale:
            ocv_cell *= cfg.ocv_scale

        ocv_pack = cfg.N_series * cfg.ocv_weight * ocv_cell

        return {
            "xp": float(xp),
            "xn": float(xn),
            "soc": float(soc),
            "Up_cell": np.nan,
            "Un_cell": np.nan,
            "Up_pack": np.nan,
            "minus_Un_pack": np.nan,
            "ocv_cell": float(ocv_cell),
            "ocv_pack": float(ocv_pack),
            "ocv_mode": "soc_proxy_shaped",
        }

    elif mode == "chebyshev_soc":
        soc = full_cell_soc_proxy(x, cfg)
        ocv_cell = full_cell_ocv_from_soc_chebyshev(soc)

        if cfg.use_ocv_scale:
            ocv_cell *= cfg.ocv_scale

        ocv_pack = cfg.N_series * cfg.ocv_weight * ocv_cell

        return {
            "xp": float(xp),
            "xn": float(xn),
            "soc": float(soc),
            "Up_cell": np.nan,
            "Un_cell": np.nan,
            "Up_pack": np.nan,
            "minus_Un_pack": np.nan,
            "ocv_cell": float(ocv_cell),
            "ocv_pack": float(ocv_pack),
            "ocv_mode": "chebyshev_soc",
        }

    else:
        raise ValueError(
            f"Unsupported OCV_MODEL_MODE={cfg.ocv_model_mode!r}. "
            "Use 'electrode_ocp', 'soc_proxy_shaped', or 'chebyshev_soc'."
        )


def electrolyte_boundary_concentrations_from_state(
    x: np.ndarray,
    cfg: Config,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)

    if STATE_SPACE_VARIANT == "reduced_7":
        ce_block = x[VARIANT_IDXS["reduced_7"]["ce"] if "ce" in VARIANT_IDXS["reduced_7"] else slice(4, 7)].copy()
        if cfg.ce_is_deviation:
            ce_block = cfg.ce0 + ce_block
        ce_block = np.maximum(ce_block, cfg.ce_guard)

        if cfg.electrolyte_state_orientation == "flipped":
            ceL = float(ce_block[-1])
            ceR = float(ce_block[0])
        else:
            ceL = float(ce_block[0])
            ceR = float(ce_block[-1])

        return ceL, ceR

    elif STATE_SPACE_VARIANT == "full_14":
        ce_block = x[VARIANT_IDXS["full_14"]["ce"]].copy()
        if cfg.ce_is_deviation:
            ce_block = cfg.ce0 + ce_block
        ce_block = np.maximum(ce_block, cfg.ce_guard)

        if cfg.electrolyte_state_orientation == "flipped":
            ceL = float(ce_block[-1])
            ceR = float(ce_block[0])
        else:
            ceL = float(ce_block[0])
            ceR = float(ce_block[-1])

        return ceL, ceR

    else:
        raise ValueError(f"Unsupported STATE_SPACE_VARIANT={STATE_SPACE_VARIANT!r}")


def exchange_current_density_details(
    x: np.ndarray,
    cfg: Config,
) -> dict[str, float]:
    xn, xp = solid_stoich_from_state(x, cfg)
    ceL, ceR = electrolyte_boundary_concentrations_from_state(x, cfg)

    theta_n = float(np.clip(xn, cfg.theta_guard, 1.0 - cfg.theta_guard))
    theta_p = float(np.clip(xp, cfg.theta_guard, 1.0 - cfg.theta_guard))
    ce_avg = float(max(0.5 * (ceL + ceR), cfg.ce_guard))

    i0n = (
        cfg.exchange_current_scale_n
        * cfg.k_n0
        * np.sqrt(ce_avg)
        * np.sqrt(theta_n * (1.0 - theta_n))
        * cfg.bv_scale
    )
    i0p = (
        cfg.exchange_current_scale_p
        * cfg.k_p0
        * np.sqrt(ce_avg)
        * np.sqrt(theta_p * (1.0 - theta_p))
        * cfg.bv_scale
    )

    i0n = float(max(i0n, cfg.I0_floor_n))
    i0p = float(max(i0p, cfg.I0_floor_p))

    return {
        "xn": float(theta_n),
        "xp": float(theta_p),
        "ce_avg": float(ce_avg),
        "I0n": float(i0n),
        "I0p": float(i0p),
    }


def reaction_overpotential_details(
    x: np.ndarray,
    cfg: Config,
    I: float,
) -> dict[str, float]:
    ex = exchange_current_density_details(x, cfg)
    Iabs = float(I)

    eta_p_raw = float((2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(Iabs / (2.0 * ex["I0p"])))
    eta_n_raw = float((2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(Iabs / (2.0 * ex["I0n"])))

    eta_p = eta_p_raw if cfg.use_eta_p else 0.0
    eta_n = eta_n_raw if cfg.use_eta_n else 0.0

    return {
        "I0p": float(ex["I0p"]),
        "I0n": float(ex["I0n"]),
        "eta_p_raw": float(eta_p_raw),
        "eta_n_raw": float(eta_n_raw),
        "eta_p": float(eta_p),
        "eta_n": float(eta_n),
    }


def electrolyte_log_term_details(
    ceL: float,
    ceR: float,
    cfg: Config,
) -> dict[str, float]:
    ceL = float(max(ceL, cfg.ce_guard))
    ceR = float(max(ceR, cfg.ce_guard))

    if cfg.ln_orientation == "right_over_left":
        ln_ratio = float(np.log(ceR / ceL))
    else:
        ln_ratio = float(np.log(ceL / ceR))

    prefactor = float(((1.0 - cfg.t_plus) * 2.0 * cfg.R * cfg.T) / cfg.F)
    value = float(prefactor * ln_ratio * cfg.electrolyte_log_blend) if cfg.use_electrolyte_log_term else 0.0

    return {
        "ceL": float(ceL),
        "ceR": float(ceR),
        "ln_ratio": float(ln_ratio),
        "prefactor": float(prefactor),
        "value": float(value),
    }


def electrolyte_resistance_parts(cfg: Config) -> dict[str, float]:
    Aeff = max(float(cfg.A), 1e-12)

    Rn_elyte = float(cfg.L1 / max(cfg.kappa_n_eff * Aeff, 1e-12))
    Rs_elyte = float(cfg.L2 / max(cfg.kappa_s_eff * Aeff, 1e-12))
    Rp_elyte = float(cfg.L3 / max(cfg.kappa_p_eff * Aeff, 1e-12))

    total = Rn_elyte + Rs_elyte + Rp_elyte

    return {
        "negative_electrolyte_res": float(Rn_elyte),
        "separator_electrolyte_res": float(Rs_elyte),
        "positive_electrolyte_res": float(Rp_elyte),
        "total_electrolyte_res": float(total),
    }


def pure_ohmic_resistance(cfg: Config) -> float:
    return float(cfg.R_ohm) if cfg.use_pure_R_ohm else 0.0


def film_resistance(cfg: Config) -> float:
    return float(cfg.Rf) if cfg.use_film_resistance else 0.0


def voltage_components_truth(x: np.ndarray, cfg: Config, I: float) -> dict[str, float]:
    xn, xp = solid_stoich_from_state(x, cfg)

    ocv_info = ocv_decomposition_from_state(x, cfg)
    ocv_pack = float(ocv_info["ocv_pack"])

    ceL, ceR = electrolyte_boundary_concentrations_from_state(x, cfg)

    eta_info = reaction_overpotential_details(x, cfg, I)
    eta_p_raw = eta_info["eta_p_raw"]
    eta_n_raw = eta_info["eta_n_raw"]

    eta_p = eta_info["eta_p"]
    eta_n = eta_info["eta_n"]

    # BV terms eta_p and eta_n are computed here as positive magnitudes.
    # For discharge, polarization should reduce terminal voltage.
    # Therefore:
    #   - "diff" keeps the legacy algebra
    #   - "sum" means a net voltage drop from both electrode overpotentials
    if cfg.eta_mode == "diff":
        eta_combo_cell = eta_p - eta_n
    elif cfg.eta_mode == "sum":
        eta_combo_cell = -(eta_p + eta_n)
    else:
        raise ValueError(f"Unsupported eta_mode={cfg.eta_mode!r}")

    eta_combo_cell *= cfg.kinetic_blend
    if not cfg.use_kinetic_term:
        eta_combo_cell = 0.0



    elyte_info = electrolyte_log_term_details(ceL, ceR, cfg)
    dphi_e_cell = elyte_info["value"]
    if not cfg.use_electrolyte_term:
        dphi_e_cell = 0.0

    r_parts = electrolyte_resistance_parts(cfg)
    r_elyte = r_parts["total_electrolyte_res"] if cfg.use_electrolyte_ohmic_resistance else 0.0
    r_pure = pure_ohmic_resistance(cfg)
    r_film = film_resistance(cfg)

    ohmic_pure_cell = -I * r_pure
    ohmic_film_cell = -I * r_film
    ohmic_elyte_res_cell = -I * r_elyte

    ohmic_cell = ohmic_pure_cell + ohmic_film_cell + ohmic_elyte_res_cell
    ohmic_cell *= cfg.ohmic_blend
    if not cfg.use_ohmic_term:
        ohmic_cell = 0.0

    eta_pack = cfg.N_series * cfg.eta_weight * eta_combo_cell
    dphi_e_pack = cfg.N_series * cfg.electrolyte_weight * dphi_e_cell
    ohmic_pack = cfg.N_series * cfg.ohmic_weight * ohmic_cell

    total = ocv_pack + eta_pack + dphi_e_pack + ohmic_pack

    return {
        "xp": float(xp),
        "xn": float(xn),

        "Up_cell": float(ocv_info["Up_cell"]) if np.isfinite(ocv_info["Up_cell"]) else np.nan,
        "Un_cell": float(ocv_info["Un_cell"]) if np.isfinite(ocv_info["Un_cell"]) else np.nan,
        "Up_pack": float(ocv_info["Up_pack"]) if np.isfinite(ocv_info["Up_pack"]) else np.nan,
        "minus_Un_pack": float(ocv_info["minus_Un_pack"]) if np.isfinite(ocv_info["minus_Un_pack"]) else np.nan,
        "ocv_cell": float(ocv_info["ocv_cell"]),
        "ocv": float(ocv_pack),

        "eta_p_raw": float(eta_p_raw),
        "eta_n_raw": float(eta_n_raw),
        "eta_p": float(cfg.N_series * cfg.eta_weight * eta_p),
        "minus_eta_n": float(-cfg.N_series * cfg.eta_weight * eta_n),
        "eta": float(eta_pack),
        "I0p": float(eta_info["I0p"]),
        "I0n": float(eta_info["I0n"]),

        "ceL": float(elyte_info["ceL"]),
        "ceR": float(elyte_info["ceR"]),
        "elyte_ln_ratio": float(elyte_info["ln_ratio"]),
        "elyte_prefactor": float(elyte_info["prefactor"]),
        "electrolyte": float(dphi_e_pack),
        "elyte_orientation_state": cfg.electrolyte_state_orientation,
        "elyte_orientation_log": cfg.ln_orientation,

        "r_pure": float(r_pure),
        "r_film": float(r_film),
        "r_elyte": float(r_elyte),
        "R_elyte_neg": float(r_parts["negative_electrolyte_res"]),
        "R_elyte_sep": float(r_parts["separator_electrolyte_res"]),
        "R_elyte_pos": float(r_parts["positive_electrolyte_res"]),
        "ohmic_pure": float(cfg.N_series * cfg.ohmic_weight * ohmic_pure_cell * cfg.ohmic_blend),
        "ohmic_film": float(cfg.N_series * cfg.ohmic_weight * ohmic_film_cell * cfg.ohmic_blend),
        "ohmic_elyte_res": float(cfg.N_series * cfg.ohmic_weight * ohmic_elyte_res_cell * cfg.ohmic_blend),
        "ohmic": float(ohmic_pack),

        "total": float(total),
    }


def truth_z_from_state(x: np.ndarray, cfg: Config, I: float) -> float:
    comps = voltage_components_truth(x, cfg, I)
    return float(comps["ocv"] + comps["eta"] + comps["electrolyte"])


def terminal_voltage_truth(x: np.ndarray, cfg: Config, I: float):
    comps = voltage_components_truth(x, cfg, I)
    return float(comps["total"])


def dynamic_voltage_only(x: np.ndarray, cfg: Config, I: float):
    comps = voltage_components_truth(x, cfg, I)
    return float(comps["eta"] + comps["electrolyte"] + comps["ohmic"])


def output_measurement_truth(x: np.ndarray, cfg: Config, I: float):
    return terminal_voltage_truth(x, cfg, I)


def build_constant_profile(T, I_const):
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    return np.full_like(T, float(I_const), dtype=np.float64)


def build_step_profile(T, t_step, I_before, I_after):
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    U = np.full_like(T, float(I_before), dtype=np.float64)
    U[T >= float(t_step)] = float(I_after)
    return U


def build_multistep_profile(T, breaks, levels):
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    if len(breaks) != len(levels):
        raise ValueError("breaks and levels must have same length")
    U = np.zeros_like(T)
    for i in range(len(breaks)):
        t0 = breaks[i]
        t1 = breaks[i + 1] if i + 1 < len(breaks) else np.inf
        mask = (T >= t0) & (T < t1)
        U[mask] = levels[i]
    return U


def build_profile_from_mode(T, mode):
    mode = str(mode).lower()

    if mode == "constant":
        return build_constant_profile(T, I_CONST)

    if mode == "step":
        return build_step_profile(T, T_STEP, I_BEFORE, I_AFTER)

    if mode == "multistep":
        return build_multistep_profile(T, MULTISTEP_BREAKS, MULTISTEP_LEVELS)

    raise ValueError(f"Unsupported mode: {mode}")


def c2d_zoh(A, B, dt):
    nx, nu = A.shape[0], B.shape[1]
    M = np.zeros((nx + nu, nx + nu), dtype=np.float64)
    M[:nx, :nx] = A
    M[:nx, nx:] = B
    Md = expm(M * dt)
    return Md[:nx, :nx], Md[:nx, nx:]


def generate_profile_data_truth(
    cfg: Config,
    T,
    I_profile,
    theta_n0=0.8,
    theta_p0=0.4,
    ce0=0.0,
    x0_override=None,
):
    T = np.asarray(T, dtype=np.float64).reshape(-1)
    I_profile = np.asarray(I_profile, dtype=np.float64).reshape(-1)

    _, A, B = assemble_truth_system(cfg, STATE_SPACE_VARIANT)

    if x0_override is None:
        x = make_x0_truth(
            cfg,
            STATE_SPACE_VARIANT,
            theta_n0=theta_n0,
            theta_p0=theta_p0,
            ce0=ce0,
        ).copy()
    else:
        x = np.asarray(x0_override, dtype=np.float64).reshape(-1).copy()

    n_states = A.shape[0]
    X = np.zeros((len(T), n_states), dtype=np.float64)
    Z = np.zeros((len(T), 1), dtype=np.float64)
    Y = np.zeros((len(T), 1), dtype=np.float64)
    U = I_profile.reshape(-1, 1)

    X[0] = x
    Z[0, 0] = truth_z_from_state(x, cfg, U[0, 0])
    Y[0, 0] = terminal_voltage_truth(x, cfg, U[0, 0])

    dt = float(np.median(np.diff(T)))
    Ad, Bd = c2d_zoh(A, B, dt)

    for k in range(1, len(T)):
        x = Ad @ x + (Bd[:, 0] * U[k - 1, 0])
        X[k] = x
        Z[k, 0] = truth_z_from_state(x, cfg, U[k, 0])
        Y[k, 0] = terminal_voltage_truth(x, cfg, U[k, 0])

    return T, U, X, Z, Y


def simulate_until_event(
    cfg: Config,
    I_value: float,
    dt: float,
    max_time: float,
    theta_n0: float = 0.8,
    theta_p0: float = 0.4,
    ce0: float = 0.0,
    x0_override: Optional[np.ndarray] = None,
):
    T = np.arange(0.0, max_time + dt, dt, dtype=np.float64)
    U = np.full_like(T, float(I_value), dtype=np.float64)
    t, Uc, X, Z, Y = generate_profile_data_truth(
        cfg,
        T,
        U,
        theta_n0=theta_n0,
        theta_p0=theta_p0,
        ce0=ce0,
        x0_override=x0_override,
    )

    reason = "max_time_reached"
    event_index = len(t) - 1

    for k in range(len(t)):
        xk = X[k]
        yk = float(Y[k, 0])
        xn, xp = solid_stoich_from_state(xk, cfg)

        if xn <= THETA_MIN_CUTOFF:
            reason = "xn_hit_min"
            event_index = k
            break
        if xp >= THETA_MAX_CUTOFF:
            reason = "xp_hit_max"
            event_index = k
            break
        if xn >= THETA_MAX_CUTOFF:
            reason = "xn_hit_max"
            event_index = k
            break
        if xp <= THETA_MIN_CUTOFF:
            reason = "xp_hit_min"
            event_index = k
            break
        if yk <= V_MIN_CUTOFF:
            reason = "voltage_hit_min"
            event_index = k
            break
        if yk >= V_MAX_CUTOFF:
            reason = "voltage_hit_max"
            event_index = k
            break

    out = {
        "I_value": float(I_value),
        "t_event": float(t[event_index]),
        "event_index": int(event_index),
        "reason": reason,
        "t": t[:event_index + 1].copy(),
        "u": Uc[:event_index + 1, 0].copy(),
        "X": X[:event_index + 1].copy(),
        "Z": Z[:event_index + 1].copy(),
        "y": Y[:event_index + 1, 0].copy(),
    }
    return out


def build_synthetic_dataset(cfg: Config):
    T = np.arange(0.0, SIM_T_END + SIM_DT, SIM_DT, dtype=np.float64)
    U_profile = build_profile_from_mode(T, SYNTHETIC_MODE)

    x0_override = None
    if USE_PRECONDITION:
        T_pre_arr = np.arange(0.0, T_PRE + SIM_DT, SIM_DT, dtype=np.float64)
        U_pre = np.full_like(T_pre_arr, I_PRE, dtype=np.float64)

        _, _, X_pre, _, _ = generate_profile_data_truth(
            cfg, T_pre_arr, U_pre, THETA_N0, THETA_P0, CE0_DEV
        )
        x0_override = X_pre[-1].copy()

    t, Uc, X, Z, Y = generate_profile_data_truth(
        cfg, T, U_profile, THETA_N0, THETA_P0, CE0_DEV, x0_override=x0_override
    )

    if ADD_VOLTAGE_NOISE:
        rng = np.random.default_rng(0)
        Y = Y + V_NOISE_STD * rng.standard_normal(size=Y.shape)

    return t, Uc, X, Z, Y, x0_override

def build_real_dataset(
    cycle_index: Optional[int] = None,
    current_mode: Optional[str] = None,
):
    if BioLogic is None:
        raise ImportError("galvani is required for MPR reading in real-data mode.")

    if cycle_index is None:
        cycle_index = CYCLE_INDEX

    if current_mode is None:
        current_mode = REAL_COMPARE_INPUT_COLUMN_MODE

    mpr = BioLogic.MPRfile(str(resolve_mpr_path(MPR_PATH)))
    df = pd.DataFrame(mpr.data)
    df.columns = [str(c).strip() for c in df.columns]

    if TIME_COL not in df.columns or V_COL not in df.columns:
        raise KeyError(
            f"Required columns not found. Need {TIME_COL!r} and {V_COL!r}. "
            f"Available columns: {df.columns.tolist()}"
        )

    current_col = "control/mA" if current_mode == "control" else I_COL
    if current_col not in df.columns:
        raise KeyError(
            f"Selected real current column {current_col!r} not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    t_all = df[TIME_COL].to_numpy(dtype=np.float64).reshape(-1)
    i_all = df[current_col].to_numpy(dtype=np.float64).reshape(-1)
    v_all = df[V_COL].to_numpy(dtype=np.float64).reshape(-1)

    if PRINT_REAL_RAW_DT_DIAGNOSTICS:
        print_dt_stats("Real raw time", t_all)

    if PRINT_REAL_CURRENT_COLUMN_DIAGNOSTICS:
        print("\nReal-data current column diagnostics:")
        for c in ALT_I_COL_CANDIDATES:
            if c in df.columns:
                arr = df[c].to_numpy(dtype=np.float64).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if len(finite):
                    print(
                        f"  {c:12s} | min={np.min(finite):.6g} "
                        f"max={np.max(finite):.6g} mean={np.mean(finite):.6g}"
                    )

    i = i_all.copy()
    if REAL_CURRENT_UNITS.lower() == "ma":
        i = i / 1000.0

    mask = i < 0.0 if RAW_DISCHARGE_SIGN == "negative" else i > 0.0
    idx = np.where(mask)[0]
    if idx.size == 0:
        raise RuntimeError("No discharge segments found.")

    splits = np.where(np.diff(idx) > 1)[0]
    groups = [g for g in np.split(idx, splits + 1) if len(g) >= MIN_CYCLE_LEN]
    if not groups:
        raise RuntimeError("No discharge segments found with current settings.")
    if cycle_index >= len(groups):
        raise IndexError(f"cycle_index={cycle_index} but only {len(groups)} discharge segments found.")

    g = groups[cycle_index]
    t_sel_raw = t_all[g] - t_all[g][0]
    i_sel_raw = i[g]
    v_sel_raw = v_all[g]

    u_raw = -i_sel_raw if RAW_DISCHARGE_SIGN == "negative" else i_sel_raw
    y_raw = v_sel_raw

    if PRINT_REAL_RAW_DT_DIAGNOSTICS:
        print_dt_stats("Selected real segment raw time", t_sel_raw)

    if RESAMPLE_REAL:
        t_sel, u_col, y_col = resample_uniform(t_sel_raw, col(u_raw), col(y_raw), REAL_DT)
    else:
        t_sel, u_col, y_col = t_sel_raw.copy(), col(u_raw), col(y_raw)

    real_meta = {
        "cycle_index": int(cycle_index),
        "current_col_used": current_col,
        "raw_duration": float(t_sel_raw[-1]) if len(t_sel_raw) else np.nan,
        "resampled_duration": float(t_sel[-1]) if len(t_sel) else np.nan,
        "raw_current_min": float(np.min(u_raw)) if len(u_raw) else np.nan,
        "raw_current_max": float(np.max(u_raw)) if len(u_raw) else np.nan,
        "raw_voltage_min": float(np.min(y_raw)) if len(y_raw) else np.nan,
        "raw_voltage_max": float(np.max(y_raw)) if len(y_raw) else np.nan,
        "raw_current_abs_mean": float(np.mean(np.abs(u_raw))) if len(u_raw) else np.nan,
        "raw_dt_stats": finite_diff_stats(t_sel_raw),
        "resampled_dt_stats": finite_diff_stats(t_sel),
        "n_groups_found": int(len(groups)),
    }

    return t_sel, u_col, None, None, y_col, real_meta


def compute_voltage_components(cfg, X_hist: np.ndarray, u_hist: np.ndarray) -> dict[str, np.ndarray]:
    """
    Aggregate per-sample voltage-component decomposition across a trajectory.

    Parameters
    ----------
    cfg : object
        Configuration object used by voltage_components_truth.
    X_hist : np.ndarray
        State trajectory with shape (N, nx).
    u_hist : np.ndarray
        Input/current trajectory with shape (N,) or (N, 1).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing total voltage terms and detailed sub-terms
        across time. Every value is returned as a 1D float64 array of length N.

    Notes
    -----
    - Uses the truth current sign, not the display/sign-flipped current.
    - Requires voltage_components_truth(...) to already be defined.
    """
    X_hist = np.asarray(X_hist, dtype=np.float64)
    u_hist = _ensure_1d_array(u_hist, dtype=float)

    if X_hist.ndim != 2:
        raise ValueError(f"X_hist must be 2D with shape (N, nx), got shape {X_hist.shape}")
    if len(X_hist) != len(u_hist):
        raise ValueError(
            f"Length mismatch: len(X_hist)={len(X_hist)} but len(u_hist)={len(u_hist)}"
        )
    if "voltage_components_truth" not in globals():
        raise NameError(
            "voltage_components_truth is not defined. "
            "Define it before calling compute_voltage_components(...)."
        )

    comp_list = [
        voltage_components_truth(X_hist[k], cfg, u_hist[k])
        for k in range(len(u_hist))
    ]

    def pack(name: str, default: float = 0.0) -> np.ndarray:
        return np.array(
            [c.get(name, default) for c in comp_list],
            dtype=np.float64
        )

    out = {
        # total terms
        "ocv": pack("ocv"),
        "eta": pack("eta"),
        "electrolyte": pack("electrolyte"),
        "ohmic": pack("ohmic"),

        # OCV decomposition
        "Up_pack": pack("Up_pack"),
        "minus_Un_pack": pack("minus_Un_pack"),
        "Up_cell": pack("Up_cell"),
        "Un_cell": pack("Un_cell"),

        # kinetic decomposition
        "eta_p": pack("eta_p"),
        "minus_eta_n": pack("minus_eta_n"),
        "I0p": pack("I0p"),
        "I0n": pack("I0n"),

        # electrolyte decomposition
        "ceL": pack("ceL"),
        "ceR": pack("ceR"),
        "elyte_ln_ratio": pack("elyte_ln_ratio"),
        "elyte_prefactor": pack("elyte_prefactor"),

        # ohmic decomposition
        "ohmic_pure": pack("ohmic_pure"),
        "ohmic_film": pack("ohmic_film"),
        "ohmic_elyte_res": pack("ohmic_elyte_res"),
        "R_elyte_neg": pack("R_elyte_neg"),
        "R_elyte_sep": pack("R_elyte_sep"),
        "R_elyte_pos": pack("R_elyte_pos"),
    }

    out["dynamic_total"] = out["eta"] + out["electrolyte"] + out["ohmic"]
    return out

# %% =====================================================
# Real-data helpers using standalone-style discharge-cycle extraction
# =====================================================
def _current_column_from_mode(mode: str) -> str:
    if mode == "control":
        return "control/mA"
    if mode == "I":
        return I_COL
    raise ValueError(f"Unsupported current mode: {mode!r}")


def _convert_current_units_to_A(i_vals: np.ndarray) -> np.ndarray:
    i_vals = np.asarray(i_vals, dtype=np.float64).reshape(-1)
    if REAL_CURRENT_UNITS.lower() == "ma":
        return i_vals / 1000.0
    return i_vals


def estimate_ts_from_time_notebook(t: np.ndarray, ts_fallback: float) -> float:
    t = _ensure_1d_array(t)
    if len(t) < 2:
        return float(ts_fallback)
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return float(ts_fallback)
    return float(np.median(dt))


def get_previous_segment_notebook(df: pd.DataFrame, start_pos: int, previous_points: int = 10) -> pd.DataFrame:
    lo = max(0, int(start_pos) - int(previous_points))
    return df.iloc[lo:start_pos].copy()


def _is_active_current_notebook(u_val: float, cycle_type: str, threshold: float) -> bool:
    if cycle_type == "charge":
        return u_val > threshold
    if cycle_type == "discharge":
        return u_val < -threshold
    raise ValueError("cycle_type must be 'charge' or 'discharge'.")


def collect_sign_cycle_notebook(
    df: pd.DataFrame,
    start_pos: int,
    cycle_type: str,
    threshold: float,
    input_col: str,
) -> tuple[pd.DataFrame, int]:
    i = int(start_pos)
    rows = []
    while i < len(df):
        u_val = float(df.iloc[i][input_col])
        if _is_active_current_notebook(u_val, cycle_type=cycle_type, threshold=threshold):
            rows.append(df.iloc[[i]])
            i += 1
        else:
            break

    cycle_df = pd.concat(rows, axis=0).copy() if rows else df.iloc[0:0].copy()
    return cycle_df, i


def find_cycles_notebook(
    df: pd.DataFrame,
    input_col: str,
    cycle_type: str,
    previous_points: int,
    min_active_current: float,
    min_cycle_len: int,
) -> list[pd.DataFrame]:
    x = df.copy().reset_index(drop=True)
    cycles = []
    i = 0

    while i < len(x):
        u_val = float(x.iloc[i][input_col])
        starts_cycle = _is_active_current_notebook(
            u_val,
            cycle_type=cycle_type,
            threshold=min_active_current,
        )

        if starts_cycle:
            prev_df = get_previous_segment_notebook(
                x,
                i,
                previous_points=previous_points,
            )
            cur_df, i = collect_sign_cycle_notebook(
                x,
                i,
                cycle_type=cycle_type,
                threshold=min_active_current,
                input_col=input_col,
            )
            if len(cur_df) >= int(min_cycle_len):
                cyc = pd.concat([prev_df, cur_df], axis=0).reset_index(drop=True)
                cycles.append(cyc)
        else:
            i += 1

    return cycles


def build_real_dataset(
    cycle_index: Optional[int] = None,
    current_mode: Optional[str] = None,
):
    if BioLogic is None:
        raise ImportError("galvani is required for MPR reading in real-data mode.")

    if cycle_index is None:
        cycle_index = CYCLE_INDEX

    if current_mode is None:
        current_mode = REAL_COMPARE_INPUT_COLUMN_MODE

    mpr = BioLogic.MPRfile(str(resolve_mpr_path(MPR_PATH)))
    df = pd.DataFrame(mpr.data)
    df.columns = [str(c).strip() for c in df.columns]

    if TIME_COL not in df.columns or V_COL not in df.columns:
        raise KeyError(
            f"Required columns not found. Need {TIME_COL!r} and {V_COL!r}. "
            f"Available columns: {df.columns.tolist()}"
        )

    current_col = _current_column_from_mode(current_mode)
    if current_col not in df.columns:
        raise KeyError(
            f"Selected real current column {current_col!r} not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    required = [TIME_COL, current_col, V_COL]
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if REAL_OE_DROP_NAN:
        df = df.dropna(subset=required).reset_index(drop=True)

    t_all = df[TIME_COL].to_numpy(dtype=np.float64).reshape(-1)

    if PRINT_REAL_RAW_DT_DIAGNOSTICS:
        print_dt_stats("Real raw time", t_all)

    if PRINT_REAL_CURRENT_COLUMN_DIAGNOSTICS:
        print("\nReal-data current column diagnostics:")
        for c in ALT_I_COL_CANDIDATES:
            if c in df.columns:
                arr = df[c].to_numpy(dtype=np.float64).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if len(finite):
                    print(
                        f"  {c:12s} | min={np.min(finite):.6g} "
                        f"max={np.max(finite):.6g} mean={np.mean(finite):.6g}"
                    )

    df_work = df.copy()
    if REAL_OE_NORMALIZE_TIME_TO_ZERO and len(df_work) > 0:
        df_work[TIME_COL] = df_work[TIME_COL] - float(df_work[TIME_COL].iloc[0])

    # convert chosen current column to A for cycle extraction
    df_work[current_col] = _convert_current_units_to_A(df_work[current_col].to_numpy(dtype=np.float64))

    cycles = find_cycles_notebook(
        df_work,
        input_col=current_col,
        cycle_type=REAL_OE_CYCLE_TYPE,
        previous_points=REAL_OE_PREVIOUS_POINTS,
        min_active_current=REAL_OE_MIN_ACTIVE_CURRENT,
        min_cycle_len=REAL_OE_MIN_CYCLE_LEN,
    )

    if len(cycles) == 0:
        raise RuntimeError(
            f"No {REAL_OE_CYCLE_TYPE} cycles found using current column {current_col!r}."
        )

    if cycle_index >= len(cycles):
        raise IndexError(f"cycle_index={cycle_index} but only {len(cycles)} cycles found.")

    cyc = cycles[cycle_index].copy().reset_index(drop=True)

    t_sel_raw = cyc[TIME_COL].to_numpy(dtype=np.float64)
    if REAL_OE_NORMALIZE_TIME_TO_ZERO and len(t_sel_raw) > 0:
        t_sel_raw = t_sel_raw - t_sel_raw[0]

    u_raw = cyc[current_col].to_numpy(dtype=np.float64)
    y_raw = cyc[V_COL].to_numpy(dtype=np.float64)

    if PRINT_REAL_RAW_DT_DIAGNOSTICS:
        print_dt_stats("Selected real segment raw time", t_sel_raw)

    if RESAMPLE_REAL:
        t_sel, u_col, y_col = resample_uniform(t_sel_raw, col(u_raw), col(y_raw), REAL_DT)
    else:
        t_sel, u_col, y_col = t_sel_raw.copy(), col(u_raw), col(y_raw)

    meta_out = {
        "cycle_index": int(cycle_index),
        "current_col_used": current_col,
        "cycle_type": REAL_OE_CYCLE_TYPE,
        "raw_duration": float(t_sel_raw[-1]) if len(t_sel_raw) else np.nan,
        "resampled_duration": float(t_sel[-1]) if len(t_sel) else np.nan,
        "raw_current_min": float(np.min(u_raw)) if len(u_raw) else np.nan,
        "raw_current_max": float(np.max(u_raw)) if len(u_raw) else np.nan,
        "raw_voltage_min": float(np.min(y_raw)) if len(y_raw) else np.nan,
        "raw_voltage_max": float(np.max(y_raw)) if len(y_raw) else np.nan,
        "raw_current_abs_mean": float(np.mean(np.abs(u_raw))) if len(u_raw) else np.nan,
        "raw_dt_stats": finite_diff_stats(t_sel_raw),
        "resampled_dt_stats": finite_diff_stats(t_sel),
        "n_groups_found": int(len(cycles)),
        "ts_time_based": estimate_ts_from_time_notebook(t_sel_raw, REAL_OE_TS_FALLBACK),
    }

    return t_sel, u_col, None, None, y_col, meta_out

# %% =====================================================
# CELL 6 — Unified dataset builder, decomposition, fit target,
#          reporting plots, scaled identification signals,
#          and real-vs-synthetic comparison diagnostics
# =====================================================
import copy

REAL_COMPARE_CACHE = None
SYNTH_COMPARE_CACHE = None


# ---------------------------------------------------------
# helper: clone cfg and override selected attributes safely
# ---------------------------------------------------------
def _clone_cfg_with_overrides(cfg_in, overrides: dict):
    cfg_out = copy.deepcopy(cfg_in)
    for k, v in overrides.items():
        if hasattr(cfg_out, k):
            setattr(cfg_out, k, v)
    return cfg_out


# ---------------------------------------------------------
# helper: build one synthetic dataset using an explicit cfg
# ---------------------------------------------------------
def _build_current_synthetic_dataset(cfg_local=None):
    if cfg_local is None:
        cfg_local = cfg

    t_loc, U_loc, X_loc, Z_loc, Y_loc, X0_loc = build_synthetic_dataset(cfg_local)
    return {
        "t": _ensure_1d_array(t_loc),
        "u_truth": _ensure_1d_array(U_loc[:, 0]),
        "U": U_loc,
        "X_true": X_loc,
        "Z_true": Z_loc,
        "Y": Y_loc,
        "X0_OVERRIDE": X0_loc,
        "cfg_used": cfg_local,
    }


# ---------------------------------------------------------
# helper: build comparison synthetic matched to real record
# IMPORTANT:
#   - match current magnitude / duration with temporary globals
#   - but apply comparison shaping through a cloned cfg
# ---------------------------------------------------------
def _build_synthetic_matched_to_real(real_meta: dict):
    global SIM_T_END, I_AFTER, I_CONST

    sim_t_end_old = SIM_T_END
    i_after_old = I_AFTER
    i_const_old = I_CONST

    try:
        if MATCH_SYNTHETIC_TO_REAL_DURATION:
            if MANUAL_SYNTHETIC_DURATION is not None:
                SIM_T_END = float(MANUAL_SYNTHETIC_DURATION)
            else:
                SIM_T_END = max(float(real_meta["resampled_duration"]), T_STEP + SIM_DT)

        if AUTO_MATCH_SYNTHETIC_CURRENT_TO_REAL:
            real_amp = float(real_meta["raw_current_abs_mean"])
            if MANUAL_SYNTHETIC_CURRENT_SCALE is not None:
                real_amp *= float(MANUAL_SYNTHETIC_CURRENT_SCALE)

            if STEP_PROFILE_STYLE == "deviation_id":
                I_AFTER = real_amp
            else:
                I_CONST = real_amp

        syn = _build_current_synthetic_dataset()
        return syn

    finally:
        SIM_T_END = sim_t_end_old
        I_AFTER = i_after_old
        I_CONST = i_const_old


# ---------------------------------------------------------
# Build main dataset for the actual identification pipeline
# ---------------------------------------------------------
if SOURCE_MODE == "synthetic":
    tmp = _build_current_synthetic_dataset(cfg_local=cfg)
    t = tmp["t"]
    U = tmp["U"]
    X_true = tmp["X_true"]
    Z_true = tmp["Z_true"]
    Y = tmp["Y"]
    X0_OVERRIDE = tmp["X0_OVERRIDE"]
    TRUE_AVAILABLE = True
else:
    t, U, X_true, Z_true, Y, REAL_META_MAIN = build_real_dataset()
    X0_OVERRIDE = None
    TRUE_AVAILABLE = False

u_truth = U[:, 0].astype(np.float64)

if CURRENT_SIGN_MODE == "flip":
    u = -u_truth.copy()
else:
    u = u_truth.copy()

y = Y[:, 0].astype(np.float64)
y_base = float(y[0]) if len(y) else 0.0
y_delta = y - y_base

z_truth = None
if Z_true is not None:
    z_truth = _ensure_1d_array(Z_true[:, 0] if np.ndim(Z_true) == 2 else Z_true)

if len(t) >= 2:
    Ts = float(np.median(np.diff(t)))
else:
    Ts = float(SIM_DT if SOURCE_MODE == "synthetic" else REAL_DT)

print("\nDataset ready.")
print("  N samples (full):", len(t))
print("  Ts:", Ts)
print("  CURRENT_SIGN_MODE:", CURRENT_SIGN_MODE)
print("  STEP_PROFILE_STYLE:", STEP_PROFILE_STYLE)

# ---------------------------------------------------------
# Synthetic decomposition only available in synthetic mode
# ---------------------------------------------------------
v_ocv = None
v_eta = None
v_elyte = None
v_ohmic = None
v_dyn = None

if TRUE_AVAILABLE and SYNTHETIC_OUTPUT_MODE == "nonlinear_voltage":
    comp = compute_voltage_components(cfg, X_true, u_truth)

    v_ocv = _ensure_1d_array(comp["ocv"])
    v_eta = _ensure_1d_array(comp["eta"])
    v_elyte = _ensure_1d_array(comp["electrolyte"])
    v_ohmic = _ensure_1d_array(comp["ohmic"])
    v_dyn = _ensure_1d_array(comp["dynamic_total"])

    v_Up_pack = _ensure_1d_array(comp["Up_pack"])
    v_minus_Un_pack = _ensure_1d_array(comp["minus_Un_pack"])
    v_Up_cell = _ensure_1d_array(comp["Up_cell"])
    v_Un_cell = _ensure_1d_array(comp["Un_cell"])

    v_eta_p = _ensure_1d_array(comp["eta_p"])
    v_minus_eta_n = _ensure_1d_array(comp["minus_eta_n"])
    v_I0p = _ensure_1d_array(comp["I0p"])
    v_I0n = _ensure_1d_array(comp["I0n"])

    v_ceL = _ensure_1d_array(comp["ceL"])
    v_ceR = _ensure_1d_array(comp["ceR"])
    v_elyte_ln_ratio = _ensure_1d_array(comp["elyte_ln_ratio"])
    v_elyte_prefactor = _ensure_1d_array(comp["elyte_prefactor"])

    v_ohmic_pure = _ensure_1d_array(comp["ohmic_pure"])
    v_ohmic_film = _ensure_1d_array(comp["ohmic_film"])
    v_ohmic_elyte_res = _ensure_1d_array(comp["ohmic_elyte_res"])
    v_r_elyte_neg = _ensure_1d_array(comp["R_elyte_neg"])
    v_r_elyte_sep = _ensure_1d_array(comp["R_elyte_sep"])
    v_r_elyte_pos = _ensure_1d_array(comp["R_elyte_pos"])

    print("  Output span (absolute):", series_span(y))
    print("  Output span (delta):", series_span(y_delta))
    if z_truth is not None:
        print("  Output span (truth z):", series_span(z_truth))
    print("  Output span (dynamic-only):", series_span(v_dyn))
    print("  y_base:", y_base)
    print("  u0:", float(u_truth[0]) if len(u_truth) else np.nan)
    print("  u_truth sign preview [first 5]:", u_truth[:5])
    print("  u_used_for_ID sign preview [first 5]:", u[:5])

    if PRINT_COMPONENT_SUMMARY:
        print("\nVoltage component summary:")
        for name, arr in [
            ("OCV", v_ocv),
            ("eta", v_eta),
            ("electrolyte", v_elyte),
            ("ohmic", v_ohmic),
            ("dynamic_total", v_dyn),
        ]:
            print(
                f"  {name:12s} | min = {np.min(arr):.6f} | "
                f"max = {np.max(arr):.6f} | span = {series_span(arr):.6f} | "
                f"rms = {np.sqrt(np.mean(arr**2)):.6f}"
            )

    if PLOT_VOLTAGE_COMPONENTS:
        plot_voltage_component_decomposition(
            t, v_ocv, v_eta, v_elyte, v_ohmic,
            title=f"Voltage components [{SOURCE_MODE}]"
        )

        plot_voltage_component_decomposition_centered(
            t, v_ocv, v_eta, v_elyte, v_ohmic,
            title=f"Voltage components [centered, {SOURCE_MODE}]"
        )

        plot_voltage_component_decomposition_normalized(
            t, v_ocv, v_eta, v_elyte, v_ohmic,
            title=f"Voltage components [normalized, {SOURCE_MODE}]"
        )

# ---------------------------------------------------------
# real vs synthetic comparison diagnostics
# ---------------------------------------------------------
if COMPARE_REAL_AND_SYNTHETIC:
    print("\n" + "=" * 60)
    print("REAL vs SYNTHETIC COMPARISON DIAGNOSTICS")
    print("=" * 60)

    try:
        resolved_mpr = resolve_mpr_path(MPR_PATH)
        print("Resolved MPR path:", resolved_mpr)

        t_real_cmp, U_real_cmp, _, _, Y_real_cmp, REAL_COMPARE_META = build_real_dataset(
            cycle_index=COMPARE_REAL_INDEX,
            current_mode=REAL_COMPARE_INPUT_COLUMN_MODE,
        )
        REAL_COMPARE_CACHE = {
            "t": _ensure_1d_array(t_real_cmp),
            "u": _ensure_1d_array(U_real_cmp[:, 0]),
            "y": _ensure_1d_array(Y_real_cmp[:, 0]),
            "meta": REAL_COMPARE_META,
        }

        SYNTH_COMPARE_CACHE = _build_synthetic_matched_to_real(REAL_COMPARE_META)
        t_syn_cmp = _ensure_1d_array(SYNTH_COMPARE_CACHE["t"])
        u_syn_cmp_truth = _ensure_1d_array(SYNTH_COMPARE_CACHE["u_truth"])
        y_syn_cmp = _ensure_1d_array(SYNTH_COMPARE_CACHE["Y"][:, 0])

        if SYNTHETIC_OUTPUT_MODE == "nonlinear_voltage":
            comp_cmp = compute_voltage_components(
                SYNTH_COMPARE_CACHE["cfg_used"],
                SYNTH_COMPARE_CACHE["X_true"],
                u_syn_cmp_truth
            )
            v_cmp_eta = _ensure_1d_array(comp_cmp["eta"])
            v_cmp_elyte = _ensure_1d_array(comp_cmp["electrolyte"])
            v_cmp_ohmic = _ensure_1d_array(comp_cmp["ohmic"])
            v_cmp_dyn = _ensure_1d_array(comp_cmp["dynamic_total"])

            print("\nComparison synthetic component summary:")
            print(
                f"  eta span         : {series_span(v_cmp_eta):.6e}\n"
                f"  electrolyte span : {series_span(v_cmp_elyte):.6e}\n"
                f"  ohmic span       : {series_span(v_cmp_ohmic):.6e}\n"
                f"  dynamic span     : {series_span(v_cmp_dyn):.6e}\n"
                f"  y span           : {series_span(y_syn_cmp):.6e}"
            )

        y_real_cmp = REAL_COMPARE_CACHE["y"]

        if MATCH_SYNTHETIC_STEP_TO_REAL_START:
            t_syn_cmp = t_syn_cmp.copy()
            t_real_plot = REAL_COMPARE_CACHE["t"].copy()
        else:
            t_real_plot = REAL_COMPARE_CACHE["t"].copy()

        y_syn_shifted = y_syn_cmp - y_syn_cmp[0] + y_real_cmp[0]

        syn_span = float(np.max(y_syn_shifted) - np.min(y_syn_shifted)) if len(y_syn_shifted) else 1.0
        real_span = float(np.max(y_real_cmp) - np.min(y_real_cmp)) if len(y_real_cmp) else 1.0
        if abs(syn_span) < 1e-12:
            affine_scale = 1.0
        else:
            affine_scale = real_span / syn_span
        affine_bias = float(y_real_cmp[0] - affine_scale * y_syn_shifted[0])
        y_syn_affine = affine_scale * y_syn_shifted + affine_bias

        if PRINT_COMPARISON_SUMMARY:
            print("\nComparison alignment summary:")
            print("  affine_scale:", affine_scale)
            print("  affine_bias:", affine_bias)
            print(
                "  real_plot_time_shift:",
                float(T_STEP) if (SHIFT_REAL_COMPARE_BY_TSTEP_FOR_PLOTS and MATCH_SYNTHETIC_STEP_TO_REAL_START) else 0.0
            )

        if COMPARE_SIDE_BY_SIDE:
            plot_side_by_side_current_voltage(
                t_real_plot,
                REAL_COMPARE_CACHE["u"],
                y_real_cmp,
                t_syn_cmp,
                u_syn_cmp_truth,
                y_syn_affine,
                left_title="Real (time-shifted for plot)",
                right_title="Synthetic (full trace, affine aligned)",
                overall_title="Real vs synthetic — side by side"
            )

        plot_alignment_comparison(
            t_real_plot,
            y_real_cmp,
            t_syn_cmp,
            y_syn_cmp,
            y_syn_shifted,
            y_syn_affine,
            title="Real vs synthetic voltage alignment"
        )

        # For the alignment figure, compare the same pair used in the affine-aligned
        # overlay: real trace vs affine-aligned synthetic trace.
        real_v_align = np.asarray(y_real_cmp, dtype=np.float64).reshape(-1)
        syn_v_align = np.asarray(y_syn_affine, dtype=np.float64).reshape(-1)

        real_t_align = np.asarray(t_real_plot, dtype=np.float64).reshape(-1)
        syn_t_align = np.asarray(t_syn_cmp, dtype=np.float64).reshape(-1)

        real_v_align_norm = normalize_for_alignment_plot(real_v_align)
        syn_v_align_norm = normalize_for_alignment_plot(syn_v_align)

        plt.figure(figsize=(11, 4.5))
        plt.plot(real_t_align, real_v_align_norm, label="Real normalized")
        plt.plot(syn_t_align, syn_v_align_norm, label="Synthetic normalized")
        plt.xlabel("Time [s]")
        plt.ylabel("Normalized voltage")
        plt.title("Real vs synthetic voltage alignment — normalized")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        if PLOT_COMPARISON_WITH_SHIFTED_VOLTAGE:
            plot_overlay_real_vs_synthetic(
                t_real_plot,
                y_real_cmp,
                t_syn_cmp,
                y_syn_shifted,
                title="Real vs synthetic absolute voltage overlay (real time-shifted, start-aligned)",
                label_real="Real (time-shifted)",
                label_syn="Synthetic shifted"
            )

        plot_overlay_real_vs_synthetic(
            t_real_plot,
            y_real_cmp,
            t_syn_cmp,
            y_syn_affine,
            title="Real vs synthetic absolute voltage overlay (real time-shifted, affine aligned)",
            label_real="Real (time-shifted)",
            label_syn="Synthetic affine aligned"
        )

        if COMPARE_NORMALIZED_SHAPES:
            plot_overlay_normalized_real_vs_synthetic(
                t_real_plot,
                y_real_cmp,
                t_syn_cmp,
                y_syn_cmp,
                title="Real vs synthetic normalized voltage overlay"
            )

        if COMPARE_DELTA_SHAPES:
            plot_overlay_real_vs_synthetic(
                t_real_plot,
                y_real_cmp - y_real_cmp[0],
                t_syn_cmp,
                y_syn_affine - y_syn_affine[0],
                title="Real vs synthetic delta-voltage overlay",
                label_real="Real delta V (time-shifted)",
                label_syn="Synthetic affine-aligned delta V"
            )

    except FileNotFoundError as e:
        REAL_COMPARE_CACHE = None
        SYNTH_COMPARE_CACHE = None
        print("\nComparison block could not load the real MPR file.")
        print(str(e))
        if USE_COMPARISON_SYNTHETIC_FOR_ID:
            raise RuntimeError(
                "USE_COMPARISON_SYNTHETIC_FOR_ID=True requires the real comparison file to be found first. "
                "Fix MPR_PATH or place the file in a searchable folder."
            ) from e

# ---------------------------------------------------------
# Identification target construction
# IMPORTANT:
#   Keep both:
#     - physical current for nonlinear ID
#     - sign-adjusted current for linear OE
# ---------------------------------------------------------
if SOURCE_MODE == "synthetic" and USE_COMPARISON_SYNTHETIC_FOR_ID:
    if SYNTH_COMPARE_CACHE is None:
        raise RuntimeError(
            "USE_COMPARISON_SYNTHETIC_FOR_ID=True, but SYNTH_COMPARE_CACHE is not available. "
            "Build the comparison synthetic first before identification."
        )

    t_id_source = t_syn_cmp.copy()

    # PHYSICAL current for nonlinear model
    u_id_truth_source = u_syn_cmp_truth.copy()

    if COMPARISON_SYNTHETIC_ID_VOLTAGE == "raw":
        y_abs_source = y_syn_cmp.copy()
        id_voltage_source_name = "comparison synthetic raw voltage"
    elif COMPARISON_SYNTHETIC_ID_VOLTAGE == "shifted":
        y_abs_source = y_syn_shifted.copy()
        id_voltage_source_name = "comparison synthetic shifted voltage"
    elif COMPARISON_SYNTHETIC_ID_VOLTAGE == "affine_aligned":
        y_abs_source = y_syn_affine.copy()
        id_voltage_source_name = "comparison synthetic affine-aligned voltage"
    else:
        raise ValueError(
            f"Unsupported COMPARISON_SYNTHETIC_ID_VOLTAGE={COMPARISON_SYNTHETIC_ID_VOLTAGE!r}"
        )

    pre_step_mask = t_id_source < float(T_STEP)
    if np.any(pre_step_mask):
        y_base_source = float(np.mean(y_abs_source[pre_step_mask]))
    else:
        y_base_source = float(y_abs_source[0])

    y_delta_source = y_abs_source - y_base_source
    v_dyn_source = None

    print("\nUsing comparison synthetic for identification.")
    print("  ID voltage source:", id_voltage_source_name)
    print("  ID duration [s]:", float(t_id_source[-1]) if len(t_id_source) else np.nan)
    print("  ID samples:", len(t_id_source))
    print("  ID baseline (from selected comparison trace):", y_base_source)




else:
    t_id_source = t.copy()

    # PHYSICAL current for nonlinear model
    u_id_truth_source = u_truth.copy()

    y_abs_source = y.copy()
    y_delta_source = y_delta.copy()
    y_base_source = float(y_base)
    v_dyn_source = None if v_dyn is None else v_dyn.copy()

    print("\nUsing default main dataset for identification.")
    print("  ID samples:", len(t_id_source))



# ---------------------------------------------------------
# HARD CHECK — prove which dataset is being used for ID
# ---------------------------------------------------------
if SOURCE_MODE == "synthetic":
    if USE_COMPARISON_SYNTHETIC_FOR_ID:
        expected_id_source = "comparison"
        if "t_syn_cmp" not in globals():
            raise RuntimeError(
                "USE_COMPARISON_SYNTHETIC_FOR_ID=True, but t_syn_cmp does not exist. "
                "Run the comparison synthetic block before this cell."
            )
        if len(t_id_source) != len(t_syn_cmp):
            raise RuntimeError(
                "Expected comparison synthetic ID source, but t_id_source length does not match t_syn_cmp."
            )
    else:
        expected_id_source = "default_main"

    print("\nID SOURCE CHECK")
    print("  expected_id_source:", expected_id_source)
    print("  USE_COMPARISON_SYNTHETIC_FOR_ID:", USE_COMPARISON_SYNTHETIC_FOR_ID)
    print("  COMPARISON_SYNTHETIC_ID_VOLTAGE:", COMPARISON_SYNTHETIC_ID_VOLTAGE)
    print("  len(t_id_source):", len(t_id_source))
    print("  t_id_source range:", float(t_id_source[0]), "to", float(t_id_source[-1]))
    print("  y_abs_source range:", float(np.min(y_abs_source)), "to", float(np.max(y_abs_source)))
    print("  y_abs_source span:", float(np.max(y_abs_source) - np.min(y_abs_source)))
# ---------------------------------------------------------
# Final target selection
# ---------------------------------------------------------
if IDENTIFICATION_TARGET == "delta_voltage":
    y_target_full = y_delta_source.copy()
    y_target_name = "deviation voltage"
    y_fit_reference = 0.0
    y_modeling_full = y_delta_source.copy()

elif IDENTIFICATION_TARGET == "absolute_voltage":
    y_target_full = y_abs_source.copy()
    y_target_name = "absolute voltage"

    if CENTER_OUTPUT_FOR_OE:
        y_fit_reference = float(y_base_source)
        y_modeling_full = y_abs_source - y_fit_reference
    else:
        y_fit_reference = 0.0
        y_modeling_full = y_abs_source.copy()

elif IDENTIFICATION_TARGET == "dynamic_only":
    if v_dyn_source is None:
        raise RuntimeError(
            "IDENTIFICATION_TARGET='dynamic_only' is not supported when "
            "using the comparison affine-aligned voltage for identification."
        )
    y_target_full = v_dyn_source.copy()
    y_target_name = "dynamic-only voltage"
    y_fit_reference = 0.0
    y_modeling_full = v_dyn_source.copy()

else:
    raise ValueError(f"Unsupported IDENTIFICATION_TARGET={IDENTIFICATION_TARGET!r}")


# ---------------------------------------------------------
# Final ID arrays
# ---------------------------------------------------------
t_id_full = _ensure_1d_array(t_id_source).copy()

# Physical current used by nonlinear CT model.
u_id_truth_full = _ensure_1d_array(u_id_truth_source).copy()

# Legacy alias used by older downstream cells.
# In this nonlinear-only notebook, keep it equal to the physical CT current.
u_id_full = u_id_truth_full.copy()

y_target_full = _ensure_1d_array(y_target_full).copy()
y_modeling_full = _ensure_1d_array(y_modeling_full).copy()

if not (
    len(t_id_full)
    == len(u_id_truth_full)
    == len(u_id_full)
    == len(y_target_full)
    == len(y_modeling_full)
):
    raise RuntimeError(
        "ID array length mismatch:\n"
        f"  len(t_id_full)       = {len(t_id_full)}\n"
        f"  len(u_id_truth_full) = {len(u_id_truth_full)}\n"
        f"  len(u_id_full)       = {len(u_id_full)}\n"
        f"  len(y_target_full)   = {len(y_target_full)}\n"
        f"  len(y_modeling_full) = {len(y_modeling_full)}"
    )


# ---------------------------------------------------------
# HARD CHECK — sign and target sanity
# ---------------------------------------------------------
print("\nFINAL ID ARRAYS CHECK")
print("  len(t_id_full):", len(t_id_full))
print("  len(u_id_truth_full):", len(u_id_truth_full))
print("  len(u_id_full):", len(u_id_full))
print("  len(y_target_full):", len(y_target_full))
print("  len(y_modeling_full):", len(y_modeling_full))

print("  u_id_truth_full range:", float(np.min(u_id_truth_full)), "to", float(np.max(u_id_truth_full)))
print("  u_id_full range:", float(np.min(u_id_full)), "to", float(np.max(u_id_full)))
print("  y_target_full range:", float(np.min(y_target_full)), "to", float(np.max(y_target_full)))
print("  y_modeling_full range:", float(np.min(y_modeling_full)), "to", float(np.max(y_modeling_full)))
print("  y_modeling_full span:", float(np.max(y_modeling_full) - np.min(y_modeling_full)))

if SOURCE_MODE == "synthetic" and USE_COMPARISON_SYNTHETIC_FOR_ID:
    span_target = float(np.max(y_target_full) - np.min(y_target_full))
    span_source = float(np.max(y_abs_source) - np.min(y_abs_source))
    if not np.isclose(span_target, span_source):
        print("WARNING: y_target_full span differs from selected comparison source span.")
        print("  span_target:", span_target)
        print("  span_source:", span_source)

if np.max(np.abs(y_modeling_full)) < 1e-6:
    raise RuntimeError("y_modeling_full is almost zero. Identification target construction is broken.")

if np.allclose(u_id_truth_full, u_id_truth_full[0]):
    raise RuntimeError("u_id_truth_full is constant. Cannot identify dynamics from constant input.")

dt_id = np.diff(t_id_full)
if len(dt_id) > 0:
    print("  ID dt min:", float(np.min(dt_id)))
    print("  ID dt max:", float(np.max(dt_id)))
    print("  ID dt median:", float(np.median(dt_id)))
    print("  ID dt strictly uniform:", bool(np.allclose(dt_id, dt_id[0], rtol=0.0, atol=1e-10)))

# ---------------------------------------------------------
# NEW — optional ID-only downsampling
# ---------------------------------------------------------
if ENABLE_ID_DOWNSAMPLE:
    t_id_full, u_id_truth_full, y_target_full, y_modeling_full = downsample_id_source(
        t=t_id_full,
        u=u_id_truth_full,
        y_abs=y_target_full,
        y_modeling=y_modeling_full,
        target_dt=ID_DOWNSAMPLE_DT,
        use_interp=ID_DOWNSAMPLE_USE_INTERP,
    )

    if IDENTIFICATION_TARGET == "absolute_voltage" and CENTER_OUTPUT_FOR_OE:
        pre_step_mask_ds = t_id_full < float(T_STEP)
        if np.any(pre_step_mask_ds):
            y_fit_reference = float(np.mean(y_target_full[pre_step_mask_ds]))
        else:
            y_fit_reference = float(y_target_full[0])
        y_modeling_full = y_target_full - y_fit_reference

    if IDENTIFICATION_TARGET == "delta_voltage":
        pre_step_mask_ds = t_id_full < float(T_STEP)
        if np.any(pre_step_mask_ds):
            y_fit_reference = 0.0

    if PRINT_ID_DOWNSAMPLE_SUMMARY:
        print("\nID source downsampling applied.")
        print("  target dt [s]:", ID_DOWNSAMPLE_DT)
        print("  resulting samples:", len(t_id_full))
        if len(t_id_full) >= 2:
            print("  resulting median dt [s]:", float(np.median(np.diff(t_id_full))))

# OE current can still be sign-flipped if desired
if CURRENT_SIGN_MODE == "flip":
    u_id_full = -u_id_truth_full.copy()
else:
    u_id_full = u_id_truth_full.copy()


# %% =====================================================
# CELL 7 — Nonlinear ID settings for synthetic validation
# =====================================================
# This cell is meant to be added AFTER your existing linear
# notebook Cell 6. Keep Cells 0–6 from the linear notebook,
# but set:
#   STATE_SPACE_VARIANT = "full_14"
# for this nonlinear-ID experiment.

# Nonlinear-ID control flags
RUN_NONLINEAR_SYNTHETIC_ID = True

# We will use the already-selected ID source from the linear
# notebook logic:
#   t_id_full
#   u_id_full
#   y_target_full
#   y_fit_reference
#   IDENTIFICATION_TARGET
# These were created by your Cells 0–6.

# For nonlinear ID, we want ABSOLUTE voltage as the main target.
NONLINEAR_ID_TARGET = "absolute_voltage"

# Optional: keep only the post-step portion for nonlinear fitting
# if your custom nonlinear prep is more stable that way.
NONLINEAR_USE_POST_STEP_ONLY = False

# Optional trimming window for nonlinear fit
NONLINEAR_T_MIN = None
NONLINEAR_T_MAX = None

# If True, use the downsampled ID signal already created by the
# linear notebook. This is what you wanted.
NONLINEAR_USE_ID_SIGNAL = True

# Whether to center the nonlinear target internally
# (usually False for the nonlinear terminal-voltage fit).
NONLINEAR_CENTER_OUTPUT = True

print("NONLINEAR_CENTER_OUTPUT:", NONLINEAR_CENTER_OUTPUT)

# Surface evaluation controls
NONLINEAR_SURFACE_GRID_N = 61
NONLINEAR_SURFACE_GUARD = 1e-3

# Plot controls
PLOT_NONLINEAR_STAGE_FITS = True
PLOT_NONLINEAR_STAGE_RESIDUALS = True
PLOT_NONLINEAR_STAGE_COMPARISON = True
PLOT_NONLINEAR_SURFACES = True

print("RUN_NONLINEAR_SYNTHETIC_ID:", RUN_NONLINEAR_SYNTHETIC_ID)
print("STATE_SPACE_VARIANT (expected full_14):", STATE_SPACE_VARIANT)
print("NONLINEAR_ID_TARGET:", NONLINEAR_ID_TARGET)
print("NONLINEAR_USE_ID_SIGNAL:", NONLINEAR_USE_ID_SIGNAL)
print("NONLINEAR_USE_POST_STEP_ONLY:", NONLINEAR_USE_POST_STEP_ONLY)

# %% =====================================================
# CELL 8 — Sanity checks and nonlinear imports
# =====================================================
# This cell assumes your custom nonlinear package is available
# in your thesis/codebase environment.

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.analysis.nonlinearity import evaluate_surface_on_grid
from battery_deg_spme.analysis.parameter_extraction import extract_monitorable_parameters
from battery_deg_spme.analysis.summaries import make_stage_summary_table
from battery_deg_spme.visualization.fit_plots import (
    plot_voltage,
    plot_residuals,
    plot_stage_fit_comparison,
)

# These objects must already exist from your linear notebook Cells 0–6
required_from_linear = [
    "t_id_full",
    "u_id_full",
    "y_target_full",
    "IDENTIFICATION_TARGET",
    "STATE_SPACE_VARIANT",
]

missing = [name for name in required_from_linear if name not in globals()]
if missing:
    raise RuntimeError(
        "Missing required objects from Cells 0–6: " + ", ".join(missing)
    )

if STATE_SPACE_VARIANT != "full_14":
    print(
        "[WARNING] STATE_SPACE_VARIANT is not 'full_14'. "
        "For this first nonlinear validation run, set it to 'full_14'."
    )

print("Nonlinear imports loaded.")
print("Required linear-ID objects found.")

# %% =====================================================
# CELL 9 — Freeze the synthetic dataset for nonlinear ID
# =====================================================
# We freeze the exact synthetic-validation signal selected by
# your linear notebook. This is the whole point:
# same generated system, same matched synthetic data source,
# new identification engine.

def _ensure_1d_nl(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)

# t_nl_full = _ensure_1d_nl(t_id_full)
# u_nl_full = _ensure_1d_nl(u_id_full)
# y_nl_full = _ensure_1d_nl(y_target_full)

# # Optional masking
# mask_nl = np.ones_like(t_nl_full, dtype=bool)

# if NONLINEAR_USE_POST_STEP_ONLY:
#     mask_nl &= t_nl_full >= float(T_STEP)

# if NONLINEAR_T_MIN is not None:
#     mask_nl &= t_nl_full >= float(NONLINEAR_T_MIN)

# if NONLINEAR_T_MAX is not None:
#     mask_nl &= t_nl_full <= float(NONLINEAR_T_MAX)

# t_nl = t_nl_full[mask_nl].copy()
# u_nl = u_nl_full[mask_nl].copy()
# y_nl = y_nl_full[mask_nl].copy()

# if len(t_nl) < 10:
#     raise RuntimeError("Nonlinear synthetic dataset is too short after masking.")

# if NONLINEAR_CENTER_OUTPUT:
#     y_nl_offset = float(y_nl[0])
#     y_nl_model = y_nl - y_nl_offset
# else:
#     y_nl_offset = 0.0
#     y_nl_model = y_nl.copy()

# Ts_nl = float(np.median(np.diff(t_nl))) if len(t_nl) >= 2 else np.nan

# print("Frozen nonlinear-ID dataset:")
# print("  samples:", len(t_nl))
# print("  Ts_nl:", Ts_nl)
# print("  t range [s]:", float(t_nl[0]), "to", float(t_nl[-1]))
# print("  u range:", float(np.min(u_nl)), "to", float(np.max(u_nl)))
# print("  y range:", float(np.min(y_nl)), "to", float(np.max(y_nl)))
# print("  y_nl_offset:", y_nl_offset)

# plt.figure(figsize=(11, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t_nl, u_nl, linewidth=2)
# plt.grid(True)
# plt.ylabel("Current [A]")
# plt.title(
#     "Frozen synthetic input for nonlinear ID "
#     + ("[post-step only, re-zeroed]" if NONLINEAR_USE_POST_STEP_ONLY else "[full trace]")
# )

# plt.subplot(2, 1, 2)
# plt.plot(t_nl, y_nl, linewidth=2)
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.title("Frozen synthetic output for nonlinear ID")
# plt.tight_layout()
# plt.show()

# %% =====================================================
# CELL 10 — Build a single synthetic cycle dataframe/package
# =====================================================

# synthetic_cycle_meta = {
#     "source": "synthetic_from_linear_validation_pipeline",
#     "state_space_variant": STATE_SPACE_VARIANT,
#     "identification_target_from_linear": IDENTIFICATION_TARGET,
#     "nonlinear_target": NONLINEAR_ID_TARGET,
#     "n_samples": int(len(t_nl)),
#     "Ts": float(Ts_nl),
#     "t_step": float(T_STEP),
# }

# display(synthetic_cycle_df.head())
# print("Synthetic cycle meta:")
# for k, v in synthetic_cycle_meta.items():
#     print(f"  {k}: {v}")


# %% =====================================================
# CELL 11 — Nonlinear pipeline adapter hooks
# Purpose:
#   Reuse the package single-cycle nonlinear fitting stages
#   on the frozen synthetic dataset from the linear notebook,
#   without going through MPR loading or cycle detection.
# =====================================================
import copy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from battery_deg_spme.config.settings import get_default_settings
from battery_deg_spme.models.spme_proxy_synth import Config as NLConfig, build_proxy_signals
from battery_deg_spme.preprocessing.signal_preparation import prepare_cycle_data

# IMPORTANT: use the synthetic-only fitting functions
from battery_deg_spme.fitting.stage2_synth import fit_stage2_for_cycle_synth
from battery_deg_spme.fitting.stage3_synth import (
    fit_stage3a_for_cycle_synth,
    fit_stage3b_for_cycle_synth,
)



def setup_runtime_for_nonlinear(settings):
    import os

    n_threads = int(
        os.environ.get(
            getattr(settings.runtime, "slurm_cpus_env_var", "SLURM_CPUS_PER_TASK"),
            str(getattr(settings.runtime, "default_num_threads", N_THREADS)),
        )
    )

    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={n_threads}"
    )
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)

    if hasattr(settings, "runtime") and hasattr(settings.runtime, "jax_enable_x64"):
        if settings.runtime.jax_enable_x64:
            jax.config.update("jax_enable_x64", True)
            return n_threads, jnp.float64

    return n_threads, jnp.float64


def build_nonlinear_cfg_from_linear_truth(settings) -> NLConfig:
    """
    Build the 14-state nonlinear proxy config using the same
    main physical/scaling knobs as the linear synthetic generator.
    """
    cfg_nl = NLConfig()

    # --- core physical constants / geometry ---
    cfg_nl.R = float(cfg.R)
    cfg_nl.F = float(cfg.F)
    cfg_nl.T = float(cfg.T)
    cfg_nl.T_ref = float(cfg.T)

    cfg_nl.L1 = float(cfg.L1)
    cfg_nl.L2 = float(cfg.L2)
    cfg_nl.L3 = float(cfg.L3)

    cfg_nl.Rn = float(cfg.Rn)
    cfg_nl.Rp = float(cfg.Rp)
    cfg_nl.A = float(cfg.A)

    cfg_nl.Dn = float(cfg.Dn)
    cfg_nl.Dp = float(cfg.Dp)
    cfg_nl.De = float(cfg.De)
    cfg_nl.eps = float(cfg.eps)

    cfg_nl.kappa_n_eff = float(cfg.kappa_n_eff)
    cfg_nl.kappa_s_eff = float(cfg.kappa_s_eff)
    cfg_nl.kappa_p_eff = float(cfg.kappa_p_eff)

    cfg_nl.a_s_n = float(cfg.a_s_n)
    cfg_nl.a_s_p = float(cfg.a_s_p)

    cfg_nl.k_n0 = float(cfg.k_n0)
    cfg_nl.k_p0 = float(cfg.k_p0)

    cfg_nl.csn_max = float(cfg.csn_max)
    cfg_nl.csp_max = float(cfg.csp_max)
    cfg_nl.ce0 = float(cfg.ce0)
    cfg_nl.t_plus = float(cfg.t_plus)
    cfg_nl.k_f = float(cfg.k_f)

    # --- ohmic / film / BV / series scaling ---
    cfg_nl.R_ohm = float(cfg.R_ohm)
    cfg_nl.Rf = float(cfg.Rf)
    cfg_nl.bv_scale = float(cfg.bv_scale)
    cfg_nl.N_series = int(cfg.N_series)

    # --- conventions ---
    # nonlinear package real-data path uses negative discharge current,
    # which matches the flipped synthetic current we are freezing here

    # IMPORTANT: match the synthetic truth convention
    cfg_nl.discharge_positive = bool(cfg.discharge_positive)

    cfg_nl.ce_is_deviation = bool(cfg.ce_is_deviation)
    cfg_nl.ln_orientation = str(cfg.ln_orientation)
    cfg_nl.eta_mode = str(cfg.eta_mode)

    # --- guards / floors ---
    cfg_nl.theta_guard = float(cfg.theta_guard)
    cfg_nl.I0_floor_p = float(cfg.I0_floor_p)
    cfg_nl.I0_floor_n = float(cfg.I0_floor_n)

    # --- fixed current placeholders ---
    cfg_nl.I_dyn = float(np.max(np.abs(u_nl))) if len(u_nl) else 0.0
    cfg_nl.I_for_voltage = float(np.max(np.abs(u_nl))) if len(u_nl) else 0.0

    cfg_nl.use_solid_stoich_rate_scale = bool(USE_SOLID_STOICH_RATE_SCALE)
    cfg_nl.solid_stoich_rate_scale = float(SOLID_STOICH_RATE_SCALE)

    return cfg_nl


def run_nonlinear_single_synthetic_pipeline(
    cycle_df: pd.DataFrame,
    settings,
):
    """
    Synthetic notebook adapter.

    Uses the frozen notebook synthetic trace directly, then runs:
      - Stage 2
      - optional Stage 3a
      - optional Stage 3b

    Important behavior:
      - Stage 3a is rejected if it worsens Stage 2 by more than 5%.
      - Stage 3b still runs after Stage 3a rejection, but initializes
        from nominal / Stage 2 dynamics instead of bad Stage 3a dynamics.
      - Final model is not selected here. Cell 15 selects by RMSE.
    """
    cfg_nl = cfg

    dtype = jnp.float64

    # ---------------------------------------------------------
    # Accept either:
    #   1) time_s as a normal column, or
    #   2) time_s as the DataFrame index
    # Your current synthetic_cycle_df uses time_s as the index.
    # ---------------------------------------------------------
    if "time_s" in cycle_df.columns:
        t_values = cycle_df["time_s"].values
    else:
        t_values = cycle_df.index.to_numpy()

    required_cols = ["current_A", "voltage_V"]
    missing_cols = [c for c in required_cols if c not in cycle_df.columns]
    if missing_cols:
        raise RuntimeError(
            "synthetic_cycle_df is missing required columns: "
            + ", ".join(missing_cols)
            + f"\nAvailable columns: {list(cycle_df.columns)}"
            + f"\nIndex name: {cycle_df.index.name}"
        )

    prep = {
        "t": np.asarray(t_values, dtype=np.float64).reshape(-1),
        "u": np.asarray(cycle_df["current_A"].values, dtype=np.float64).reshape(-1, 1),
        "y": np.asarray(cycle_df["voltage_V"].values, dtype=np.float64).reshape(-1, 1),
    }

    print("\nNonlinear adapter dataframe check:")
    print("  cycle_df columns:", list(cycle_df.columns))
    print("  cycle_df index name:", cycle_df.index.name)
    print("  time source:", "column time_s" if "time_s" in cycle_df.columns else "index")
    print("  len(t):", len(prep["t"]))
    print("  t range:", float(prep["t"][0]), "to", float(prep["t"][-1]))

    print("\nNonlinear adapter prep:")
    print("  len(t):", len(prep["t"]))
    print("  u range:", float(np.min(prep["u"])), "to", float(np.max(prep["u"])))
    print("  y range:", float(np.min(prep["y"])), "to", float(np.max(prep["y"])))
    print("  y span:", float(np.max(prep["y"]) - np.min(prep["y"])))

    proxy = build_proxy_signals(
        t_np=prep["t"],
        u_np=prep["u"],
        cfg=cfg_nl,
        xn0=settings.initial_state.xn0,
        xp0=settings.initial_state.xp0,
        ce0_dev=settings.initial_state.ce0_dev,
    )

    stage2 = fit_stage2_for_cycle_synth(
        t_np=prep["t"],
        u_np=prep["u"],
        y_np=prep["y"],
        proxy=proxy,
        cfg=cfg_nl,
        settings=settings,
        dtype=dtype,
    )

    stage3a = None
    stage3b = None

    def _rmse_local(y_true, y_hat):
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)
        n = min(len(y_true), len(y_hat))
        return float(np.sqrt(np.mean((y_hat[:n] - y_true[:n]) ** 2)))

    # ---------------------------------------------------------
    # Stage 3a: run, then reject if it damages the fit
    # ---------------------------------------------------------
    if settings.experiment.run_stage3 and settings.experiment.run_stage3a:
        stage3a_candidate = fit_stage3a_for_cycle_synth(
            t_np=prep["t"],
            u_np=prep["u"],
            y_np=prep["y"],
            proxy=proxy,
            stage2_result=stage2,
            cfg=cfg_nl,
            settings=settings,
            dtype=dtype,
        )

        stage2_rmse_for_guard = _rmse_local(prep["y"], stage2["yhat"])
        stage3a_rmse_for_guard = _rmse_local(prep["y"], stage3a_candidate["yhat"])

        print("\nStage 3a guard:")
        print("  stage2_rmse:", stage2_rmse_for_guard)
        print("  stage3a_rmse:", stage3a_rmse_for_guard)

        if stage3a_rmse_for_guard > 1.05 * stage2_rmse_for_guard:
            print("  Rejecting Stage 3a because it worsened the fit.")
            stage3a = None
        else:
            print("  Accepting Stage 3a.")
            stage3a = stage3a_candidate

    # ---------------------------------------------------------
    # Stage 3b: run even if Stage 3a was rejected
    # ---------------------------------------------------------
    if settings.experiment.run_stage3 and settings.experiment.run_stage3b:
        stage3b = fit_stage3b_for_cycle_synth(
            t_np=prep["t"],
            u_np=prep["u"],
            y_np=prep["y"],
            proxy=proxy,
            stage2_result=stage2,
            stage3a_result=stage3a,
            cfg=cfg_nl,
            settings=settings,
            dtype=dtype,
        )

    # Do not choose the final stage here.
    # Cell 15 will select the final stage by actual RMSE.
    final_result = stage2
    final_stage_name = "stage2_unverified"

    return {
        "cfg": cfg_nl,
        "cycle_df": cycle_df,
        "chosen_cycle_idx": 0,
        "selection_note": "synthetic single cycle injected from nonlinear-validation notebook",
        "prep": prep,
        "proxy": proxy,
        "stage2": stage2,
        "stage3a": stage3a,
        "stage3b": stage3b,
        "final_result": final_result,
        "final_stage_name": final_stage_name,
    }


print("Cell 11 loaded.")
print("This adapter uses SYNTH functions:")
print("  - fit_stage2_for_cycle_synth(...)")
print("  - fit_stage3a_for_cycle_synth(...)")
print("  - fit_stage3b_for_cycle_synth(...)")
print("and injects the notebook synthetic trace directly.")

# %% =====================================================
# CELL 12 — Configure nonlinear settings object
# Purpose:
#   Start from package defaults, then override only what is
#   needed for this synthetic-validation notebook.
# =====================================================
settings_nl = get_default_settings()
settings_nl_original = copy.deepcopy(settings_nl)

# ---------------------------------------------------------
# Dataset / columns
# ---------------------------------------------------------
if hasattr(settings_nl, "data"):
    if hasattr(settings_nl.data, "t_col"):
        settings_nl.data.t_col = "time_s"
    if hasattr(settings_nl.data, "time_col"):
        settings_nl.data.time_col = "time_s"

    if hasattr(settings_nl.data, "i_col"):
        settings_nl.data.i_col = "current_A"
    if hasattr(settings_nl.data, "v_col"):
        settings_nl.data.v_col = "voltage_V"

    if hasattr(settings_nl.data, "force_units"):
        settings_nl.data.force_units = "A"
    if hasattr(settings_nl.data, "v_ref"):
        settings_nl.data.v_ref = "none"

    # keep the frozen notebook trace exactly as is
    if hasattr(settings_nl.data, "resample"):
        settings_nl.data.resample = False
    if hasattr(settings_nl.data, "enforce_discharge_only"):
        settings_nl.data.enforce_discharge_only = False
    if hasattr(settings_nl.data, "raw_discharge_sign"):
        settings_nl.data.raw_discharge_sign = "negative"
    if hasattr(settings_nl.data, "tmax"):
        settings_nl.data.tmax = -1.0
    if hasattr(settings_nl.data, "drop_first_n"):
        settings_nl.data.drop_first_n = 0

# ---------------------------------------------------------
# Initial state
# ---------------------------------------------------------
if hasattr(settings_nl, "initial_state"):
    if hasattr(settings_nl.initial_state, "xn0"):
        settings_nl.initial_state.xn0 = float(THETA_N0)
    if hasattr(settings_nl.initial_state, "xp0"):
        settings_nl.initial_state.xp0 = float(THETA_P0)
    if hasattr(settings_nl.initial_state, "ce0_dev"):
        settings_nl.initial_state.ce0_dev = float(CE0_DEV)

# ---------------------------------------------------------
# Surrogate controls
# ---------------------------------------------------------
NONLINEAR_SURFACE_GRID_N = 60
NONLINEAR_SURFACE_GUARD = 1e-4

if hasattr(settings_nl, "surrogate"):
    if hasattr(settings_nl.surrogate, "poly_deg"):
        settings_nl.surrogate.poly_deg = 4
    if hasattr(settings_nl.surrogate, "use_ln_feature"):
        settings_nl.surrogate.use_ln_feature = False
    if hasattr(settings_nl.surrogate, "clip_raw_z"):
        settings_nl.surrogate.clip_raw_z = 2
    if hasattr(settings_nl.surrogate, "clip_raw_a"):
        settings_nl.surrogate.clip_raw_a = 20
    if hasattr(settings_nl.surrogate, "clip_raw_b"):
        settings_nl.surrogate.clip_raw_b = 5
    if hasattr(settings_nl.surrogate, "nonlinearity_grid_n"):
        settings_nl.surrogate.nonlinearity_grid_n = int(NONLINEAR_SURFACE_GRID_N)
    if hasattr(settings_nl.surrogate, "nonlinearity_guard"):
        settings_nl.surrogate.nonlinearity_guard = float(NONLINEAR_SURFACE_GUARD)

# ---------------------------------------------------------
# Optimization
# ---------------------------------------------------------
if hasattr(settings_nl, "optimization"):
    if hasattr(settings_nl.optimization, "rho_th_stage2"):
        settings_nl.optimization.rho_th_stage2 = 5e-5
    if hasattr(settings_nl.optimization, "rho_th_stage3a"):
        settings_nl.optimization.rho_th_stage3a = 5e-2
    if hasattr(settings_nl.optimization, "rho_th_stage3b"):
        settings_nl.optimization.rho_th_stage3b = 1e-1

    if hasattr(settings_nl.optimization, "adam_epochs_stage2"):
        settings_nl.optimization.adam_epochs_stage2 = 12000
    if hasattr(settings_nl.optimization, "adam_eta_stage2"):
        settings_nl.optimization.adam_eta_stage2 = 3e-4

    if hasattr(settings_nl.optimization, "adam_epochs_stage3a"):
        settings_nl.optimization.adam_epochs_stage3a = 0
    if hasattr(settings_nl.optimization, "adam_eta_stage3a"):
        settings_nl.optimization.adam_eta_stage3a = 2e-5

    if hasattr(settings_nl.optimization, "adam_epochs_stage3b"):
        settings_nl.optimization.adam_epochs_stage3b = 0
    if hasattr(settings_nl.optimization, "adam_eta_stage3b"):
        settings_nl.optimization.adam_eta_stage3b = 1e-5

    if hasattr(settings_nl.optimization, "use_lbfgs"):
        settings_nl.optimization.use_lbfgs = True
    if hasattr(settings_nl.optimization, "lbfgs_epochs"):
        settings_nl.optimization.lbfgs_epochs = 5000


    setattr(settings_nl.optimization, "lbfgs_epochs_stage3a", 0)

    setattr(settings_nl.optimization, "lbfgs_epochs_stage3b", 0)


    # if hasattr(settings_nl.optimization, "lbfgs_epochs_stage3b"):
    #     settings_nl.optimization.lbfgs_epochs_stage3b = 100

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
if hasattr(settings_nl, "solver"):
    if hasattr(settings_nl.solver, "dt0_div"):
        settings_nl.solver.dt0_div = 5
    if hasattr(settings_nl.solver, "max_steps"):
        settings_nl.solver.max_steps = 20000
    if hasattr(settings_nl.solver, "solver_rtol"):
        settings_nl.solver.solver_rtol = 1e-8
    if hasattr(settings_nl.solver, "solver_atol"):
        settings_nl.solver.solver_atol = 1e-10

# ---------------------------------------------------------
# Runtime
# ---------------------------------------------------------
if hasattr(settings_nl, "runtime"):
    if hasattr(settings_nl.runtime, "jax_enable_x64"):
        settings_nl.runtime.jax_enable_x64 = True
    if hasattr(settings_nl.runtime, "default_num_threads"):
        settings_nl.runtime.default_num_threads = int(N_THREADS)

# ---------------------------------------------------------
# Experiment switches
# ---------------------------------------------------------
if hasattr(settings_nl, "experiment"):
    if hasattr(settings_nl.experiment, "run_stage3"):
        settings_nl.experiment.run_stage3 = True
    if hasattr(settings_nl.experiment, "run_stage3a"):
        settings_nl.experiment.run_stage3a = True
    if hasattr(settings_nl.experiment, "run_stage3b"):
        settings_nl.experiment.run_stage3b = True
    if hasattr(settings_nl.experiment, "n_series_real"):
        settings_nl.experiment.n_series_real = int(N_SERIES)

print("Configured nonlinear settings object.")
print(settings_nl)


print("Updated nonlinear settings:")
print("  poly_deg:", settings_nl.surrogate.poly_deg)
print("  use_ln_feature:", settings_nl.surrogate.use_ln_feature)
print("  clip_raw_z:", settings_nl.surrogate.clip_raw_z)
print("  clip_raw_a:", settings_nl.surrogate.clip_raw_a)
print("  clip_raw_b:", settings_nl.surrogate.clip_raw_b)

print("  Stage 2 Adam epochs:", settings_nl.optimization.adam_epochs_stage2)
print("  Stage 2 Adam eta:", settings_nl.optimization.adam_eta_stage2)
print("  Stage 2 LBFGS epochs:", settings_nl.optimization.lbfgs_epochs)
print("  Stage 2 rho:", settings_nl.optimization.rho_th_stage2)
print("  Stage 3 enabled:", getattr(settings_nl.experiment, "run_stage3", False))

print("  Stage 3a Adam epochs:", settings_nl.optimization.adam_epochs_stage3a)
print(
    "  Stage 3a LBFGS epochs:",
    getattr(settings_nl.optimization, "lbfgs_epochs_stage3a", settings_nl.optimization.lbfgs_epochs),
)

print("  Stage 3b Adam epochs:", settings_nl.optimization.adam_epochs_stage3b)
print(
    "  Stage 3b LBFGS epochs:",
    getattr(settings_nl.optimization, "lbfgs_epochs_stage3b", settings_nl.optimization.lbfgs_epochs),
)

# %% =====================================================
# CELL 12 PATCH — Safer Stage 3 / Stage 3b configuration
# Purpose:
#   Keep Stage 3 for thesis story, but make it conservative.
#
# Main changes:
#   1) Stage 3 is enabled.
#   2) Stage 3a/3b get small optimizer budgets.
#   3) Stage 3b is prevented from moving too aggressively.
#   4) R0 is allowed to stay near the good Stage 2/edge-refit value.
# =====================================================

import copy

if "settings_nl" not in globals():
    raise RuntimeError("settings_nl is missing. Run the original Cell 12 setup first.")

# Work directly on settings_nl
settings_nl.experiment.run_stage3 = True
settings_nl.experiment.run_stage3a = True
settings_nl.experiment.run_stage3b = True
settings_nl.experiment.final_stage_name = "stage3b"

# ---------------------------------------------------------
# Surrogate constraints
# ---------------------------------------------------------
# Keep the nonlinear static surface controlled.
# Your previous broad full-grid learned surface was too wild,
# so do not let Stage 3b freely reshape everything.
settings_nl.surrogate.poly_deg = 4
settings_nl.surrogate.use_ln_feature = False

# Keep Z moderately bounded.
settings_nl.surrogate.clip_raw_z = 1.25

# IMPORTANT:
# raw_a/raw_b were not the direct problem alone, but wide bounds
# can make Stage 3b move dynamics too much. These are safer than 20/5.
settings_nl.surrogate.clip_raw_a = 8.0
settings_nl.surrogate.clip_raw_b = 2.0

# ---------------------------------------------------------
# Optimizer settings
# ---------------------------------------------------------
# Stage 2: strong baseline
settings_nl.optimization.adam_epochs_stage2 = 12000
settings_nl.optimization.adam_eta_stage2 = 3e-4

# Stage 3a: small, conservative dynamics-only refinement
settings_nl.optimization.adam_epochs_stage3a = 300
settings_nl.optimization.adam_eta_stage3a = 5e-6

# Stage 3b: very small joint refinement
# Do NOT make this huge yet. Stage 3b has many coupled degrees of freedom.
settings_nl.optimization.adam_epochs_stage3b = 300
settings_nl.optimization.adam_eta_stage3b = 2e-6

# L-BFGS can be too aggressive for this coupled Stage 3b problem.
# Start with Adam-only for Stage 3. If this behaves, then test LBFGS later.
settings_nl.optimization.use_lbfgs = True
settings_nl.optimization.lbfgs_epochs = 5000

# If your package has separate Stage 3 L-BFGS fields, set them too.
# These assignments are guarded because some versions may not have them.
if hasattr(settings_nl.optimization, "lbfgs_epochs_stage3a"):
    settings_nl.optimization.lbfgs_epochs_stage3a = 0

if hasattr(settings_nl.optimization, "lbfgs_epochs_stage3b"):
    settings_nl.optimization.lbfgs_epochs_stage3b = 0

# ---------------------------------------------------------
# Regularization / trust-region style controls
# ---------------------------------------------------------
# Smaller rho means less aggressive movement away from Stage 2.
# Your previous rho_th_stage3a=0.05 and stage3b=0.1 are too loose
# for a single step trajectory with limited excitation.
settings_nl.optimization.rho_th_stage2 = 5e-5
settings_nl.optimization.rho_th_stage3a = 5e-3
settings_nl.optimization.rho_th_stage3b = 1e-3

# ---------------------------------------------------------
# Solver
# ---------------------------------------------------------
settings_nl.solver.dt0_div = 5
settings_nl.solver.max_steps = 20000
settings_nl.solver.solver_rtol = 1e-8
settings_nl.solver.solver_atol = 1e-10

print("=" * 72)
print("UPDATED SAFE STAGE 3 SETTINGS")
print("=" * 72)
print("Stage 3 enabled:", settings_nl.experiment.run_stage3)
print("Stage 3a enabled:", settings_nl.experiment.run_stage3a)
print("Stage 3b enabled:", settings_nl.experiment.run_stage3b)
print("poly_deg:", settings_nl.surrogate.poly_deg)
print("clip_raw_z:", settings_nl.surrogate.clip_raw_z)
print("clip_raw_a:", settings_nl.surrogate.clip_raw_a)
print("clip_raw_b:", settings_nl.surrogate.clip_raw_b)
print("Stage 2 Adam epochs:", settings_nl.optimization.adam_epochs_stage2)
print("Stage 3a Adam epochs:", settings_nl.optimization.adam_epochs_stage3a)
print("Stage 3b Adam epochs:", settings_nl.optimization.adam_epochs_stage3b)
print("Stage 3a Adam eta:", settings_nl.optimization.adam_eta_stage3a)
print("Stage 3b Adam eta:", settings_nl.optimization.adam_eta_stage3b)
print("rho Stage 2:", settings_nl.optimization.rho_th_stage2)
print("rho Stage 3a:", settings_nl.optimization.rho_th_stage3a)
print("rho Stage 3b:", settings_nl.optimization.rho_th_stage3b)
print("=" * 72)

# %% =====================================================
# CELL 13 — Freeze the nonlinear-ID dataset from the linear notebook
# =====================================================

t_nl = np.asarray(t_id_full, dtype=np.float64).reshape(-1)
u_nl = np.asarray(u_id_truth_full, dtype=np.float64).reshape(-1)
y_nl_abs = np.asarray(y_target_full, dtype=np.float64).reshape(-1)

print("Nonlinear current sanity check:")
print("  min(u_nl) =", float(np.min(u_nl)))
print("  max(u_nl) =", float(np.max(u_nl)))
print("  unique preview:", np.unique(np.round(u_nl[:20], 8)))

# ---------------------------------------------------------
# Center voltage using the PRE-STEP mean from the full trace
# This is the key fix for the synthetic nonlinear pipeline.
# ---------------------------------------------------------
pre_step_mask_full = t_nl < float(T_STEP)
if np.any(pre_step_mask_full):
    y_nl_baseline = float(np.mean(y_nl_abs[pre_step_mask_full]))
else:
    y_nl_baseline = float(y_nl_abs[0])

y_nl = y_nl_abs - y_nl_baseline

# Optional post-step crop
if NONLINEAR_USE_POST_STEP_ONLY:
    mask_post = t_nl >= float(T_STEP)
    t_nl = t_nl[mask_post]
    u_nl = u_nl[mask_post]
    y_nl = y_nl[mask_post]
    y_nl_abs = y_nl_abs[mask_post]

    # re-zero time for plotting clarity
    t_nl = t_nl - t_nl[0]

    if np.allclose(u_nl, u_nl[0]):
        raise RuntimeError(
            "Post-step-only nonlinear dataset has constant input after cropping. "
            "Keep NONLINEAR_USE_POST_STEP_ONLY = False for staged fitting."
        )

NONLINEAR_ID_TARGET = "centered_absolute_voltage"
NONLINEAR_CENTER_OUTPUT = True

y_nl_offset = 0.0
y_nl_model = y_nl.copy()

Ts_nl = float(np.median(np.diff(t_nl))) if len(t_nl) >= 2 else np.nan

print("Frozen nonlinear-ID dataset:")
print("  NONLINEAR_USE_POST_STEP_ONLY:", NONLINEAR_USE_POST_STEP_ONLY)
print("  samples:", len(t_nl))
print("  Ts_nl:", Ts_nl)
print("  t range [s]:", float(t_nl[0]), "to", float(t_nl[-1]))
print("  u range:", float(np.min(u_nl)), "to", float(np.max(u_nl)))
print("  abs-voltage range:", float(np.min(y_nl_abs)), "to", float(np.max(y_nl_abs)))
print("  centered-voltage range:", float(np.min(y_nl)), "to", float(np.max(y_nl)))
print("  y_nl_baseline:", y_nl_baseline)
print("  first 5 centered samples:", y_nl[:5])

plt.figure(figsize=(11, 6))
plt.subplot(2, 1, 1)
plt.plot(t_nl, u_nl, linewidth=2)
plt.grid(True)
plt.ylabel("Current [A]")
plt.title("Frozen synthetic input for nonlinear ID (physical sign)")

plt.subplot(2, 1, 2)
plt.plot(t_nl, y_nl, linewidth=2, label="Centered voltage")
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Centered voltage [V]")
plt.title(
    "Frozen synthetic output for nonlinear ID "
    + ("[post-step only]" if NONLINEAR_USE_POST_STEP_ONLY else "[full trace]")
)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# %% =====================================================
# CELL 14 — Build a single synthetic cycle dataframe/package
# =====================================================
# synthetic_cycle_df = pd.DataFrame(
#     {
#         "current_A": u_nl,
#         "voltage_V": y_nl_model,
#     },
#     index=pd.Index(t_nl, name="time_s"),
# )

synthetic_cycle_df = pd.DataFrame(
    {
        "current_A": u_nl,
        "voltage_V": y_nl,
    },
    index=pd.Index(t_nl, name="time_s"),
)

synthetic_cycle_df = synthetic_cycle_df.reset_index()

print("synthetic_cycle_df voltage preview:")
print(synthetic_cycle_df.head())
print("voltage mean =", float(synthetic_cycle_df["voltage_V"].mean()))
print("voltage min =", float(synthetic_cycle_df["voltage_V"].min()))
print("voltage max =", float(synthetic_cycle_df["voltage_V"].max()))

synthetic_cycle_meta = {
    "source": "synthetic_from_linear_validation_pipeline",
    "state_space_variant": STATE_SPACE_VARIANT,
    "identification_target_from_linear": IDENTIFICATION_TARGET,
    "nonlinear_target": NONLINEAR_ID_TARGET,
    "n_samples": int(len(t_nl)),
    "Ts": float(Ts_nl),
    "t_step": float(T_STEP),
}

display(synthetic_cycle_df.head())

print("Synthetic cycle meta:")
for k, v in synthetic_cycle_meta.items():
    print(f"  {k}: {v}")

print(type(t_id_full), len(t_id_full))
print(type(u_id_full), len(u_id_full))
print(type(y_target_full), len(y_target_full))
print("Ts =", np.median(np.diff(t_id_full)))
print("target min/max =", np.min(y_target_full), np.max(y_target_full))

# %% =====================================================
# CHECK — strict uniformity before nonlinear fit
# =====================================================
dt_check = np.diff(t_nl)
print("dt min:", np.min(dt_check))
print("dt max:", np.max(dt_check))
print("dt median:", np.median(dt_check))
print("strict uniform:", np.allclose(dt_check, dt_check[0], rtol=0.0, atol=1e-10))

if not np.allclose(dt_check, dt_check[0], rtol=0.0, atol=1e-10):
    raise RuntimeError("t_nl is not strictly equally spaced. Rebuild from Cell 6.")


v0 = float(synthetic_cycle_df["voltage_V"].iloc[0])
vpre = float(synthetic_cycle_df["voltage_V"].iloc[:20].mean())

print("Centering check:")
print("  first sample =", v0)
print("  first-20 mean =", vpre)

if not (abs(v0) < 5e-3 or abs(vpre) < 5e-3):
    raise RuntimeError(
        "synthetic_cycle_df is not centered enough for the synthetic nonlinear pipeline. "
        "Check Cell 13 baseline subtraction."
    )

# %% =====================================================
# CELL 15 — Run nonlinear staged fitting on the frozen synthetic dataset
# Purpose:
#   1) Run the synthetic nonlinear pipeline on the exact frozen ID trace.
#   2) Extract Stage 2 / Stage 3a / Stage 3b safely.
#   3) Select the final nonlinear model by actual RMSE.
#   4) Overwrite the pipeline's placeholder final stage.
# =====================================================

if not RUN_NONLINEAR_SYNTHETIC_ID:
    raise RuntimeError("RUN_NONLINEAR_SYNTHETIC_ID is False. Set it True before running Cell 15.")

required_for_cell15 = [
    "synthetic_cycle_df",
    "settings_nl",
    "run_nonlinear_single_synthetic_pipeline",
]

missing_cell15 = [name for name in required_for_cell15 if name not in globals()]
if missing_cell15:
    raise RuntimeError(
        "Missing required objects before Cell 15: " + ", ".join(missing_cell15)
    )

print("=" * 72)
print("CELL 15 — RUNNING NONLINEAR SYNTHETIC PIPELINE")
print("=" * 72)

nl_result = run_nonlinear_single_synthetic_pipeline(
    cycle_df=synthetic_cycle_df,
    settings=settings_nl,
)

cfg_nl = nl_result["cfg"]
prep_nl = nl_result["prep"]
proxy_nl = nl_result["proxy"]

stage2_nl = nl_result.get("stage2", None)
stage3a_nl = nl_result.get("stage3a", None)
stage3b_nl = nl_result.get("stage3b", None)

if stage2_nl is None:
    raise RuntimeError("Nonlinear pipeline did not return stage2. Cannot continue.")

print("\nNonlinear pipeline completed.")
print("  prep samples:", len(prep_nl["t"]))
print("  proxy state shape:", np.asarray(proxy_nl["X_proxy"]).shape)
print("  stage2 exists:", stage2_nl is not None)
print("  stage3a exists:", stage3a_nl is not None)
print("  stage3b exists:", stage3b_nl is not None)


def _stage_rmse(stage: dict, stage_name: str) -> float:
    """
    Return RMSE from a stage result.
    Falls back to direct y/yhat computation if needed.
    """
    if stage is None:
        return np.inf

    metrics = stage.get("metrics", None)
    if isinstance(metrics, dict) and "rmse" in metrics:
        val = float(metrics["rmse"])
        if np.isfinite(val):
            return val

    err_summary = stage.get("err_summary", None)
    if isinstance(err_summary, dict) and "rmse" in err_summary:
        val = float(err_summary["rmse"])
        if np.isfinite(val):
            return val

    if "yhat" in stage and "prep" in nl_result and "y" in nl_result["prep"]:
        y_true = np.asarray(nl_result["prep"]["y"], dtype=np.float64).reshape(-1)
        y_hat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
        n = min(len(y_true), len(y_hat))
        if n > 0:
            return float(np.sqrt(np.mean((y_hat[:n] - y_true[:n]) ** 2)))

    print(f"[WARNING] Could not determine RMSE for {stage_name}; assigning inf.")
    return np.inf


candidate_stages = {
    "stage2": stage2_nl,
}

if stage3a_nl is not None:
    candidate_stages["stage3a"] = stage3a_nl

if stage3b_nl is not None:
    candidate_stages["stage3b"] = stage3b_nl

candidate_rmse = {
    name: _stage_rmse(stage, name)
    for name, stage in candidate_stages.items()
}

print("\nFINAL STAGE SELECTION CHECK")
for name, val in candidate_rmse.items():
    print(f"  {name:7s} RMSE = {val:.10g}")

best_stage_name = min(candidate_rmse, key=candidate_rmse.get)
best_stage_result = candidate_stages[best_stage_name]

print("\nSelected final nonlinear stage:", best_stage_name)
print("Selected final nonlinear RMSE :", candidate_rmse[best_stage_name])

# Overwrite placeholder final stage from Cell 11 adapter.
nl_result["final_result"] = best_stage_result
nl_result["final_stage_name"] = best_stage_name
nl_result["final_stage_rmse"] = candidate_rmse[best_stage_name]
nl_result["candidate_stage_rmse"] = candidate_rmse

final_nl = best_stage_result
final_nl_stage_name = best_stage_name

print("\nPipeline final stage overwritten safely:")
print("  nl_result['final_stage_name'] =", nl_result["final_stage_name"])
print("  nl_result['final_stage_rmse'] =", nl_result["final_stage_rmse"])

print("\nStage keys:")
print("  stage2 keys:", sorted(stage2_nl.keys()))
print("  stage3a keys:", None if stage3a_nl is None else sorted(stage3a_nl.keys()))
print("  stage3b keys:", None if stage3b_nl is None else sorted(stage3b_nl.keys()))
print("  final keys:", sorted(final_nl.keys()))

# Stage summary table
stage_summary_rows = []
for name, stage in candidate_stages.items():
    yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    ytrue = np.asarray(prep_nl["y"], dtype=np.float64).reshape(-1)
    n = min(len(ytrue), len(yhat))

    score = score_fit(ytrue[:n], yhat[:n], warmup=0)

    stage_summary_rows.append(
        {
            "stage": name,
            "selected_final": bool(name == best_stage_name),
            "rmse": score["rmse"],
            "r2": score["r2"],
            "bfr_percent": score["bfr_percent"],
            "sse": score["sse"],
        }
    )

stage_summary_nl_df = (
    pd.DataFrame(stage_summary_rows)
    .sort_values("rmse")
    .reset_index(drop=True)
)

print("\nNonlinear stage summary:")
display(stage_summary_nl_df)

# Final selected fit plot
t_fit_nl = np.asarray(prep_nl["t"], dtype=np.float64).reshape(-1)
y_fit_nl = np.asarray(prep_nl["y"], dtype=np.float64).reshape(-1)
yhat_final_nl = np.asarray(final_nl["yhat"], dtype=np.float64).reshape(-1)

plot_voltage(
    t_fit_nl,
    y_fit_nl,
    yhat_final_nl,
    title=f"Final selected nonlinear fit: {final_nl_stage_name}",
)

plot_residuals(
    t_fit_nl,
    y_fit_nl,
    yhat_final_nl,
    title=f"Final selected nonlinear residuals: {final_nl_stage_name}",
)

print("=" * 72)
print("CELL 15 COMPLETE")
print("=" * 72)

# %% =====================================================
# Nonlinear fit verdict
# =====================================================

best_rmse = float(nl_result["final_stage_rmse"])
best_stage = nl_result["final_stage_name"]

ytrue = np.asarray(nl_result["prep"]["y"], dtype=np.float64).reshape(-1)
yhat = np.asarray(nl_result["final_result"]["yhat"], dtype=np.float64).reshape(-1)

n = min(len(ytrue), len(yhat))
score = score_fit(ytrue[:n], yhat[:n], warmup=0)

print("=" * 72)
print("NONLINEAR FIT VERDICT")
print("=" * 72)

print("Best stage:", best_stage)
print("RMSE:", score["rmse"])
print("R2:", score["r2"])
print("BFR [%]:", score["bfr_percent"])
print("Max abs error:", float(np.max(np.abs(yhat[:n] - ytrue[:n]))))

if "R0_hat" in nl_result["final_result"]:
    print("R0_hat:", nl_result["final_result"]["R0_hat"])

if score["rmse"] < 0.006 and score["bfr_percent"] > 80:
    print("Verdict: GOOD fit for this synthetic trajectory.")
elif score["rmse"] < 0.010 and score["bfr_percent"] > 70:
    print("Verdict: ACCEPTABLE but can still improve.")
else:
    print("Verdict: NOT GOOD ENOUGH yet. Improve Stage 2 before using Stage 3.")

print("=" * 72)

# %% =====================================================
# CELL 15B — Stage 3b acceptance diagnostics and safe final selection
# Purpose:
#   Stage 3b is a joint refinement stage. It should only be
#   accepted if it improves the actual voltage fit.
#
# Placement:
#   Put immediately AFTER Cell 15.
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "nl_result" not in globals():
    raise RuntimeError("Run Cell 15 first. nl_result is missing.")

prep_nl = nl_result["prep"]
y_true_15b = np.asarray(prep_nl["y"], dtype=np.float64).reshape(-1)
t_15b = np.asarray(prep_nl["t"], dtype=np.float64).reshape(-1)

stage_order_15b = ["stage2", "stage3a", "stage3b"]

def _stage_score_15b(stage):
    if stage is None or "yhat" not in stage:
        return {
            "rmse": np.inf,
            "r2": -np.inf,
            "bfr_percent": -np.inf,
            "sse": np.inf,
            "max_abs": np.inf,
        }

    yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(y_true_15b), len(yhat))

    y = y_true_15b[:n]
    yh = yhat[:n]
    err = yh - y

    sse = float(np.sum(err**2))
    rmse_val = float(np.sqrt(np.mean(err**2)))
    max_abs_val = float(np.max(np.abs(err)))

    denom = float(np.sum((y - np.mean(y))**2))
    r2_val = float(1.0 - sse / denom) if denom > 1e-14 else np.nan

    bfr_val = float(
        100.0 * (1.0 - np.linalg.norm(err) / (np.linalg.norm(y - np.mean(y)) + 1e-14))
    )

    return {
        "rmse": rmse_val,
        "r2": r2_val,
        "bfr_percent": bfr_val,
        "sse": sse,
        "max_abs": max_abs_val,
    }


rows_15b = []
for name in stage_order_15b:
    stage = nl_result.get(name, None)
    score = _stage_score_15b(stage)

    rows_15b.append({
        "stage": name,
        "exists": stage is not None,
        "rmse": score["rmse"],
        "r2": score["r2"],
        "bfr_percent": score["bfr_percent"],
        "sse": score["sse"],
        "max_abs": score["max_abs"],
        "R0_hat": float(stage.get("R0_hat", np.nan)) if isinstance(stage, dict) else np.nan,
    })

stage_acceptance_df_15b = (
    pd.DataFrame(rows_15b)
    .replace([np.inf, -np.inf], np.nan)
    .sort_values(["rmse", "max_abs"], na_position="last")
    .reset_index(drop=True)
)

print("=" * 72)
print("CELL 15B — STAGE ACCEPTANCE DIAGNOSTICS")
print("=" * 72)
display(stage_acceptance_df_15b)

# Best stage by RMSE
best_stage_name_15b = str(stage_acceptance_df_15b.iloc[0]["stage"])
best_stage_15b = nl_result[best_stage_name_15b]

# Explicit Stage 3b guard
stage2_score_15b = _stage_score_15b(nl_result.get("stage2", None))
stage3b_score_15b = _stage_score_15b(nl_result.get("stage3b", None))

stage3b_accept_tol = 1.02  # accept if within 2% of Stage 2 or better
stage3b_is_acceptable_15b = (
    np.isfinite(stage3b_score_15b["rmse"])
    and stage3b_score_15b["rmse"] <= stage3b_accept_tol * stage2_score_15b["rmse"]
)

print("\nStage 3b acceptance check:")
print("  Stage 2 RMSE :", stage2_score_15b["rmse"])
print("  Stage 3b RMSE:", stage3b_score_15b["rmse"])
print("  Acceptance ratio Stage3b/Stage2:",
      stage3b_score_15b["rmse"] / max(stage2_score_15b["rmse"], 1e-14))
print("  Stage 3b acceptable:", stage3b_is_acceptable_15b)

# Final selection:
#   Use best RMSE stage, but never force Stage 3b if it is worse.
nl_result["final_stage_name"] = best_stage_name_15b
nl_result["final_stage"] = best_stage_15b
nl_result["final_stage_rmse"] = float(stage_acceptance_df_15b.iloc[0]["rmse"])

print("\nSafe final selection:")
print("  final_stage_name:", nl_result["final_stage_name"])
print("  final_stage_rmse:", nl_result["final_stage_rmse"])

# Plot all stages
plt.figure(figsize=(11, 4.5))
plt.plot(t_15b, y_true_15b, linewidth=2.5, label="Frozen data")

for name in stage_order_15b:
    stage = nl_result.get(name, None)
    if stage is None or "yhat" not in stage:
        continue
    yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(t_15b), len(yhat))
    plt.plot(t_15b[:n], yhat[:n], "--", linewidth=2, label=name)

plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Stage comparison with safe Stage 3b acceptance")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# Residual plot
plt.figure(figsize=(11, 4.5))
for name in stage_order_15b:
    stage = nl_result.get(name, None)
    if stage is None or "yhat" not in stage:
        continue
    yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(y_true_15b), len(yhat))
    plt.plot(t_15b[:n], yhat[:n] - y_true_15b[:n], linewidth=2, label=f"{name} residual")

plt.axhline(0.0, linewidth=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Predicted - truth [V]")
plt.title("Residual comparison: Stage 2 vs Stage 3a vs Stage 3b")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

print("=" * 72)
print("CELL 15B COMPLETE")
print("=" * 72)

# %% =====================================================
# CELL 15C — Conservative Stage 3b retry sweep
# Purpose:
#   Try safer Stage 3b settings without trusting one run.
#
# Placement:
#   Put AFTER Cell 15B.
# =====================================================

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "settings_nl" not in globals():
    raise RuntimeError("settings_nl is missing.")

if "synthetic_cycle_df" not in globals():
    raise RuntimeError("synthetic_cycle_df is missing. Run Cell 14 first.")

if "run_nonlinear_single_synthetic_pipeline" not in globals():
    raise RuntimeError("run_nonlinear_single_synthetic_pipeline is missing.")

def _configure_stage3b_retry_settings(
    base_settings,
    clip_raw_z=1.25,
    clip_raw_a=8.0,
    clip_raw_b=2.0,
    rho3a=5e-3,
    rho3b=1e-3,
    adam3a=300,
    adam3b=300,
    eta3a=5e-6,
    eta3b=2e-6,
):
    s = copy.deepcopy(base_settings)

    s.experiment.run_stage3 = True
    s.experiment.run_stage3a = True
    s.experiment.run_stage3b = True
    s.experiment.final_stage_name = "stage3b"

    s.surrogate.poly_deg = 4
    s.surrogate.use_ln_feature = False
    s.surrogate.clip_raw_z = float(clip_raw_z)
    s.surrogate.clip_raw_a = float(clip_raw_a)
    s.surrogate.clip_raw_b = float(clip_raw_b)

    s.optimization.adam_epochs_stage2 = 12000
    s.optimization.adam_eta_stage2 = 3e-4

    s.optimization.adam_epochs_stage3a = int(adam3a)
    s.optimization.adam_epochs_stage3b = int(adam3b)
    s.optimization.adam_eta_stage3a = float(eta3a)
    s.optimization.adam_eta_stage3b = float(eta3b)

    s.optimization.rho_th_stage2 = 5e-5
    s.optimization.rho_th_stage3a = float(rho3a)
    s.optimization.rho_th_stage3b = float(rho3b)

    # Keep LBFGS off for Stage 3 if those fields exist.
    if hasattr(s.optimization, "lbfgs_epochs_stage3a"):
        s.optimization.lbfgs_epochs_stage3a = 0
    if hasattr(s.optimization, "lbfgs_epochs_stage3b"):
        s.optimization.lbfgs_epochs_stage3b = 0

    return s


def _score_result_stage(result, stage_key):
    stage = result.get(stage_key, None)
    if stage is None or "yhat" not in stage:
        return {
            "rmse": np.inf,
            "r2": -np.inf,
            "bfr_percent": -np.inf,
            "max_abs": np.inf,
            "R0_hat": np.nan,
        }

    y = np.asarray(result["prep"]["y"], dtype=np.float64).reshape(-1)
    yh = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(y), len(yh))

    y = y[:n]
    yh = yh[:n]
    err = yh - y

    sse = float(np.sum(err**2))
    rmse_val = float(np.sqrt(np.mean(err**2)))
    max_abs_val = float(np.max(np.abs(err)))

    denom = float(np.sum((y - np.mean(y))**2))
    r2_val = float(1.0 - sse / denom) if denom > 1e-14 else np.nan
    bfr_val = float(
        100.0 * (1.0 - np.linalg.norm(err) / (np.linalg.norm(y - np.mean(y)) + 1e-14))
    )

    return {
        "rmse": rmse_val,
        "r2": r2_val,
        "bfr_percent": bfr_val,
        "max_abs": max_abs_val,
        "R0_hat": float(stage.get("R0_hat", np.nan)),
    }


stage3b_retry_grid = [
    # Very conservative
    {
        "clip_raw_z": 1.00,
        "clip_raw_a": 6.0,
        "clip_raw_b": 1.5,
        "rho3a": 2e-3,
        "rho3b": 5e-4,
        "adam3a": 200,
        "adam3b": 200,
        "eta3a": 3e-6,
        "eta3b": 1e-6,
    },
    # Conservative default
    {
        "clip_raw_z": 1.25,
        "clip_raw_a": 8.0,
        "clip_raw_b": 2.0,
        "rho3a": 5e-3,
        "rho3b": 1e-3,
        "adam3a": 300,
        "adam3b": 300,
        "eta3a": 5e-6,
        "eta3b": 2e-6,
    },
    # Slightly freer dynamics
    {
        "clip_raw_z": 1.50,
        "clip_raw_a": 10.0,
        "clip_raw_b": 2.5,
        "rho3a": 1e-2,
        "rho3b": 2e-3,
        "adam3a": 400,
        "adam3b": 400,
        "eta3a": 5e-6,
        "eta3b": 2e-6,
    },
]

retry_rows = []
retry_results = {}

print("=" * 72)
print("CELL 15C — CONSERVATIVE STAGE 3B RETRY SWEEP")
print("=" * 72)

for k, cfg_try in enumerate(stage3b_retry_grid):
    print("\n" + "-" * 72)
    print(f"Running Stage 3b retry {k + 1}/{len(stage3b_retry_grid)}")
    print(cfg_try)
    print("-" * 72)

    settings_try = _configure_stage3b_retry_settings(settings_nl, **cfg_try)

    result_try = run_nonlinear_single_synthetic_pipeline(
        cycle_df=synthetic_cycle_df,
        settings=settings_try,
    )

    score2 = _score_result_stage(result_try, "stage2")
    score3a = _score_result_stage(result_try, "stage3a")
    score3b = _score_result_stage(result_try, "stage3b")

    row = {
        "trial": k,
        **cfg_try,
        "stage2_rmse": score2["rmse"],
        "stage3a_rmse": score3a["rmse"],
        "stage3b_rmse": score3b["rmse"],
        "stage2_bfr": score2["bfr_percent"],
        "stage3a_bfr": score3a["bfr_percent"],
        "stage3b_bfr": score3b["bfr_percent"],
        "stage2_max_abs": score2["max_abs"],
        "stage3a_max_abs": score3a["max_abs"],
        "stage3b_max_abs": score3b["max_abs"],
        "stage2_R0": score2["R0_hat"],
        "stage3b_R0": score3b["R0_hat"],
        "stage3b_over_stage2_rmse_ratio": score3b["rmse"] / max(score2["rmse"], 1e-14),
    }

    retry_rows.append(row)
    retry_results[k] = result_try

stage3b_retry_df = (
    pd.DataFrame(retry_rows)
    .sort_values(["stage3b_rmse", "stage3b_max_abs"])
    .reset_index(drop=True)
)

print("\nStage 3b retry summary:")
display(stage3b_retry_df)

best_retry_trial = int(stage3b_retry_df.iloc[0]["trial"])
best_stage3b_retry_result = retry_results[best_retry_trial]

print("\nBest Stage 3b retry trial:", best_retry_trial)
print(stage3b_retry_df.iloc[0])

# Compare best retry with current nl_result
base_stage2_score = _score_result_stage(nl_result, "stage2")
best_retry_stage3b_score = _score_result_stage(best_stage3b_retry_result, "stage3b")

print("\nAcceptance comparison:")
print("  Current/base Stage 2 RMSE:", base_stage2_score["rmse"])
print("  Best retry Stage 3b RMSE:", best_retry_stage3b_score["rmse"])
print("  Ratio Stage3b/Stage2:", best_retry_stage3b_score["rmse"] / max(base_stage2_score["rmse"], 1e-14))

# Only overwrite if Stage 3b is actually better or very close.
if best_retry_stage3b_score["rmse"] <= 1.02 * base_stage2_score["rmse"]:
    nl_result_stage3b_retry = best_stage3b_retry_result
    print("\nAccepted: best Stage 3b retry is close enough to Stage 2.")
else:
    nl_result_stage3b_retry = None
    print("\nRejected: Stage 3b retry is still worse than Stage 2. Keep Stage 2 / edge-refined Stage 2.")

# Plot best retry
t_retry = np.asarray(best_stage3b_retry_result["prep"]["t"], dtype=np.float64).reshape(-1)
y_retry = np.asarray(best_stage3b_retry_result["prep"]["y"], dtype=np.float64).reshape(-1)

plt.figure(figsize=(11, 4.5))
plt.plot(t_retry, y_retry, linewidth=2.5, label="Frozen data")

for stage_name in ["stage2", "stage3a", "stage3b"]:
    stage = best_stage3b_retry_result.get(stage_name, None)
    if stage is None or "yhat" not in stage:
        continue
    yhat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(t_retry), len(yhat))
    plt.plot(t_retry[:n], yhat[:n], "--", linewidth=2, label=f"best retry {stage_name}")

plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Best conservative Stage 3b retry")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

print("=" * 72)
print("CELL 15C COMPLETE")
print("=" * 72)


# %% =====================================================
# CELL 15D — Nonlinear CT-ID pole extraction and comparison
# Purpose:
#   Extract truth poles and the best available estimated poles
#   of the linear dynamic block in the nonlinear/Wiener-style model.
#
# Important:
#   - The static nonlinear surface Z(xn, xp) has no poles.
#   - Poles belong to the upstream linear block:
#         dx/dt = A x + B u
#   - Stage 2 fits Z(x) and R0 using fixed proxy dynamics.
#     Therefore Stage 2 has no newly estimated dynamic poles.
#   - Stage 3b is the attempted dynamic-refinement stage.
#     If it exposes an assembled A_hat, use eig(A_hat).
#     If it only exposes rawA_hat_stage3b, report it carefully.
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from IPython.display import display  # replaced by batch-safe display


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def _as_square_matrix(obj, name="candidate"):
    """
    Convert obj to a square finite numpy matrix if possible.
    Returns None if obj is not a square 2D matrix.
    """
    if obj is None:
        return None

    try:
        arr = np.asarray(obj, dtype=np.float64)
    except Exception:
        return None

    if arr.ndim != 2:
        return None

    if arr.shape[0] != arr.shape[1]:
        return None

    if not np.all(np.isfinite(arr)):
        print(f"[warning] {name} is square but contains non-finite values.")
        return None

    return arr


def _get_attr_or_key(obj, key):
    """
    Safely get key from either a dict-like object or an object attribute.
    """
    if obj is None:
        return None

    if isinstance(obj, dict) and key in obj:
        return obj[key]

    if hasattr(obj, key):
        return getattr(obj, key)

    return None


def _find_square_matrix_recursive(
    obj,
    preferred_names=(
        "A_hat_stage3b",
        "A_hat",
        "A_ct",
        "A_learned",
        "A_stage3b",
        "A",
    ),
):
    """
    Search for a clean assembled square matrix.

    Note:
    rawA_hat_stage3b is deliberately NOT included here because it may be
    raw optimizer parameters rather than the physical assembled A matrix.
    We inspect rawA_hat_stage3b separately later.
    """
    # Direct square matrix
    A = _as_square_matrix(obj)
    if A is not None:
        return A, "direct_object"

    # Preferred names first
    for name in preferred_names:
        val = _get_attr_or_key(obj, name)
        A = _as_square_matrix(val, name=name)
        if A is not None:
            return A, name

    # Search dict values one level deep
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Skip raw optimizer arrays here
            if "raw" in str(k).lower():
                continue

            A = _as_square_matrix(v, name=str(k))
            if A is not None:
                return A, str(k)

            if isinstance(v, dict):
                for kk, vv in v.items():
                    if "raw" in str(kk).lower():
                        continue

                    A = _as_square_matrix(vv, name=f"{k}.{kk}")
                    if A is not None:
                        return A, f"{k}.{kk}"

    # Search object __dict__ one level deep
    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if "raw" in str(k).lower():
                continue

            A = _as_square_matrix(v, name=str(k))
            if A is not None:
                return A, str(k)

    return None, None


def _continuous_pole_table(poles, label):
    """
    Build a clean continuous-time pole table.
    For real negative poles, tau = -1 / pole.
    Near-zero poles are treated as conservation/integrator-like modes.
    """
    poles = np.asarray(poles, dtype=np.complex128).reshape(-1)

    rows = []
    for i, p in enumerate(poles):
        real_part = float(np.real(p))
        imag_part = float(np.imag(p))
        mag = float(np.abs(p))

        if abs(real_part) > 1e-12 and real_part < 0:
            tau_s = float(-1.0 / real_part)
        else:
            tau_s = np.inf

        if real_part < -1e-10:
            stability_label = "stable"
        elif abs(real_part) <= 1e-10:
            stability_label = "near_zero"
        else:
            stability_label = "unstable"

        rows.append(
            {
                "source": label,
                "index": i,
                "pole_real": real_part,
                "pole_imag": imag_part,
                "pole_magnitude": mag,
                "time_constant_s": tau_s,
                "stability_label": stability_label,
            }
        )

    return pd.DataFrame(rows)


def _nearest_match_table(true_poles, estimated_poles, estimated_label):
    """
    Greedy nearest-pole matching from true poles to estimated poles.
    """
    true_poles = list(np.asarray(true_poles, dtype=np.complex128).reshape(-1))
    estimated_poles = list(np.asarray(estimated_poles, dtype=np.complex128).reshape(-1))

    rows = []
    remaining = estimated_poles.copy()

    for i, p_true in enumerate(true_poles):
        if len(remaining) == 0:
            rows.append(
                {
                    "truth_index": i,
                    "truth_pole_real": float(np.real(p_true)),
                    "truth_pole_imag": float(np.imag(p_true)),
                    "estimated_label": estimated_label,
                    "estimated_pole_real": np.nan,
                    "estimated_pole_imag": np.nan,
                    "abs_error": np.nan,
                    "relative_error": np.nan,
                }
            )
            continue

        distances = [abs(p_true - p_est) for p_est in remaining]
        j = int(np.argmin(distances))
        p_est = remaining.pop(j)

        abs_err = float(abs(p_true - p_est))
        denom = max(float(abs(p_true)), 1e-12)

        rows.append(
            {
                "truth_index": i,
                "truth_pole_real": float(np.real(p_true)),
                "truth_pole_imag": float(np.imag(p_true)),
                "estimated_label": estimated_label,
                "estimated_pole_real": float(np.real(p_est)),
                "estimated_pole_imag": float(np.imag(p_est)),
                "abs_error": abs_err,
                "relative_error": abs_err / denom,
            }
        )

    return pd.DataFrame(rows)


def _print_matrix_diagnostics(A, name):
    """
    Print quick diagnostics before eigenvalue calculation.
    """
    print(f"\n{name} diagnostics:")
    print("  shape:", A.shape)
    print("  finite:", bool(np.all(np.isfinite(A))))
    print("  min:", float(np.min(A)))
    print("  max:", float(np.max(A)))
    print("  Frobenius norm:", float(np.linalg.norm(A, ord="fro")))


print("=" * 72)
print("CELL 15D — NONLINEAR CT-ID LINEAR-BLOCK POLE EXTRACTION")
print("=" * 72)

# ---------------------------------------------------------
# 1) Truth continuous-time poles
# ---------------------------------------------------------
if "A_truth" not in globals():
    raise RuntimeError("A_truth is not available. Run the truth-model construction cells first.")

A_truth_15d = _as_square_matrix(A_truth, name="A_truth")

if A_truth_15d is None:
    raise RuntimeError("A_truth exists but is not a valid square finite matrix.")

_print_matrix_diagnostics(A_truth_15d, "A_truth")

truth_ct_poles_15d = np.linalg.eigvals(A_truth_15d)

truth_pole_df_15d = (
    _continuous_pole_table(truth_ct_poles_15d, label="truth_A")
    .sort_values(["pole_real", "pole_imag"])
    .reset_index(drop=True)
)

print("\nTruth continuous-time poles from A_truth:")
display(truth_pole_df_15d)

# ---------------------------------------------------------
# 2) Stage 2 interpretation
# ---------------------------------------------------------
print("\nStage 2 interpretation:")
print("  Stage 2 estimates thetaZ and R0 using fixed proxy dynamics.")
print("  Therefore Stage 2 does not provide newly estimated poles.")
print("  If a Stage 2/proxy A matrix is found, it is reported as fixed-proxy poles only.")

stage2_ct_poles_15d = None
stage2_pole_df_15d = None
stage2_match_df_15d = None

stage2_A_15d = None
stage2_A_source_15d = None

if "nl_result" in globals():
    stage2_obj_15d = nl_result.get("stage2", None)
    proxy_obj_15d = nl_result.get("proxy", None)

    stage2_A_15d, stage2_A_source_15d = _find_square_matrix_recursive(stage2_obj_15d)

    if stage2_A_15d is None:
        stage2_A_15d, stage2_A_source_15d = _find_square_matrix_recursive(proxy_obj_15d)

if stage2_A_15d is not None:
    _print_matrix_diagnostics(stage2_A_15d, f"Stage 2/proxy A from {stage2_A_source_15d}")

    stage2_ct_poles_15d = np.linalg.eigvals(stage2_A_15d)

    stage2_pole_df_15d = (
        _continuous_pole_table(
            stage2_ct_poles_15d,
            label=f"stage2_fixed_proxy_A_from_{stage2_A_source_15d}",
        )
        .sort_values(["pole_real", "pole_imag"])
        .reset_index(drop=True)
    )

    print("\nStage 2/proxy fixed poles:")
    display(stage2_pole_df_15d)

    stage2_match_df_15d = _nearest_match_table(
        truth_ct_poles_15d,
        stage2_ct_poles_15d,
        estimated_label="stage2_fixed_proxy_not_newly_estimated",
    )

    print("\nTruth vs Stage 2 fixed-proxy nearest pole match:")
    display(stage2_match_df_15d)

else:
    print("\nNo Stage 2/proxy square A matrix found. This is okay.")

# ---------------------------------------------------------
# 3) Stage 3b estimated dynamic poles
# ---------------------------------------------------------
stage3b_ct_poles_15d = None
stage3b_pole_df_15d = None
stage3b_match_df_15d = None

stage3b_A_15d = None
stage3b_A_source_15d = None
stage3b_A_quality_15d = "not_available"

if "nl_result" not in globals():
    print("\nnl_result is missing. Run the nonlinear pipeline first.")

else:
    stage3b_obj_15d = nl_result.get("stage3b", None)

    if stage3b_obj_15d is None:
        print("\nStage 3b result is None. No estimated dynamic poles available.")

    else:
        print("\nStage 3b object found.")

        if isinstance(stage3b_obj_15d, dict):
            print("Stage 3b keys:")
            print(sorted(stage3b_obj_15d.keys()))

        # 3a) First try clean assembled A_hat-like matrix
        stage3b_A_15d, stage3b_A_source_15d = _find_square_matrix_recursive(stage3b_obj_15d)

        if stage3b_A_15d is not None:
            stage3b_A_quality_15d = "assembled_or_clean_candidate"
            print(f"\nRecovered clean Stage 3b A matrix from: {stage3b_A_source_15d}")

        # 3b) If no clean matrix, inspect rawA_hat_stage3b carefully
        elif isinstance(stage3b_obj_15d, dict) and "rawA_hat_stage3b" in stage3b_obj_15d:
            rawA_15d = np.asarray(stage3b_obj_15d["rawA_hat_stage3b"], dtype=np.float64)

            print("\nNo clean assembled A_hat found.")
            print("Inspecting rawA_hat_stage3b:")
            print("  rawA_hat_stage3b shape:", rawA_15d.shape)
            print("  rawA_hat_stage3b ndim:", rawA_15d.ndim)

            rawA_square_15d = _as_square_matrix(rawA_15d, name="rawA_hat_stage3b")

            if rawA_square_15d is not None:
                stage3b_A_15d = rawA_square_15d
                stage3b_A_source_15d = "rawA_hat_stage3b"
                stage3b_A_quality_15d = "raw_square_candidate_needs_caution"

                print(
                    "\nUsing rawA_hat_stage3b because it is square. "
                    "Interpret this cautiously: confirm whether this raw matrix is already "
                    "the assembled continuous-time A matrix."
                )

            else:
                print(
                    "\nrawA_hat_stage3b is not a usable square matrix. "
                    "Cannot compute Stage 3b poles from it directly."
                )

        else:
            print("\nNo clean A_hat and no rawA_hat_stage3b found.")

        # 3c) Compute Stage 3b poles if a candidate A exists
        if stage3b_A_15d is not None:
            _print_matrix_diagnostics(
                stage3b_A_15d,
                f"Stage 3b A candidate from {stage3b_A_source_15d}",
            )

            stage3b_ct_poles_15d = np.linalg.eigvals(stage3b_A_15d)

            stage3b_pole_df_15d = (
                _continuous_pole_table(
                    stage3b_ct_poles_15d,
                    label=f"stage3b_estimated_A_from_{stage3b_A_source_15d}",
                )
                .sort_values(["pole_real", "pole_imag"])
                .reset_index(drop=True)
            )

            stage3b_pole_df_15d["A_source_quality"] = stage3b_A_quality_15d

            print("\nStage 3b estimated/attempted continuous-time poles:")
            display(stage3b_pole_df_15d)

            stage3b_match_df_15d = _nearest_match_table(
                truth_ct_poles_15d,
                stage3b_ct_poles_15d,
                estimated_label=f"stage3b_from_{stage3b_A_source_15d}",
            )

            print("\nTruth vs Stage 3b nearest pole match:")
            display(stage3b_match_df_15d)

# ---------------------------------------------------------
# 4) Combined pole table
# ---------------------------------------------------------
tables_to_concat_15d = [truth_pole_df_15d]

if stage2_pole_df_15d is not None:
    tables_to_concat_15d.append(stage2_pole_df_15d)

if stage3b_pole_df_15d is not None:
    tables_to_concat_15d.append(stage3b_pole_df_15d)

nonlinear_pole_summary_df_15d = pd.concat(
    tables_to_concat_15d,
    ignore_index=True,
)

print("\nCombined nonlinear pole summary:")
display(nonlinear_pole_summary_df_15d)

# ---------------------------------------------------------
# 5) Pole plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    np.real(truth_ct_poles_15d),
    np.imag(truth_ct_poles_15d),
    s=90,
    marker="o",
    label="Truth poles: eig(A_truth)",
)

if stage2_ct_poles_15d is not None:
    plt.scatter(
        np.real(stage2_ct_poles_15d),
        np.imag(stage2_ct_poles_15d),
        s=70,
        marker="x",
        label="Stage 2 fixed proxy poles",
    )

if stage3b_ct_poles_15d is not None:
    plt.scatter(
        np.real(stage3b_ct_poles_15d),
        np.imag(stage3b_ct_poles_15d),
        s=70,
        marker="^",
        label=f"Stage 3b poles from {stage3b_A_source_15d}",
    )

plt.axvline(0.0, linewidth=1)
plt.axhline(0.0, linewidth=1)
plt.grid(True, alpha=0.3)
plt.xlabel("Real part")
plt.ylabel("Imaginary part")
plt.title("Nonlinear CT-ID linear-block pole comparison")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6) Final interpretation
# ---------------------------------------------------------
print("=" * 72)
print("CELL 15D INTERPRETATION")
print("=" * 72)
print("1) The degree-4 static nonlinear surface Z(xn, xp) has no poles.")
print("2) The poles belong to the upstream linear dynamic block dx/dt = Ax + Bu.")
print("3) Truth poles are eig(A_truth).")
print("4) Stage 2 is the selected voltage-fit model but does not estimate a new A.")
print("5) Stage 3b is the attempted dynamic-refinement stage.")
print("6) If Stage 3b poles were computed from rawA_hat_stage3b, report them cautiously.")
print("7) Since Stage 3b was not accepted by RMSE, report its poles as attempted estimated poles, not final accepted poles.")
print("=" * 72)

# %% =====================================================
# CELL 16A — Stage 2 sweep helper definitions
# Purpose:
#   Define the helper functions needed by Cell 16R.
#   Run this after Cell 15 and before loading/resuming the sweep.
# =====================================================

import copy
import os
import pickle
import pandas as pd
import numpy as np

def _as_bool(x) -> bool:
    """
    Robust bool conversion for values loaded from CSV/PKL/pandas/numpy.

    Handles:
      - Python bool: True / False
      - NumPy bool_: np.True_ / np.False_
      - integers/floats: 1/0
      - strings: 'True'/'False', 'yes'/'no', '1'/'0'
      - missing values: NaN/None -> False
    """
    if x is None:
        return False

    # Handles Python bool and NumPy bool_.
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    # Handles pandas NA / np.nan safely.
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass

    if isinstance(x, (int, np.integer)):
        return bool(int(x))

    if isinstance(x, (float, np.floating)):
        if not np.isfinite(float(x)):
            return False
        return bool(int(x))

    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f", ""):
            return False

    raise ValueError(f"Cannot convert to bool: {x!r} of type {type(x)}")

def _make_synthetic_cycle_df_for_dt(
    target_dt: float,
    use_interp: bool = False,
) -> pd.DataFrame:
    """
    Rebuild the nonlinear synthetic cycle dataframe from the full ID arrays,
    then downsample to target_dt.

    Requires:
      - t_id_full
      - u_id_truth_full
      - y_target_full
      - y_modeling_full
      - downsample_id_source
      - NONLINEAR_CENTER_OUTPUT
    """
    required = [
        "t_id_full",
        "u_id_truth_full",
        "y_target_full",
        "y_modeling_full",
        "downsample_id_source",
        "NONLINEAR_CENTER_OUTPUT",
    ]

    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Missing required objects for _make_synthetic_cycle_df_for_dt: "
            + ", ".join(missing)
            + "\nRun Cells 0–15 before Cell 16A."
        )

    t_ds, u_ds, y_abs_ds, y_model_ds = downsample_id_source(
        t=t_id_full,
        u=u_id_truth_full,
        y_abs=y_target_full,
        y_modeling=y_modeling_full,
        target_dt=float(target_dt),
        use_interp=bool(use_interp),
    )

    if NONLINEAR_CENTER_OUTPUT:
        y_nl = y_model_ds.copy()
    else:
        y_nl = y_abs_ds.copy()

    df = pd.DataFrame(
        {
            "time_s": np.asarray(t_ds, dtype=np.float64).reshape(-1),
            "current_A": np.asarray(u_ds, dtype=np.float64).reshape(-1),
            "voltage_V": np.asarray(y_nl, dtype=np.float64).reshape(-1),
        }
    )

    return df


def _configure_stage2_only_settings(
    poly_deg: int,
    clip_raw_z: float,
    adam_epochs: int = 8000,
    lbfgs_epochs: int = 2500,
):
    """
    Clean Stage-2-only settings.
    Stage 3 is disabled here so we can find the best Stage 2 baseline.
    """
    if "settings_nl" not in globals():
        raise RuntimeError("settings_nl is missing. Run Cell 12 before Cell 16A.")

    s = copy.deepcopy(settings_nl)

    # Surrogate
    s.surrogate.poly_deg = int(poly_deg)
    s.surrogate.use_ln_feature = False
    s.surrogate.clip_raw_z = float(clip_raw_z)
    s.surrogate.clip_raw_a = 20
    s.surrogate.clip_raw_b = 5

    # Solver
    if hasattr(s, "solver"):
        s.solver.dt0_div = 5.0
        s.solver.max_steps = 20000
        s.solver.solver_rtol = 1e-8
        s.solver.solver_atol = 1e-10

    # Stage 2 optimizer
    s.optimization.adam_epochs_stage2 = int(adam_epochs)
    s.optimization.adam_eta_stage2 = 3e-4
    s.optimization.rho_th_stage2 = 5e-5
    s.optimization.use_lbfgs = True
    s.optimization.lbfgs_epochs = int(lbfgs_epochs)

    # Stage 3 OFF
    s.optimization.adam_epochs_stage3a = 0
    s.optimization.adam_epochs_stage3b = 0
    setattr(s.optimization, "lbfgs_epochs_stage3a", 0)
    setattr(s.optimization, "lbfgs_epochs_stage3b", 0)

    if hasattr(s, "experiment"):
        s.experiment.run_stage3 = False
        s.experiment.run_stage3a = False
        s.experiment.run_stage3b = False

    return s


def _score_stage_result(result: dict, stage_key: str = "stage2") -> dict:
    """
    Score a stage result using the same score_fit function used elsewhere.
    """
    if stage_key not in result:
        raise KeyError(f"{stage_key!r} not found in result keys: {list(result.keys())}")

    stage = result[stage_key]

    y_true = np.asarray(result["prep"]["y"], dtype=np.float64).reshape(-1)
    y_hat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)

    n = min(len(y_true), len(y_hat))
    score = score_fit(y_true[:n], y_hat[:n], warmup=0)

    return {
        "rmse": score["rmse"],
        "r2": score["r2"],
        "bfr_percent": score["bfr_percent"],
        "max_abs": float(np.max(np.abs(y_hat[:n] - y_true[:n]))),
        "R0_hat": float(stage.get("R0_hat", np.nan)),
        "thetaZ_l2": float(
            np.linalg.norm(
                np.asarray(stage["thetaZ_hat"], dtype=np.float64).reshape(-1)
            )
        ),
    }


print("Cell 16A helper definitions loaded:")
print("  _make_synthetic_cycle_df_for_dt:", callable(_make_synthetic_cycle_df_for_dt))
print("  _configure_stage2_only_settings:", callable(_configure_stage2_only_settings))
print("  _score_stage_result:", callable(_score_stage_result))

# # %% =====================================================
# # CELL 16B — Load partial sweep results after restart
# # Robust version: PKL first, CSV fallback
# # =====================================================

# import os
# import pickle
# import pandas as pd

# if os.path.exists("partial_stage2_sweep_rows.pkl"):
#     with open("partial_stage2_sweep_rows.pkl", "rb") as f:
#         stage2_sweep_rows = pickle.load(f)

#     print("Loaded partial sweep rows from PKL:", len(stage2_sweep_rows))

# elif os.path.exists("partial_stage2_sweep_results.csv"):
#     partial_sweep_df = pd.read_csv("partial_stage2_sweep_results.csv")
#     stage2_sweep_rows = partial_sweep_df.to_dict("records")

#     print("Loaded partial sweep rows from CSV:", len(stage2_sweep_rows))

# else:
#     stage2_sweep_rows = []
#     print("No partial sweep file found.")

# stage2_sweep_df = (
#     pd.DataFrame(stage2_sweep_rows)
#     .sort_values(["rmse", "max_abs"])
#     .reset_index(drop=True)
# ) if len(stage2_sweep_rows) > 0 else pd.DataFrame()

# display(stage2_sweep_df)


# # %% =====================================================
# # %% =====================================================
# # CELL 16R — Resume Stage 2 sweep without repeating completed cases
# # =====================================================

# required_for_16r = [
#     "_make_synthetic_cycle_df_for_dt",
#     "_configure_stage2_only_settings",
#     "_score_stage_result",
#     "_as_bool",
#     "run_nonlinear_single_synthetic_pipeline",
# ]

# missing_for_16r = [name for name in required_for_16r if name not in globals()]

# if missing_for_16r:
#     raise RuntimeError(
#         "Missing required objects before Cell 16R: "
#         + ", ".join(missing_for_16r)
#         + "\nRun Cell 16A first."
#     )

# if "stage2_sweep_rows" not in globals():
#     stage2_sweep_rows = []

# if "stage2_sweep_results" not in globals():
#     stage2_sweep_results = {}

# completed_keys = set()

# for row in stage2_sweep_rows:
#     completed_keys.add(
#         (
#             float(row["dt"]),
#             _as_bool(row["use_interp"]),
#             int(row["poly_deg"]),
#             float(row["clip_raw_z"]),
#         )
#     )

# print("Completed sweep cases:", len(completed_keys))

# for dt_try in [10.0, 5.0, 2.0]:
#     for use_interp_try in [False]:
#         df_try = _make_synthetic_cycle_df_for_dt(
#             target_dt=dt_try,
#             use_interp=use_interp_try,
#         )

#         for deg_try in [3, 4, 5]:
#             for clip_z_try in [2.0, 3.0]:
#                 key = (float(dt_try), bool(use_interp_try), int(deg_try), float(clip_z_try))

#                 if key in completed_keys:
#                     print("Skipping completed:", key)
#                     continue

#                 print("=" * 72)
#                 print(
#                     f"STAGE 2 SWEEP: dt={dt_try}, interp={use_interp_try}, "
#                     f"poly_deg={deg_try}, clip_raw_z={clip_z_try}"
#                 )
#                 print("=" * 72)

#                 settings_try = _configure_stage2_only_settings(
#                     poly_deg=deg_try,
#                     clip_raw_z=clip_z_try,
#                     adam_epochs=8000,
#                     lbfgs_epochs=2500,
#                 )

#                 result_try = run_nonlinear_single_synthetic_pipeline(
#                     cycle_df=df_try,
#                     settings=settings_try,
#                 )

#                 metrics_try = _score_stage_result(result_try, stage_key="stage2")

#                 row = {
#                     "dt": dt_try,
#                     "use_interp": use_interp_try,
#                     "poly_deg": deg_try,
#                     "clip_raw_z": clip_z_try,
#                     "n_samples": len(result_try["prep"]["t"]),
#                     **metrics_try,
#                 }

#                 stage2_sweep_rows.append(row)
#                 stage2_sweep_results[key] = result_try
#                 completed_keys.add(key)

#                 pd.DataFrame(stage2_sweep_rows).to_csv(
#                     "partial_stage2_sweep_results.csv",
#                     index=False,
#                 )

#                 with open("partial_stage2_sweep_rows.pkl", "wb") as f:
#                     pickle.dump(stage2_sweep_rows, f)

# stage2_sweep_df = (
#     pd.DataFrame(stage2_sweep_rows)
#     .sort_values(["rmse", "max_abs"])
#     .reset_index(drop=True)
# )

# print("\nStage 2 sweep summary:")
# display(stage2_sweep_df)

# # %% =====================================================
# # CELL 16 — Focused Stage 2 refinement sweep
# # Purpose:
# #   Improve Stage 2 first with a manageable number of runs.
# #   Stage 3 stays OFF here.
# # =====================================================

# import copy

# def _make_synthetic_cycle_df_for_dt(target_dt: float, use_interp: bool = False) -> pd.DataFrame:
#     """
#     Rebuild the nonlinear synthetic cycle dataframe from the full ID arrays,
#     then downsample to target_dt.
#     """
#     t_ds, u_ds, y_abs_ds, y_model_ds = downsample_id_source(
#         t=t_id_full,
#         u=u_id_truth_full,
#         y_abs=y_target_full,
#         y_modeling=y_modeling_full,
#         target_dt=float(target_dt),
#         use_interp=bool(use_interp),
#     )

#     y_nl = y_model_ds.copy() if NONLINEAR_CENTER_OUTPUT else y_abs_ds.copy()

#     return pd.DataFrame(
#         {
#             "time_s": np.asarray(t_ds, dtype=np.float64).reshape(-1),
#             "current_A": np.asarray(u_ds, dtype=np.float64).reshape(-1),
#             "voltage_V": np.asarray(y_nl, dtype=np.float64).reshape(-1),
#         }
#     )


# def _configure_stage2_only_settings(
#     poly_deg: int,
#     clip_raw_z: float,
#     adam_epochs: int = 8000,
#     lbfgs_epochs: int = 2500,
# ):
#     """
#     Clean Stage-2-only settings.
#     Stage 3 is disabled here so we can find the best baseline.
#     """
#     s = copy.deepcopy(settings_nl)

#     # Surrogate
#     s.surrogate.poly_deg = int(poly_deg)
#     s.surrogate.use_ln_feature = False
#     s.surrogate.clip_raw_z = float(clip_raw_z)
#     s.surrogate.clip_raw_a = 20
#     s.surrogate.clip_raw_b = 5

#     # Solver
#     if hasattr(s, "solver"):
#         s.solver.dt0_div = 5.0
#         s.solver.max_steps = 20000
#         s.solver.solver_rtol = 1e-8
#         s.solver.solver_atol = 1e-10

#     # Stage 2 optimizer
#     s.optimization.adam_epochs_stage2 = int(adam_epochs)
#     s.optimization.adam_eta_stage2 = 3e-4
#     s.optimization.rho_th_stage2 = 5e-5
#     s.optimization.use_lbfgs = True
#     s.optimization.lbfgs_epochs = int(lbfgs_epochs)

#     # Stage 3 OFF
#     s.optimization.adam_epochs_stage3a = 0
#     s.optimization.adam_epochs_stage3b = 0
#     setattr(s.optimization, "lbfgs_epochs_stage3a", 0)
#     setattr(s.optimization, "lbfgs_epochs_stage3b", 0)

#     if hasattr(s, "experiment"):
#         s.experiment.run_stage3 = False
#         s.experiment.run_stage3a = False
#         s.experiment.run_stage3b = False

#     return s


# def _score_stage_result(result: dict, stage_key: str = "stage2") -> dict:
#     stage = result[stage_key]

#     y_true = np.asarray(result["prep"]["y"], dtype=np.float64).reshape(-1)
#     y_hat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)

#     n = min(len(y_true), len(y_hat))
#     score = score_fit(y_true[:n], y_hat[:n], warmup=0)

#     return {
#         "rmse": score["rmse"],
#         "r2": score["r2"],
#         "bfr_percent": score["bfr_percent"],
#         "max_abs": float(np.max(np.abs(y_hat[:n] - y_true[:n]))),
#         "R0_hat": float(stage.get("R0_hat", np.nan)),
#         "thetaZ_l2": float(
#             np.linalg.norm(
#                 np.asarray(stage["thetaZ_hat"], dtype=np.float64).reshape(-1)
#             )
#         ),
#     }


# stage2_sweep_rows = []
# stage2_sweep_results = {}

# # Focused first pass.
# # This should be much faster than the full 72-run sweep.
# for dt_try in [10.0, 5.0, 2.0]:
#     for use_interp_try in [False]:
#         df_try = _make_synthetic_cycle_df_for_dt(
#             target_dt=dt_try,
#             use_interp=use_interp_try,
#         )

#         for deg_try in [3, 4, 5]:
#             for clip_z_try in [2.0, 3.0]:
#                 print("=" * 72)
#                 print(
#                     f"STAGE 2 SWEEP: dt={dt_try}, interp={use_interp_try}, "
#                     f"poly_deg={deg_try}, clip_raw_z={clip_z_try}"
#                 )
#                 print("=" * 72)

#                 settings_try = _configure_stage2_only_settings(
#                     poly_deg=deg_try,
#                     clip_raw_z=clip_z_try,
#                     adam_epochs=8000,
#                     lbfgs_epochs=2500,
#                 )

#                 result_try = run_nonlinear_single_synthetic_pipeline(
#                     cycle_df=df_try,
#                     settings=settings_try,
#                 )

#                 metrics_try = _score_stage_result(result_try, stage_key="stage2")

#                 row = {
#                     "dt": dt_try,
#                     "use_interp": use_interp_try,
#                     "poly_deg": deg_try,
#                     "clip_raw_z": clip_z_try,
#                     "n_samples": len(result_try["prep"]["t"]),
#                     **metrics_try,
#                 }

#                 stage2_sweep_rows.append(row)
#                 stage2_sweep_results[(dt_try, use_interp_try, deg_try, clip_z_try)] = result_try

# stage2_sweep_df = (
#     pd.DataFrame(stage2_sweep_rows)
#     .sort_values(["rmse", "max_abs"])
#     .reset_index(drop=True)
# )

# print("\nStage 2 refinement sweep summary:")
# display(stage2_sweep_df)

# best_row = stage2_sweep_df.iloc[0]
# best_stage2_key = (
#     float(best_row["dt"]),
#     _as_bool(best_row["use_interp"]),
#     int(best_row["poly_deg"]),
#     float(best_row["clip_raw_z"]),
# )

# best_stage2_result = stage2_sweep_results[best_stage2_key]

# print("=" * 72)
# print("BEST STAGE 2 RESULT")
# print("=" * 72)
# print(best_row)
# print("=" * 72)

# plot_voltage(
#     best_stage2_result["prep"]["t"],
#     best_stage2_result["prep"]["y"],
#     best_stage2_result["stage2"]["yhat"],
#     title=(
#         "Best Stage 2 fit: "
#         f"dt={best_row['dt']}, interp={best_row['use_interp']}, "
#         f"deg={best_row['poly_deg']}, clip_z={best_row['clip_raw_z']}"
#     ),
# )

# plot_residuals(
#     best_stage2_result["prep"]["t"],
#     best_stage2_result["prep"]["y"],
#     best_stage2_result["stage2"]["yhat"],
#     title="Best Stage 2 residuals",
# )

# # %% =====================================================
# # CELL 17 — Stage 3 refinement from best Stage 2
# # Purpose:
# #   Use the best Stage 2 settings from Cell 16, then test whether
# #   Stage 3a/3b can improve it.
# # =====================================================

# if "stage2_sweep_df" not in globals() or len(stage2_sweep_df) == 0:
#     raise RuntimeError("Run Cell 16 first. stage2_sweep_df is missing.")

# best_row = stage2_sweep_df.iloc[0]

# best_dt = float(best_row["dt"])
# best_use_interp = _as_bool(best_row["use_interp"])
# best_deg = int(best_row["poly_deg"])
# best_clip_z = float(best_row["clip_raw_z"])

# print("=" * 72)
# print("STAGE 3 TEST FROM BEST STAGE 2")
# print("=" * 72)
# print("best_dt:", best_dt)
# print("best_use_interp:", best_use_interp)
# print("best_deg:", best_deg)
# print("best_clip_z:", best_clip_z)
# print("best Stage 2 RMSE:", float(best_row["rmse"]))
# print("=" * 72)

# synthetic_cycle_df_best = _make_synthetic_cycle_df_for_dt(
#     target_dt=best_dt,
#     use_interp=best_use_interp,
# )

# settings_stage3 = _configure_stage2_only_settings(
#     poly_deg=best_deg,
#     clip_raw_z=best_clip_z,
#     adam_epochs=12000,
#     lbfgs_epochs=5000,
# )

# # Turn Stage 3 ON, but keep it very gentle.
# settings_stage3.optimization.adam_epochs_stage3a = 0
# settings_stage3.optimization.adam_epochs_stage3b = 0

# setattr(settings_stage3.optimization, "lbfgs_epochs_stage3a", 50)
# setattr(settings_stage3.optimization, "lbfgs_epochs_stage3b", 50)

# settings_stage3.optimization.rho_th_stage3a = 1e-1
# settings_stage3.optimization.rho_th_stage3b = 2e-1

# if hasattr(settings_stage3, "experiment"):
#     settings_stage3.experiment.run_stage3 = True
#     settings_stage3.experiment.run_stage3a = True
#     settings_stage3.experiment.run_stage3b = True

# stage3_refine_result = run_nonlinear_single_synthetic_pipeline(
#     cycle_df=synthetic_cycle_df_best,
#     settings=settings_stage3,
# )

# stage_candidates = {
#     "stage2": stage3_refine_result["stage2"],
# }

# if stage3_refine_result.get("stage3a", None) is not None:
#     stage_candidates["stage3a"] = stage3_refine_result["stage3a"]

# if stage3_refine_result.get("stage3b", None) is not None:
#     stage_candidates["stage3b"] = stage3_refine_result["stage3b"]

# rows = []
# y_true = np.asarray(stage3_refine_result["prep"]["y"], dtype=np.float64).reshape(-1)

# for name, stage in stage_candidates.items():
#     y_hat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
#     n = min(len(y_true), len(y_hat))
#     score = score_fit(y_true[:n], y_hat[:n], warmup=0)

#     rows.append(
#         {
#             "stage": name,
#             "rmse": score["rmse"],
#             "r2": score["r2"],
#             "bfr_percent": score["bfr_percent"],
#             "max_abs": float(np.max(np.abs(y_hat[:n] - y_true[:n]))),
#             "R0_hat": float(stage.get("R0_hat", np.nan)),
#         }
#     )

# stage3_refine_df = (
#     pd.DataFrame(rows)
#     .sort_values(["rmse", "max_abs"])
#     .reset_index(drop=True)
# )

# print("Stage 3 refinement summary:")
# display(stage3_refine_df)

# best_stage_name = str(stage3_refine_df.loc[0, "stage"])
# best_stage = stage_candidates[best_stage_name]

# print("=" * 72)
# print("BEST FINAL MODEL")
# print("=" * 72)
# print("best_stage_name:", best_stage_name)
# print("best_rmse:", float(stage3_refine_df.loc[0, "rmse"]))
# print("best_bfr:", float(stage3_refine_df.loc[0, "bfr_percent"]))
# print("best_max_abs:", float(stage3_refine_df.loc[0, "max_abs"]))
# print("=" * 72)

# plot_voltage(
#     stage3_refine_result["prep"]["t"],
#     stage3_refine_result["prep"]["y"],
#     best_stage["yhat"],
#     title=f"Best final fit after Stage 3 refinement: {best_stage_name}",
# )

# plot_residuals(
#     stage3_refine_result["prep"]["t"],
#     stage3_refine_result["prep"]["y"],
#     best_stage["yhat"],
#     title=f"Best final residuals: {best_stage_name}",
# )


# %% =====================================================
# CELL 17B — Plot best refined nonlinear result
# Purpose:
#   Plot the result from Cell 17 if available.
#   Falls back to Cell 15 only if Cell 17 has not been run.
# =====================================================

def _plot_stage_fit(t, y, yhat, title):
    plt.figure(figsize=(11, 4.5))
    plt.plot(t, y, linewidth=2, label="Frozen synthetic data")
    plt.plot(t, yhat, "--", linewidth=2, label="Model fit")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _plot_stage_residual(t, y, yhat, title):
    err = np.asarray(yhat).reshape(-1) - np.asarray(y).reshape(-1)

    plt.figure(figsize=(11, 4.0))
    plt.plot(t, err, linewidth=2)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Pred - truth [V]")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# Prefer the refined Cell 17 result.
if "stage3_refine_result" in globals():
    plot_result = stage3_refine_result
    print("Cell 17B plotting source: stage3_refine_result from Cell 17")
elif "best_stage2_result" in globals():
    plot_result = best_stage2_result
    print("Cell 17B plotting source: best_stage2_result from Cell 16")
else:
    plot_result = nl_result
    print("Cell 17B plotting source: nl_result from Cell 15")

t_plot_nl = np.asarray(plot_result["prep"]["t"], dtype=np.float64).reshape(-1)
y_plot_nl = np.asarray(plot_result["prep"]["y"], dtype=np.float64).reshape(-1)

stage2_plot = plot_result.get("stage2", None)
stage3a_plot = plot_result.get("stage3a", None)
stage3b_plot = plot_result.get("stage3b", None)

if stage2_plot is None:
    raise RuntimeError("Plot result has no stage2. Cannot plot.")

PLOT_STAGE_FITS_POST_STEP_ONLY = False

if PLOT_STAGE_FITS_POST_STEP_ONLY:
    plot_mask = t_plot_nl >= float(T_STEP)
else:
    plot_mask = np.ones_like(t_plot_nl, dtype=bool)

t_plot_show = t_plot_nl[plot_mask]
y_plot_show = y_plot_nl[plot_mask]

# Individual plots
_plot_stage_fit(
    t_plot_show,
    y_plot_show,
    np.asarray(stage2_plot["yhat"]).reshape(-1)[plot_mask],
    "Refined nonlinear Stage 2 fit",
)

_plot_stage_residual(
    t_plot_show,
    y_plot_show,
    np.asarray(stage2_plot["yhat"]).reshape(-1)[plot_mask],
    "Refined nonlinear Stage 2 residuals",
)

if stage3a_plot is not None:
    _plot_stage_fit(
        t_plot_show,
        y_plot_show,
        np.asarray(stage3a_plot["yhat"]).reshape(-1)[plot_mask],
        "Refined nonlinear Stage 3a fit",
    )

    _plot_stage_residual(
        t_plot_show,
        y_plot_show,
        np.asarray(stage3a_plot["yhat"]).reshape(-1)[plot_mask],
        "Refined nonlinear Stage 3a residuals",
    )

if stage3b_plot is not None:
    _plot_stage_fit(
        t_plot_show,
        y_plot_show,
        np.asarray(stage3b_plot["yhat"]).reshape(-1)[plot_mask],
        "Refined nonlinear Stage 3b fit",
    )

    _plot_stage_residual(
        t_plot_show,
        y_plot_show,
        np.asarray(stage3b_plot["yhat"]).reshape(-1)[plot_mask],
        "Refined nonlinear Stage 3b residuals",
    )

# Overlay plot
plt.figure(figsize=(11, 4.5))
plt.plot(t_plot_show, y_plot_show, linewidth=2.5, label="Frozen synthetic data")

plt.plot(
    t_plot_show,
    np.asarray(stage2_plot["yhat"]).reshape(-1)[plot_mask],
    "--",
    linewidth=2,
    label="Stage 2",
)

if stage3a_plot is not None:
    plt.plot(
        t_plot_show,
        np.asarray(stage3a_plot["yhat"]).reshape(-1)[plot_mask],
        "--",
        linewidth=2,
        label="Stage 3a",
    )

if stage3b_plot is not None:
    plt.plot(
        t_plot_show,
        np.asarray(stage3b_plot["yhat"]).reshape(-1)[plot_mask],
        "--",
        linewidth=2,
        label="Stage 3b",
    )

plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Best refined nonlinear staged fits")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Summary table for the plotted result
rows = []
stage_candidates_plot = {"stage2": stage2_plot}

if stage3a_plot is not None:
    stage_candidates_plot["stage3a"] = stage3a_plot

if stage3b_plot is not None:
    stage_candidates_plot["stage3b"] = stage3b_plot

for name, stage in stage_candidates_plot.items():
    y_hat = np.asarray(stage["yhat"], dtype=np.float64).reshape(-1)
    n = min(len(y_plot_nl), len(y_hat))
    score = score_fit(y_plot_nl[:n], y_hat[:n], warmup=0)

    rows.append(
        {
            "stage": name,
            "rmse": score["rmse"],
            "r2": score["r2"],
            "bfr_percent": score["bfr_percent"],
            "max_abs": float(np.max(np.abs(y_hat[:n] - y_plot_nl[:n]))),
            "R0_hat": float(stage.get("R0_hat", np.nan)),
        }
    )

plot_summary_df = (
    pd.DataFrame(rows)
    .sort_values(["rmse", "max_abs"])
    .reset_index(drop=True)
)

print("Cell 17B plotted-result summary:")
display(plot_summary_df)


# %% =====================================================
# CELL 18 — Edge-weighted Stage 2 refinement
# Purpose:
#   Improve the sharp post-step voltage drop that Stage 2
#   slightly under-captures.
#
# Idea:
#   Stage 2 uses:
#       Vhat = (Z - Z0) - N_series * I * R0
#
#   The immediate step edge is mostly controlled by R0.
#   This cell searches R0 with extra weight near the first step,
#   then solves the static surrogate coefficients by weighted LS.
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from battery_deg_spme.models.parameterization_synth import raw_from_pos


def _get_refined_base_result():
    """
    Prefer the refined Cell 17 result.
    Fall back safely if Cell 17 was not run.
    """
    if "stage3_refine_result" in globals():
        return stage3_refine_result, "stage3_refine_result"
    if "best_stage2_result" in globals():
        return best_stage2_result, "best_stage2_result"
    if "nl_result" in globals():
        return nl_result, "nl_result"
    raise RuntimeError("No nonlinear result found. Run Cell 15 or Cell 17 first.")


def _find_step_index_from_input(u: np.ndarray) -> int:
    """
    Detect first current step from the input signal.
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    u0 = float(np.median(u[:max(3, min(10, len(u)))]))
    du = np.abs(u - u0)

    thresh = max(1e-9, 0.05 * max(float(np.max(np.abs(u))), 1e-12))
    idx = np.where(du > thresh)[0]

    if len(idx) == 0:
        return 0

    return int(idx[0])


def _score_candidate(y_true: np.ndarray, y_hat: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(y_hat, dtype=np.float64).reshape(-1)

    n = min(len(y_true), len(y_hat))
    y_true = y_true[:n]
    y_hat = y_hat[:n]

    err = y_hat - y_true
    sse = float(np.sum(err ** 2))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs": float(np.max(np.abs(err))),
        "r2": float(1.0 - sse / denom) if denom > 0 else np.nan,
        "bfr_percent": float(
            100.0 * (1.0 - np.linalg.norm(err) / np.linalg.norm(y_true - np.mean(y_true)))
        ) if denom > 0 else np.nan,
    }


def edge_weighted_stage2_refit(
    result: dict,
    edge_window_s: float = 30.0,
    edge_weight: float = 20.0,
    first_edge_points: int = 3,
    first_edge_weight: float = 60.0,
    r0_min: float = 1e-6,
    r0_max: float = 0.012,
    n_r0_grid: int = 220,
    ridge_theta: float = 1e-8,
):
    """
    Edge-weighted Stage 2 refinement.

    This does not change the state trajectory. It re-solves the static
    surrogate coefficients and R0 using the already-built Stage 2 feature
    matrix Phi.

    It is useful when the overall tail is good but the instantaneous
    current-step voltage drop is slightly missed.
    """
    stage2 = result["stage2"]

    t = np.asarray(result["prep"]["t"], dtype=np.float64).reshape(-1)
    u = np.asarray(result["prep"]["u"], dtype=np.float64).reshape(-1)
    y = np.asarray(result["prep"]["y"], dtype=np.float64).reshape(-1)

    Phi = np.asarray(stage2["Phi"], dtype=np.float64)

    if Phi.shape[0] != len(y):
        raise RuntimeError(
            f"Phi length mismatch: Phi rows={Phi.shape[0]}, y length={len(y)}"
        )

    # Anchor the static surface at the first sample, matching the Stage 2 model.
    Phi_c = Phi - Phi[0:1, :]

    k_step = _find_step_index_from_input(u)
    t_step_est = float(t[k_step])

    weights = np.ones_like(y, dtype=np.float64)

    # Emphasize the first decay region after the current step.
    edge_mask = (t >= t_step_est) & (t <= t_step_est + float(edge_window_s))
    weights[edge_mask] *= float(edge_weight)

    # Strongly emphasize the first few samples after the step.
    k1 = min(len(weights), k_step + int(first_edge_points))
    weights[k_step:k1] *= float(first_edge_weight)

    # Keep the pre-step flat region controlled too.
    pre_mask = t < t_step_est
    weights[pre_mask] *= 5.0

    # Normalize weights to avoid changing numerical scale too much.
    weights = weights / max(float(np.mean(weights)), 1e-12)

    sqrt_w = np.sqrt(weights).reshape(-1, 1)

    # Search R0. For each R0, solve thetaZ by weighted ridge:
    #   y + N*I*R0 = Phi_c @ thetaZ
    r0_grid = np.linspace(float(r0_min), float(r0_max), int(n_r0_grid))

    best = None

    A = Phi_c
    A_w = sqrt_w * A

    # Ridge matrix for thetaZ.
    ridge_A = np.sqrt(float(ridge_theta)) * np.eye(A.shape[1], dtype=np.float64)

    for r0_try in r0_grid:
        y_eff = y + float(cfg.N_series) * u * float(r0_try)

        b_w = sqrt_w.reshape(-1) * y_eff

        A_aug = np.vstack([A_w, ridge_A])
        b_aug = np.concatenate([b_w, np.zeros(A.shape[1], dtype=np.float64)])

        theta_try, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)

        yhat_try = A @ theta_try - float(cfg.N_series) * u * float(r0_try)

        # Use weighted objective for selection.
        err = yhat_try - y
        weighted_rmse = float(np.sqrt(np.mean(weights * err ** 2)))

        score = _score_candidate(y, yhat_try)

        item = {
            "r0": float(r0_try),
            "thetaZ": theta_try.astype(np.float64),
            "yhat": yhat_try.astype(np.float64),
            "weighted_rmse": weighted_rmse,
            **score,
        }

        if best is None or item["weighted_rmse"] < best["weighted_rmse"]:
            best = item

    if best is None:
        raise RuntimeError("Edge-weighted Stage 2 refinement failed.")

    # Build a stage2-like result dictionary so we can compare and optionally
    # feed it into Stage 3b.
    stage2_edge = dict(stage2)
    stage2_edge["thetaZ_hat"] = best["thetaZ"]
    stage2_edge["R0_hat"] = float(best["r0"])
    stage2_edge["rawR0_hat"] = float(raw_from_pos(float(best["r0"]), floor=1e-12))
    stage2_edge["yhat"] = best["yhat"].reshape(-1, 1)
    stage2_edge["edge_refit"] = {
        "k_step": int(k_step),
        "t_step_est": float(t_step_est),
        "edge_window_s": float(edge_window_s),
        "edge_weight": float(edge_weight),
        "first_edge_points": int(first_edge_points),
        "first_edge_weight": float(first_edge_weight),
        "weighted_rmse": float(best["weighted_rmse"]),
    }

    return stage2_edge, best, weights


base_result_for_edge, base_source_name = _get_refined_base_result()

print("=" * 72)
print("EDGE-WEIGHTED STAGE 2 REFINEMENT")
print("=" * 72)
print("Base result source:", base_source_name)

stage2_base = base_result_for_edge["stage2"]

y_true_edge = np.asarray(base_result_for_edge["prep"]["y"], dtype=np.float64).reshape(-1)
y_stage2_base = np.asarray(stage2_base["yhat"], dtype=np.float64).reshape(-1)

base_score = _score_candidate(y_true_edge, y_stage2_base)

print("Base Stage 2:")
print("  RMSE:", base_score["rmse"])
print("  BFR [%]:", base_score["bfr_percent"])
print("  max_abs:", base_score["max_abs"])
print("  R0_hat:", float(stage2_base.get("R0_hat", np.nan)))

stage2_edge, stage2_edge_info, edge_weights = edge_weighted_stage2_refit(
    base_result_for_edge,
    edge_window_s=35.0,
    edge_weight=25.0,
    first_edge_points=4,
    first_edge_weight=80.0,
    r0_min=1e-6,
    r0_max=0.012,
    n_r0_grid=260,
    ridge_theta=1e-8,
)

edge_score = _score_candidate(
    y_true_edge,
    np.asarray(stage2_edge["yhat"], dtype=np.float64).reshape(-1),
)

print("\nEdge-refined Stage 2:")
print("  RMSE:", edge_score["rmse"])
print("  BFR [%]:", edge_score["bfr_percent"])
print("  max_abs:", edge_score["max_abs"])
print("  R0_hat:", float(stage2_edge["R0_hat"]))
print("  weighted_rmse:", float(stage2_edge_info["weighted_rmse"]))
print("  step index:", stage2_edge["edge_refit"]["k_step"])
print("  estimated step time:", stage2_edge["edge_refit"]["t_step_est"])

edge_compare_df = pd.DataFrame(
    [
        {
            "stage": "stage2_base",
            "rmse": base_score["rmse"],
            "bfr_percent": base_score["bfr_percent"],
            "max_abs": base_score["max_abs"],
            "R0_hat": float(stage2_base.get("R0_hat", np.nan)),
        },
        {
            "stage": "stage2_edge_refit",
            "rmse": edge_score["rmse"],
            "bfr_percent": edge_score["bfr_percent"],
            "max_abs": edge_score["max_abs"],
            "R0_hat": float(stage2_edge["R0_hat"]),
        },
    ]
).sort_values(["rmse", "max_abs"]).reset_index(drop=True)

display(edge_compare_df)

t_edge = np.asarray(base_result_for_edge["prep"]["t"], dtype=np.float64).reshape(-1)

plt.figure(figsize=(11, 4.5))
plt.plot(t_edge, y_true_edge, linewidth=2.5, label="Frozen synthetic data")
plt.plot(t_edge, y_stage2_base, "--", linewidth=2, label="Stage 2 base")
plt.plot(
    t_edge,
    np.asarray(stage2_edge["yhat"]).reshape(-1),
    "--",
    linewidth=2,
    label="Stage 2 edge-refit",
)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Stage 2 edge-refinement comparison")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Zoom around the current step
k_step_edge = int(stage2_edge["edge_refit"]["k_step"])
t_step_edge = float(stage2_edge["edge_refit"]["t_step_est"])

zoom_mask = (t_edge >= t_step_edge - 20.0) & (t_edge <= t_step_edge + 80.0)

plt.figure(figsize=(11, 4.5))
plt.plot(t_edge[zoom_mask], y_true_edge[zoom_mask], linewidth=2.5, label="Frozen synthetic data")
plt.plot(t_edge[zoom_mask], y_stage2_base[zoom_mask], "--", linewidth=2, label="Stage 2 base")
plt.plot(
    t_edge[zoom_mask],
    np.asarray(stage2_edge["yhat"]).reshape(-1)[zoom_mask],
    "--",
    linewidth=2,
    label="Stage 2 edge-refit",
)
plt.grid(True)
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
plt.title("Zoom: immediate post-step edge")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()




# # %% =====================================================
# # CELL 18 — Build learned-surface and truth-surface functions
# # Purpose:
# #   Compare the learned nonlinear surface against the truth
# #   equilibrium/nonlinear surface used by the synthetic generator.
# # =====================================================
# from battery_deg_spme.analysis.nonlinearity import (
#     evaluate_surface_on_grid,
#     compare_surfaces_on_grid,
#     compute_shape_drift,
# )

# def build_learned_surface_fn(thetaZ_hat, zhat_from_thetaZ, cfg_local, state_template):
#     thetaZ_hat = np.asarray(thetaZ_hat, dtype=np.float64).reshape(-1)
#     state_template = np.asarray(state_template, dtype=np.float64).reshape(-1)

#     def learned_surface_fn(xn: float, xp: float) -> float:
#         x = state_template.copy()
#         x[3] = float(xn) * float(cfg_local.csn_max)
#         x[7] = float(xp) * float(cfg_local.csp_max)
#         return float(zhat_from_thetaZ(x, thetaZ_hat))

#     return learned_surface_fn


# # learned surface from final stage
# final_thetaZ_hat = np.asarray(final_nl["thetaZ_hat"], dtype=np.float64).reshape(-1)
# final_state_template = np.asarray(final_nl["xhat"][0], dtype=np.float64).reshape(-1)
# zhat_from_thetaZ_nl = stage2_nl["zhat_from_thetaZ"]

# learned_surface_fn_nl = build_learned_surface_fn(
#     thetaZ_hat=final_thetaZ_hat,
#     zhat_from_thetaZ=zhat_from_thetaZ_nl,
#     cfg_local=cfg_nl,
#     state_template=final_state_template,
# )

# # truth surface from linear synthetic generator
# # compare to truth_z_from_state(...), because the learned Z surface
# # is the equilibrium/nonlinear term before ohmic IR subtraction
# truth_surface_state_template = np.asarray(final_state_template, dtype=np.float64).copy()

# truth_surface_ref = float(truth_z_from_state(truth_surface_state_template, cfg, 0.0))

# def truth_surface_fn_nl(xn: float, xp: float) -> float:
#     x = truth_surface_state_template.copy()
#     x[3] = float(xn) * float(cfg.csn_max)
#     x[7] = float(xp) * float(cfg.csp_max)
#     return float(truth_z_from_state(x, cfg, 0.0) - truth_surface_ref)

# print("Built learned and truth surface functions.")



# # %% =====================================================
# # CELL 19 — Stage 3b test from edge-refined Stage 2
# # Purpose:
# #   Feed the edge-refined Stage 2 result into Stage 3b.
# #   Accept Stage 3b only if it beats the edge-refined Stage 2.
# # =====================================================

# from battery_deg_spme.fitting.stage3_synth import fit_stage3b_for_cycle_synth

# if "stage2_edge" not in globals():
#     raise RuntimeError("Run Cell 18 first. stage2_edge is missing.")

# if "base_result_for_edge" not in globals():
#     raise RuntimeError("Run Cell 18 first. base_result_for_edge is missing.")

# settings_stage3_edge = copy.deepcopy(settings_nl)

# # Keep Stage 3 gentle.
# settings_stage3_edge.optimization.adam_epochs_stage3a = 0
# settings_stage3_edge.optimization.adam_epochs_stage3b = 0

# setattr(settings_stage3_edge.optimization, "lbfgs_epochs_stage3a", 0)
# setattr(settings_stage3_edge.optimization, "lbfgs_epochs_stage3b", 80)

# settings_stage3_edge.optimization.rho_th_stage3b = 5e-1

# if hasattr(settings_stage3_edge, "experiment"):
#     settings_stage3_edge.experiment.run_stage3 = True
#     settings_stage3_edge.experiment.run_stage3a = False
#     settings_stage3_edge.experiment.run_stage3b = True

# print("=" * 72)
# print("STAGE 3B FROM EDGE-REFINED STAGE 2")
# print("=" * 72)
# print("Stage 3b Adam epochs:", settings_stage3_edge.optimization.adam_epochs_stage3b)
# print("Stage 3b LBFGS epochs:", getattr(settings_stage3_edge.optimization, "lbfgs_epochs_stage3b", None))
# print("Stage 3b rho_th:", settings_stage3_edge.optimization.rho_th_stage3b)
# print("=" * 72)

# stage3b_from_edge = fit_stage3b_for_cycle_synth(
#     t_np=base_result_for_edge["prep"]["t"],
#     u_np=base_result_for_edge["prep"]["u"],
#     y_np=base_result_for_edge["prep"]["y"],
#     proxy=base_result_for_edge["proxy"],
#     stage2_result=stage2_edge,
#     stage3a_result=None,
#     cfg=base_result_for_edge["cfg"],
#     settings=settings_stage3_edge,
#     dtype=jnp.float64,
# )

# y_true = np.asarray(base_result_for_edge["prep"]["y"], dtype=np.float64).reshape(-1)
# y_edge = np.asarray(stage2_edge["yhat"], dtype=np.float64).reshape(-1)
# y_3b_edge = np.asarray(stage3b_from_edge["yhat"], dtype=np.float64).reshape(-1)

# score_edge = _score_candidate(y_true, y_edge)
# score_3b_edge = _score_candidate(y_true, y_3b_edge)

# stage3_edge_compare_df = pd.DataFrame(
#     [
#         {
#             "stage": "stage2_edge_refit",
#             "rmse": score_edge["rmse"],
#             "bfr_percent": score_edge["bfr_percent"],
#             "max_abs": score_edge["max_abs"],
#             "R0_hat": float(stage2_edge["R0_hat"]),
#         },
#         {
#             "stage": "stage3b_from_edge",
#             "rmse": score_3b_edge["rmse"],
#             "bfr_percent": score_3b_edge["bfr_percent"],
#             "max_abs": score_3b_edge["max_abs"],
#             "R0_hat": float(stage3b_from_edge.get("R0_hat", np.nan)),
#         },
#     ]
# ).sort_values(["rmse", "max_abs"]).reset_index(drop=True)

# display(stage3_edge_compare_df)

# best_edge_stage_name = str(stage3_edge_compare_df.loc[0, "stage"])

# if best_edge_stage_name == "stage3b_from_edge":
#     final_edge_stage = stage3b_from_edge
#     print("ACCEPTING Stage 3b from edge-refined Stage 2.")
# else:
#     final_edge_stage = stage2_edge
#     print("KEEPING edge-refined Stage 2. Stage 3b did not improve it.")

# plt.figure(figsize=(11, 4.5))
# plt.plot(t_edge, y_true, linewidth=2.5, label="Frozen synthetic data")
# plt.plot(t_edge, y_edge, "--", linewidth=2, label="Stage 2 edge-refit")
# plt.plot(t_edge, y_3b_edge, "--", linewidth=2, label="Stage 3b from edge-refit")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.title("Stage 3b test from edge-refined Stage 2")
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(11, 4.5))
# plt.plot(t_edge[zoom_mask], y_true[zoom_mask], linewidth=2.5, label="Frozen synthetic data")
# plt.plot(t_edge[zoom_mask], y_edge[zoom_mask], "--", linewidth=2, label="Stage 2 edge-refit")
# plt.plot(t_edge[zoom_mask], y_3b_edge[zoom_mask], "--", linewidth=2, label="Stage 3b from edge-refit")
# plt.grid(True)
# plt.xlabel("Time [s]")
# plt.ylabel("Voltage [V]")
# plt.title("Zoom: Stage 3b from edge-refined Stage 2")
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()

# %% =====================================================
# CELL 19A — Build learned/truth surface functions safely
# Purpose:
#   Build surface functions after the final nonlinear fit exists.
#   This cell does not rerun fitting.
# =====================================================

from battery_deg_spme.analysis.nonlinearity import (
    evaluate_surface_on_grid,
    compare_surfaces_on_grid,
)

def _select_surface_source_result():
    """
    Choose the best available nonlinear result for surface plotting.
    Priority:
      1) final_edge_stage from Cell 19 edge/Stage3 test, if available
      2) stage2_edge from Cell 18 edge-refit, if available
      3) nl_result['final_result'] from Cell 15
      4) nl_result['stage2'] from Cell 15
    """
    if "final_edge_stage" in globals():
        return final_edge_stage, base_result_for_edge, "final_edge_stage"

    if "stage2_edge" in globals():
        return stage2_edge, base_result_for_edge, "stage2_edge"

    if "nl_result" in globals() and isinstance(nl_result, dict):
        if "final_result" in nl_result and nl_result["final_result"] is not None:
            return nl_result["final_result"], nl_result, "nl_result['final_result']"

        if "stage2" in nl_result and nl_result["stage2"] is not None:
            return nl_result["stage2"], nl_result, "nl_result['stage2']"

    raise RuntimeError(
        "No fitted nonlinear stage found. Run Cell 15 first, or Cell 18/19 if using edge refinement."
    )


def build_learned_surface_fn(thetaZ_hat, zhat_from_thetaZ, cfg_local, state_template):
    thetaZ_hat = np.asarray(thetaZ_hat, dtype=np.float64).reshape(-1)
    state_template = np.asarray(state_template, dtype=np.float64).reshape(-1)

    def learned_surface_fn(xn: float, xp: float) -> float:
        x = state_template.copy()

        # full_14 surface indices
        x[3] = float(xn) * float(cfg_local.csn_max)
        x[7] = float(xp) * float(cfg_local.csp_max)

        return float(zhat_from_thetaZ(x, thetaZ_hat))

    return learned_surface_fn


surface_stage_nl, surface_result_nl, surface_source_name = _select_surface_source_result()

print("=" * 72)
print("SURFACE SOURCE")
print("=" * 72)
print("surface_source_name:", surface_source_name)
print("available stage keys:", list(surface_stage_nl.keys()))
print("=" * 72)

cfg_nl_surface = surface_result_nl.get("cfg", cfg)

thetaZ_surface = np.asarray(surface_stage_nl["thetaZ_hat"], dtype=np.float64).reshape(-1)

# zhat_from_thetaZ is usually carried by Stage 2.
# Stage 3b may not carry the callable, so fall back to Stage 2.
if "zhat_from_thetaZ" in surface_stage_nl:
    zhat_from_thetaZ_surface = surface_stage_nl["zhat_from_thetaZ"]
elif "stage2" in surface_result_nl and "zhat_from_thetaZ" in surface_result_nl["stage2"]:
    zhat_from_thetaZ_surface = surface_result_nl["stage2"]["zhat_from_thetaZ"]
elif "nl_result" in globals() and "stage2" in nl_result and "zhat_from_thetaZ" in nl_result["stage2"]:
    zhat_from_thetaZ_surface = nl_result["stage2"]["zhat_from_thetaZ"]
else:
    raise RuntimeError("Could not find zhat_from_thetaZ for learned surface.")

if "xhat" in surface_stage_nl and surface_stage_nl["xhat"] is not None:
    surface_state_template = np.asarray(surface_stage_nl["xhat"][0], dtype=np.float64).reshape(-1)
elif "proxy" in surface_result_nl and "X_proxy" in surface_result_nl["proxy"]:
    surface_state_template = np.asarray(surface_result_nl["proxy"]["X_proxy"][0], dtype=np.float64).reshape(-1)
else:
    raise RuntimeError("Could not find state template for surface evaluation.")

learned_surface_fn_nl = build_learned_surface_fn(
    thetaZ_hat=thetaZ_surface,
    zhat_from_thetaZ=zhat_from_thetaZ_surface,
    cfg_local=cfg_nl_surface,
    state_template=surface_state_template,
)

# Truth surface: compare learned Z surface to truth_z_from_state centered
# at the same template state.
truth_surface_state_template = np.asarray(surface_state_template, dtype=np.float64).copy()

if "truth_z_from_state" not in globals():
    raise RuntimeError("truth_z_from_state is missing. Run earlier truth-model cells first.")

truth_surface_ref = float(truth_z_from_state(truth_surface_state_template, cfg, 0.0))

def truth_surface_fn_nl(xn: float, xp: float) -> float:
    x = truth_surface_state_template.copy()

    # full_14 surface indices
    x[3] = float(xn) * float(cfg.csn_max)
    x[7] = float(xp) * float(cfg.csp_max)

    return float(truth_z_from_state(x, cfg, 0.0) - truth_surface_ref)

print("Built learned_surface_fn_nl and truth_surface_fn_nl safely.")

# %% =====================================================
# CELL 19B — Surface evaluation and comparison
# =====================================================

grid_n = (
    settings_nl.surrogate.nonlinearity_grid_n
    if hasattr(settings_nl, "surrogate") and hasattr(settings_nl.surrogate, "nonlinearity_grid_n")
    else 60
)

grid_guard = (
    settings_nl.surrogate.nonlinearity_guard
    if hasattr(settings_nl, "surrogate") and hasattr(settings_nl.surrogate, "nonlinearity_guard")
    else 1e-4
)

surface_learned_nl = evaluate_surface_on_grid(
    surface_fn=learned_surface_fn_nl,
    n_per_axis=grid_n,
    guard=grid_guard,
)

surface_truth_nl = evaluate_surface_on_grid(
    surface_fn=truth_surface_fn_nl,
    n_per_axis=grid_n,
    guard=grid_guard,
)

surface_compare_nl = compare_surfaces_on_grid(
    ref_surface_fn=truth_surface_fn_nl,
    learned_surface_fn=learned_surface_fn_nl,
    n_per_axis=grid_n,
    guard=grid_guard,
    title="truth_vs_learned_nonlinearity",
)

print("Surface comparison metrics:")
print("  RMSE   :", surface_compare_nl.rmse)
print("  MAE    :", surface_compare_nl.mae)
print("  max_abs:", surface_compare_nl.max_abs)
print("  drift  :", surface_compare_nl.drift)

# %% =====================================================
# CELL 19B — Supervisor diagnostic: state ranges and truth nonlinearity on visited range
# Purpose:
#   Diagnose why the learned nonlinear surface can differ from truth.
#
#   This cell answers:
#     1) What range of states did the experiment actually visit?
#     2) What are the learned proxy xp/xn ranges?
#     3) What are the truth xp/xn ranges under the same input?
#     4) What does the TRUE nonlinearity look like ONLY inside
#        the visited xp/xn region?
#     5) Does the learned trajectory only need to match a narrow
#        slice of the nonlinear surface?
#
# Placement:
#   Put this AFTER Cell 19A, because Cell 19A builds:
#     - truth_surface_fn_nl
#     - learned_surface_fn_nl
#     - surface_stage_nl
#     - surface_result_nl
#     - cfg_nl_surface
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ct
# from IPython.display import display  # replaced by batch-safe display


# ---------------------------------------------------------
# 19B.0 — Guard checks
# ---------------------------------------------------------
required_names_19b = [
    "surface_stage_nl",
    "surface_result_nl",
    "cfg_nl_surface",
    "truth_surface_fn_nl",
    "learned_surface_fn_nl",
    "S_truth",
    "make_x0_truth",
    "STATE_SPACE_VARIANT",
    "THETA_N0",
    "THETA_P0",
    "CE0_DEV",
]

missing_19b = [name for name in required_names_19b if name not in globals()]
if missing_19b:
    raise RuntimeError(
        "Cell 19B needs Cell 19A and earlier truth-model cells to run first.\n"
        f"Missing objects: {missing_19b}"
    )

if "xhat" not in surface_stage_nl or surface_stage_nl["xhat"] is None:
    raise RuntimeError("surface_stage_nl has no xhat. Run the nonlinear fit cell first.")


# ---------------------------------------------------------
# 19B.1 — Helper functions
# ---------------------------------------------------------
def _arr1(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _arr2(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def _safe_span(x):
    x = _arr1(x)
    if len(x) == 0:
        return np.nan
    return float(np.nanmax(x) - np.nanmin(x))


def _safe_min(x):
    x = _arr1(x)
    return float(np.nanmin(x)) if len(x) else np.nan


def _safe_max(x):
    x = _arr1(x)
    return float(np.nanmax(x)) if len(x) else np.nan


def _normalize_for_plot(x, eps=1e-12):
    x = _arr1(x)
    xc = x - x[0]
    denom = np.nanmax(np.abs(xc))
    if not np.isfinite(denom) or denom < eps:
        return np.zeros_like(xc)
    return xc / denom


def _get_prep_from_surface_result(surface_result):
    """
    Pull the same time/input used by the final nonlinear fitting result.
    """
    if isinstance(surface_result, dict) and "prep" in surface_result:
        prep = surface_result["prep"]
    elif "nl_result" in globals() and isinstance(nl_result, dict) and "prep" in nl_result:
        prep = nl_result["prep"]
    else:
        raise RuntimeError("Could not find prep dictionary containing t and u.")

    if "t" not in prep:
        raise RuntimeError("prep dictionary has no key 't'.")

    if "u" in prep:
        u_key = "u"
    elif "current_A" in prep:
        u_key = "current_A"
    else:
        raise RuntimeError("prep dictionary has no key 'u' or 'current_A'.")

    return prep, _arr1(prep["t"]), _arr1(prep[u_key])


def _simulate_truth_states_on_id_grid(t, u):
    """
    Simulate the original truth state-space model on the same frozen ID grid.

    This gives the TRUE state path visited by the same input used for nonlinear ID.
    """
    x0_truth = make_x0_truth(
        cfg_nl_surface,
        STATE_SPACE_VARIANT,
        theta_n0=THETA_N0,
        theta_p0=THETA_P0,
        ce0=CE0_DEV,
    )

    resp = ct.forced_response(
        S_truth,
        T=t,
        U=u,
        X0=x0_truth,
        return_x=True,
    )

    # python-control versions differ slightly, so handle both safely.
    if hasattr(resp, "states"):
        X = np.asarray(resp.states, dtype=np.float64)
    elif isinstance(resp, tuple) and len(resp) >= 3:
        X = np.asarray(resp[2], dtype=np.float64)
    else:
        raise RuntimeError("Could not extract truth states from forced_response output.")

    # control returns states as (n_states, n_time); convert to (n_time, n_states)
    if X.shape[0] != len(t) and X.shape[1] == len(t):
        X = X.T

    if X.shape[0] != len(t):
        raise RuntimeError(
            f"Truth state shape mismatch. Expected first dimension {len(t)}, got {X.shape}."
        )

    return X


def _state_labels(n):
    if n == 14:
        return [
            "cn_0", "cn_1", "cn_2", "cn_surf",
            "cp_0", "cp_1", "cp_2", "cp_surf",
            "ce_0", "ce_1", "ce_2", "ce_3", "ce_4", "ce_5",
        ]
    if n == 7:
        return [
            "cn_bulk", "cn_surf",
            "cp_bulk", "cp_surf",
            "ce_left", "ce_mid", "ce_right",
        ]
    return [f"x{i}" for i in range(n)]


def _surface_index_pair(n_states):
    """
    Match the convention already used in Cell 19A:
      full_14: xn surface index = 3, xp surface index = 7

    If reduced_7 is used later, use the usual reduced surface indices.
    """
    if n_states == 14:
        return 3, 7
    if n_states == 7:
        return 1, 3
    raise RuntimeError(f"Unsupported state dimension for xp/xn extraction: {n_states}")


def _plot_state_family(t, X, labels, title, ylabel):
    plt.figure(figsize=(12, 5))
    for j, lab in enumerate(labels):
        plt.plot(t, X[:, j], linewidth=1.8, label=lab)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_normalized_state_family(t, X, labels, title):
    plt.figure(figsize=(12, 5))
    for j, lab in enumerate(labels):
        plt.plot(t, _normalize_for_plot(X[:, j]), linewidth=1.8, label=lab)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized change from initial value")
    plt.title(title)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()


def _plot_range_bars(range_df, column, title, xlabel):
    df = range_df.copy().sort_values(column, ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["state"], df[column])
    plt.grid(True, axis="x", alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel("State")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _eval_surface_grid(fn, xn_grid, xp_grid):
    Z = np.empty((len(xp_grid), len(xn_grid)), dtype=np.float64)
    for i, xp in enumerate(xp_grid):
        for j, xn in enumerate(xn_grid):
            Z[i, j] = float(fn(float(xn), float(xp)))
    return Z


def _plot_surface_contour(xn_grid, xp_grid, Z, title, zlabel):
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(xn_grid, xp_grid, Z, levels=30)
    plt.colorbar(cs, label=zlabel)
    plt.xlabel(r"$x_n$ / negative-electrode stoichiometry")
    plt.ylabel(r"$x_p$ / positive-electrode stoichiometry")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _plot_surface_3d(xn_grid, xp_grid, Z, title, zlabel):
    XN, XP = np.meshgrid(xn_grid, xp_grid)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XN, XP, Z, linewidth=0, antialiased=True, alpha=0.95)
    fig.colorbar(surf, shrink=0.65, aspect=12, label=zlabel)
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$x_p$")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 19B.2 — Get learned proxy states and true states on same ID grid
# ---------------------------------------------------------
prep_19b, t_19b, u_19b = _get_prep_from_surface_result(surface_result_nl)

X_learned_19b = np.asarray(surface_stage_nl["xhat"], dtype=np.float64)
if X_learned_19b.shape[0] != len(t_19b) and X_learned_19b.shape[1] == len(t_19b):
    X_learned_19b = X_learned_19b.T

X_truth_19b = _simulate_truth_states_on_id_grid(t_19b, u_19b)

n_learned = X_learned_19b.shape[1]
n_truth = X_truth_19b.shape[1]

labels_learned = _state_labels(n_learned)
labels_truth = _state_labels(n_truth)

xn_idx_learned, xp_idx_learned = _surface_index_pair(n_learned)
xn_idx_truth, xp_idx_truth = _surface_index_pair(n_truth)

xn_learned_19b = X_learned_19b[:, xn_idx_learned] / float(cfg_nl_surface.csn_max)
xp_learned_19b = X_learned_19b[:, xp_idx_learned] / float(cfg_nl_surface.csp_max)

xn_truth_19b = X_truth_19b[:, xn_idx_truth] / float(cfg_nl_surface.csn_max)
xp_truth_19b = X_truth_19b[:, xp_idx_truth] / float(cfg_nl_surface.csp_max)

print("=" * 72)
print("CELL 19B — SUPERVISOR STATE-RANGE / NONLINEARITY DIAGNOSTIC")
print("=" * 72)
print("t range [s]:", float(t_19b[0]), "to", float(t_19b[-1]))
print("samples:", len(t_19b))
print("learned state shape:", X_learned_19b.shape)
print("truth state shape:", X_truth_19b.shape)
print("xn learned range:", _safe_min(xn_learned_19b), "to", _safe_max(xn_learned_19b), "span:", _safe_span(xn_learned_19b))
print("xp learned range:", _safe_min(xp_learned_19b), "to", _safe_max(xp_learned_19b), "span:", _safe_span(xp_learned_19b))
print("xn truth range  :", _safe_min(xn_truth_19b), "to", _safe_max(xn_truth_19b), "span:", _safe_span(xn_truth_19b))
print("xp truth range  :", _safe_min(xp_truth_19b), "to", _safe_max(xp_truth_19b), "span:", _safe_span(xp_truth_19b))
print("=" * 72)


# ---------------------------------------------------------
# 19B.3 — State range tables
# ---------------------------------------------------------
truth_rows = []
for j, lab in enumerate(labels_truth):
    truth_rows.append({
        "source": "truth",
        "state": lab,
        "min": _safe_min(X_truth_19b[:, j]),
        "max": _safe_max(X_truth_19b[:, j]),
        "span": _safe_span(X_truth_19b[:, j]),
        "initial": float(X_truth_19b[0, j]),
        "final": float(X_truth_19b[-1, j]),
        "final_minus_initial": float(X_truth_19b[-1, j] - X_truth_19b[0, j]),
    })

learned_rows = []
for j, lab in enumerate(labels_learned):
    learned_rows.append({
        "source": "learned_proxy",
        "state": lab,
        "min": _safe_min(X_learned_19b[:, j]),
        "max": _safe_max(X_learned_19b[:, j]),
        "span": _safe_span(X_learned_19b[:, j]),
        "initial": float(X_learned_19b[0, j]),
        "final": float(X_learned_19b[-1, j]),
        "final_minus_initial": float(X_learned_19b[-1, j] - X_learned_19b[0, j]),
    })

truth_state_range_df_19b = pd.DataFrame(truth_rows)
learned_state_range_df_19b = pd.DataFrame(learned_rows)

print("\nTruth-state ranges:")
display(truth_state_range_df_19b)

print("\nLearned proxy-state ranges:")
display(learned_state_range_df_19b)

stoich_range_df_19b = pd.DataFrame([
    {
        "source": "truth",
        "xn_min": _safe_min(xn_truth_19b),
        "xn_max": _safe_max(xn_truth_19b),
        "xn_span": _safe_span(xn_truth_19b),
        "xp_min": _safe_min(xp_truth_19b),
        "xp_max": _safe_max(xp_truth_19b),
        "xp_span": _safe_span(xp_truth_19b),
    },
    {
        "source": "learned_proxy",
        "xn_min": _safe_min(xn_learned_19b),
        "xn_max": _safe_max(xn_learned_19b),
        "xn_span": _safe_span(xn_learned_19b),
        "xp_min": _safe_min(xp_learned_19b),
        "xp_max": _safe_max(xp_learned_19b),
        "xp_span": _safe_span(xp_learned_19b),
    },
])

print("\nStoichiometry ranges used by nonlinear surface:")
display(stoich_range_df_19b)


# ---------------------------------------------------------
# 19B.4 — Plot all true and learned proxy states
# ---------------------------------------------------------
_plot_state_family(
    t_19b,
    X_truth_19b,
    labels_truth,
    title="Truth state trajectories on frozen nonlinear-ID input",
    ylabel="Truth state value",
)

_plot_normalized_state_family(
    t_19b,
    X_truth_19b,
    labels_truth,
    title="Truth state trajectories — normalized change from initial value",
)

_plot_state_family(
    t_19b,
    X_learned_19b,
    labels_learned,
    title="Learned proxy state trajectories from final nonlinear stage",
    ylabel="Learned proxy state value",
)

_plot_normalized_state_family(
    t_19b,
    X_learned_19b,
    labels_learned,
    title="Learned proxy state trajectories — normalized change from initial value",
)

_plot_range_bars(
    truth_state_range_df_19b,
    column="span",
    title="Truth state spans on visited trajectory",
    xlabel="State span: max - min",
)

_plot_range_bars(
    learned_state_range_df_19b,
    column="span",
    title="Learned proxy state spans on fitted trajectory",
    xlabel="State span: max - min",
)


# ---------------------------------------------------------
# 19B.5 — Plot xp/xn trajectories and their visited box
# ---------------------------------------------------------
plt.figure(figsize=(11, 5))
plt.plot(t_19b, xn_truth_19b, linewidth=2, label="Truth $x_n$")
plt.plot(t_19b, xp_truth_19b, linewidth=2, label="Truth $x_p$")
plt.plot(t_19b, xn_learned_19b, "--", linewidth=2, label="Learned proxy $x_n$")
plt.plot(t_19b, xp_learned_19b, "--", linewidth=2, label="Learned proxy $x_p$")
plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Stoichiometry")
plt.title("Visited stoichiometry trajectories: truth vs learned proxy")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 6))
plt.plot(xn_truth_19b, xp_truth_19b, linewidth=2, label="Truth path")
plt.plot(xn_learned_19b, xp_learned_19b, "--", linewidth=2, label="Learned proxy path")
plt.scatter([xn_truth_19b[0]], [xp_truth_19b[0]], s=80, marker="o", label="Truth start")
plt.scatter([xn_truth_19b[-1]], [xp_truth_19b[-1]], s=80, marker="x", label="Truth end")
plt.grid(True, alpha=0.3)
plt.xlabel(r"$x_n$")
plt.ylabel(r"$x_p$")
plt.title("Visited path in nonlinear surface coordinates")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 19B.5b — Supervisor plot: xn-xp plane comparison
# ---------------------------------------------------------
# Purpose:
#   Show the visited nonlinear-state plane directly:
#       x-axis: negative-electrode surface stoichiometry x_n
#       y-axis: positive-electrode surface stoichiometry x_p
#
#   This answers:
#       - Do truth and learned trajectories occupy the same xn/xp region?
#       - Is the learned model using a different internal state path?
#       - Is the nonlinear fit only matching voltage along a narrow path?
#
# Placement:
#   Put this immediately AFTER 19B.5 and BEFORE 19B.6.
#   It uses variables already created earlier in CELL 19B:
#       t_19b, xn_truth_19b, xp_truth_19b,
#       xn_learned_19b, xp_learned_19b
# ---------------------------------------------------------

required_19b5b = [
    "t_19b",
    "xn_truth_19b",
    "xp_truth_19b",
    "xn_learned_19b",
    "xp_learned_19b",
]

missing_19b5b = [name for name in required_19b5b if name not in globals()]
if missing_19b5b:
    raise RuntimeError(
        "CELL 19B.5b is missing required variables. "
        "Run CELL 19B up to section 19B.5 first. Missing: "
        + ", ".join(missing_19b5b)
    )

# Convert everything safely to 1D float arrays
t_plot_19b5b = np.asarray(t_19b, dtype=np.float64).reshape(-1)

xn_truth_plot_19b5b = np.asarray(xn_truth_19b, dtype=np.float64).reshape(-1)
xp_truth_plot_19b5b = np.asarray(xp_truth_19b, dtype=np.float64).reshape(-1)

xn_learned_plot_19b5b = np.asarray(xn_learned_19b, dtype=np.float64).reshape(-1)
xp_learned_plot_19b5b = np.asarray(xp_learned_19b, dtype=np.float64).reshape(-1)

# Make sure all arrays have the same usable length
n_19b5b = min(
    len(t_plot_19b5b),
    len(xn_truth_plot_19b5b),
    len(xp_truth_plot_19b5b),
    len(xn_learned_plot_19b5b),
    len(xp_learned_plot_19b5b),
)

t_plot_19b5b = t_plot_19b5b[:n_19b5b]

xn_truth_plot_19b5b = xn_truth_plot_19b5b[:n_19b5b]
xp_truth_plot_19b5b = xp_truth_plot_19b5b[:n_19b5b]

xn_learned_plot_19b5b = xn_learned_plot_19b5b[:n_19b5b]
xp_learned_plot_19b5b = xp_learned_plot_19b5b[:n_19b5b]

# Remove any bad samples to prevent matplotlib or metric crashes
finite_mask_19b5b = (
    np.isfinite(t_plot_19b5b)
    & np.isfinite(xn_truth_plot_19b5b)
    & np.isfinite(xp_truth_plot_19b5b)
    & np.isfinite(xn_learned_plot_19b5b)
    & np.isfinite(xp_learned_plot_19b5b)
)

t_plot_19b5b = t_plot_19b5b[finite_mask_19b5b]

xn_truth_plot_19b5b = xn_truth_plot_19b5b[finite_mask_19b5b]
xp_truth_plot_19b5b = xp_truth_plot_19b5b[finite_mask_19b5b]

xn_learned_plot_19b5b = xn_learned_plot_19b5b[finite_mask_19b5b]
xp_learned_plot_19b5b = xp_learned_plot_19b5b[finite_mask_19b5b]

if len(t_plot_19b5b) < 2:
    raise RuntimeError(
        "CELL 19B.5b has fewer than two valid finite samples after masking. "
        "Check xn/xp extraction earlier in CELL 19B."
    )

# Shared axis limits so truth and learned are visually comparable
xn_all_19b5b = np.concatenate([xn_truth_plot_19b5b, xn_learned_plot_19b5b])
xp_all_19b5b = np.concatenate([xp_truth_plot_19b5b, xp_learned_plot_19b5b])

xn_span_19b5b = float(np.max(xn_all_19b5b) - np.min(xn_all_19b5b))
xp_span_19b5b = float(np.max(xp_all_19b5b) - np.min(xp_all_19b5b))

xn_pad_19b5b = max(0.05 * xn_span_19b5b, 1e-4)
xp_pad_19b5b = max(0.05 * xp_span_19b5b, 1e-4)

xn_xlim_19b5b = (
    float(np.min(xn_all_19b5b) - xn_pad_19b5b),
    float(np.max(xn_all_19b5b) + xn_pad_19b5b),
)

xp_ylim_19b5b = (
    float(np.min(xp_all_19b5b) - xp_pad_19b5b),
    float(np.max(xp_all_19b5b) + xp_pad_19b5b),
)

# ---------------------------------------------------------
# Plot 1: Truth and learned paths overlaid in the same xn-xp plane
# ---------------------------------------------------------
plt.figure(figsize=(7.5, 6.5))

plt.plot(
    xn_truth_plot_19b5b,
    xp_truth_plot_19b5b,
    linewidth=2.5,
    label="Truth path",
)

plt.plot(
    xn_learned_plot_19b5b,
    xp_learned_plot_19b5b,
    "--",
    linewidth=2.5,
    label="Learned proxy path",
)

# Start and end markers
plt.scatter(
    [xn_truth_plot_19b5b[0]],
    [xp_truth_plot_19b5b[0]],
    s=90,
    marker="o",
    label="Truth start",
)

plt.scatter(
    [xn_truth_plot_19b5b[-1]],
    [xp_truth_plot_19b5b[-1]],
    s=90,
    marker="x",
    label="Truth end",
)

plt.scatter(
    [xn_learned_plot_19b5b[0]],
    [xp_learned_plot_19b5b[0]],
    s=90,
    marker="s",
    label="Learned start",
)

plt.scatter(
    [xn_learned_plot_19b5b[-1]],
    [xp_learned_plot_19b5b[-1]],
    s=90,
    marker="D",
    label="Learned end",
)

plt.xlim(xn_xlim_19b5b)
plt.ylim(xp_ylim_19b5b)
plt.grid(True, alpha=0.3)
plt.xlabel(r"Negative-electrode state, $x_n$")
plt.ylabel(r"Positive-electrode state, $x_p$")
plt.title(r"Visited nonlinear-state plane: $x_n$ vs $x_p$")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot 2: Same comparison, but colored by time
# This shows whether both paths evolve in the same direction.
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

sc0 = axs[0].scatter(
    xn_truth_plot_19b5b,
    xp_truth_plot_19b5b,
    c=t_plot_19b5b,
    s=35,
)

axs[0].plot(
    xn_truth_plot_19b5b,
    xp_truth_plot_19b5b,
    linewidth=1.5,
    alpha=0.7,
)

axs[0].scatter(
    [xn_truth_plot_19b5b[0]],
    [xp_truth_plot_19b5b[0]],
    s=100,
    marker="o",
    label="Start",
)

axs[0].scatter(
    [xn_truth_plot_19b5b[-1]],
    [xp_truth_plot_19b5b[-1]],
    s=100,
    marker="x",
    label="End",
)

axs[0].set_title("Truth path")
axs[0].set_xlabel(r"$x_n$")
axs[0].set_ylabel(r"$x_p$")
axs[0].grid(True, alpha=0.3)
axs[0].legend(loc="best")

sc1 = axs[1].scatter(
    xn_learned_plot_19b5b,
    xp_learned_plot_19b5b,
    c=t_plot_19b5b,
    s=35,
)

axs[1].plot(
    xn_learned_plot_19b5b,
    xp_learned_plot_19b5b,
    linewidth=1.5,
    alpha=0.7,
)

axs[1].scatter(
    [xn_learned_plot_19b5b[0]],
    [xp_learned_plot_19b5b[0]],
    s=100,
    marker="o",
    label="Start",
)

axs[1].scatter(
    [xn_learned_plot_19b5b[-1]],
    [xp_learned_plot_19b5b[-1]],
    s=100,
    marker="x",
    label="End",
)

axs[1].set_title("Learned proxy path")
axs[1].set_xlabel(r"$x_n$")
axs[1].grid(True, alpha=0.3)
axs[1].legend(loc="best")

for ax in axs:
    ax.set_xlim(xn_xlim_19b5b)
    ax.set_ylim(xp_ylim_19b5b)

cbar = fig.colorbar(sc1, ax=axs.ravel().tolist(), shrink=0.95)
cbar.set_label("Time [s]")

fig.suptitle(r"Truth vs learned trajectory in the $x_n$-$x_p$ plane")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Plot 3: Pointwise mismatch between truth and learned path
# ---------------------------------------------------------
dxn_19b5b = xn_learned_plot_19b5b - xn_truth_plot_19b5b
dxp_19b5b = xp_learned_plot_19b5b - xp_truth_plot_19b5b
path_error_19b5b = np.sqrt(dxn_19b5b**2 + dxp_19b5b**2)

plt.figure(figsize=(11, 4.5))
plt.plot(t_plot_19b5b, dxn_19b5b, linewidth=2, label=r"$x_{n,\ learned} - x_{n,\ truth}$")
plt.plot(t_plot_19b5b, dxp_19b5b, linewidth=2, label=r"$x_{p,\ learned} - x_{p,\ truth}$")
plt.plot(t_plot_19b5b, path_error_19b5b, "--", linewidth=2, label="Euclidean path mismatch")
plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("State mismatch")
plt.title(r"Pointwise mismatch between truth and learned $x_n$/$x_p$ paths")
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Print numerical summary for thesis interpretation
# ---------------------------------------------------------
summary_19b5b = pd.DataFrame(
    {
        "quantity": [
            "truth_xn_min",
            "truth_xn_max",
            "truth_xn_span",
            "truth_xp_min",
            "truth_xp_max",
            "truth_xp_span",
            "learned_xn_min",
            "learned_xn_max",
            "learned_xn_span",
            "learned_xp_min",
            "learned_xp_max",
            "learned_xp_span",
            "mean_abs_xn_mismatch",
            "mean_abs_xp_mismatch",
            "mean_path_mismatch",
            "max_path_mismatch",
        ],
        "value": [
            float(np.min(xn_truth_plot_19b5b)),
            float(np.max(xn_truth_plot_19b5b)),
            float(np.max(xn_truth_plot_19b5b) - np.min(xn_truth_plot_19b5b)),
            float(np.min(xp_truth_plot_19b5b)),
            float(np.max(xp_truth_plot_19b5b)),
            float(np.max(xp_truth_plot_19b5b) - np.min(xp_truth_plot_19b5b)),
            float(np.min(xn_learned_plot_19b5b)),
            float(np.max(xn_learned_plot_19b5b)),
            float(np.max(xn_learned_plot_19b5b) - np.min(xn_learned_plot_19b5b)),
            float(np.min(xp_learned_plot_19b5b)),
            float(np.max(xp_learned_plot_19b5b)),
            float(np.max(xp_learned_plot_19b5b) - np.min(xp_learned_plot_19b5b)),
            float(np.mean(np.abs(dxn_19b5b))),
            float(np.mean(np.abs(dxp_19b5b))),
            float(np.mean(path_error_19b5b)),
            float(np.max(path_error_19b5b)),
        ],
    }
)

print("\nCELL 19B.5b — xn/xp trajectory-plane summary")
display(summary_19b5b)

# ---------------------------------------------------------
# 19B.6 — TRUE nonlinearity only on the actually visited xp/xn range
# ---------------------------------------------------------
guard_19b = float(globals().get("NONLINEAR_SURFACE_GUARD", 1e-4))
grid_n_19b = int(globals().get("NONLINEAR_SURFACE_GRID_N", 60))

# Main supervisor version:
# Use the union of truth and learned visited ranges.
# This shows the TRUE surface only where the experiment/proxy actually goes.
xn_min_visit = max(
    guard_19b,
    min(_safe_min(xn_truth_19b), _safe_min(xn_learned_19b)),
)
xn_max_visit = min(
    1.0 - guard_19b,
    max(_safe_max(xn_truth_19b), _safe_max(xn_learned_19b)),
)
xp_min_visit = max(
    guard_19b,
    min(_safe_min(xp_truth_19b), _safe_min(xp_learned_19b)),
)
xp_max_visit = min(
    1.0 - guard_19b,
    max(_safe_max(xp_truth_19b), _safe_max(xp_learned_19b)),
)

# If the visited range is extremely narrow, add a tiny visual padding.
# This avoids a misleading single-line surface plot.
xn_pad = max(0.05 * (xn_max_visit - xn_min_visit), 5e-4)
xp_pad = max(0.05 * (xp_max_visit - xp_min_visit), 5e-4)

xn_min_plot = max(guard_19b, xn_min_visit - xn_pad)
xn_max_plot = min(1.0 - guard_19b, xn_max_visit + xn_pad)
xp_min_plot = max(guard_19b, xp_min_visit - xp_pad)
xp_max_plot = min(1.0 - guard_19b, xp_max_visit + xp_pad)

xn_grid_visit = np.linspace(xn_min_plot, xn_max_plot, grid_n_19b)
xp_grid_visit = np.linspace(xp_min_plot, xp_max_plot, grid_n_19b)

Z_truth_visit_19b = _eval_surface_grid(
    truth_surface_fn_nl,
    xn_grid_visit,
    xp_grid_visit,
)

Z_learned_visit_19b = _eval_surface_grid(
    learned_surface_fn_nl,
    xn_grid_visit,
    xp_grid_visit,
)

Z_error_visit_19b = Z_learned_visit_19b - Z_truth_visit_19b

visit_surface_metrics_19b = {
    "truth_surface_min_on_visited_box": float(np.nanmin(Z_truth_visit_19b)),
    "truth_surface_max_on_visited_box": float(np.nanmax(Z_truth_visit_19b)),
    "truth_surface_span_on_visited_box": float(np.nanmax(Z_truth_visit_19b) - np.nanmin(Z_truth_visit_19b)),
    "learned_surface_min_on_visited_box": float(np.nanmin(Z_learned_visit_19b)),
    "learned_surface_max_on_visited_box": float(np.nanmax(Z_learned_visit_19b)),
    "learned_surface_span_on_visited_box": float(np.nanmax(Z_learned_visit_19b) - np.nanmin(Z_learned_visit_19b)),
    "surface_error_rmse_on_visited_box": float(np.sqrt(np.nanmean(Z_error_visit_19b ** 2))),
    "surface_error_mae_on_visited_box": float(np.nanmean(np.abs(Z_error_visit_19b))),
    "surface_error_max_abs_on_visited_box": float(np.nanmax(np.abs(Z_error_visit_19b))),
    "xn_plot_min": float(xn_min_plot),
    "xn_plot_max": float(xn_max_plot),
    "xp_plot_min": float(xp_min_plot),
    "xp_plot_max": float(xp_max_plot),
}

print("\nVisited-box surface metrics:")
for k, v in visit_surface_metrics_19b.items():
    print(f"  {k}: {v}")

_plot_surface_contour(
    xn_grid_visit,
    xp_grid_visit,
    Z_truth_visit_19b,
    title="TRUE nonlinearity only on visited $x_n$/$x_p$ range",
    zlabel="Truth nonlinear voltage map",
)

_plot_surface_3d(
    xn_grid_visit,
    xp_grid_visit,
    Z_truth_visit_19b,
    title="TRUE nonlinearity only on visited $x_n$/$x_p$ range",
    zlabel="Truth nonlinear voltage map",
)

_plot_surface_contour(
    xn_grid_visit,
    xp_grid_visit,
    Z_learned_visit_19b,
    title="LEARNED nonlinearity on same visited $x_n$/$x_p$ range",
    zlabel="Learned nonlinear voltage map",
)

_plot_surface_contour(
    xn_grid_visit,
    xp_grid_visit,
    Z_error_visit_19b,
    title="Surface error on visited range: learned - truth",
    zlabel="Surface error",
)


# ---------------------------------------------------------
# 19B.7 — Overlay visited path on true nonlinearity contour
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
cs = plt.contourf(xn_grid_visit, xp_grid_visit, Z_truth_visit_19b, levels=30)
plt.colorbar(cs, label="Truth nonlinear voltage map")

plt.plot(
    xn_truth_19b,
    xp_truth_19b,
    linewidth=2.5,
    label="Truth visited path",
)
plt.plot(
    xn_learned_19b,
    xp_learned_19b,
    "--",
    linewidth=2.5,
    label="Learned proxy visited path",
)

plt.scatter([xn_truth_19b[0]], [xp_truth_19b[0]], s=80, marker="o", label="Truth start")
plt.scatter([xn_truth_19b[-1]], [xp_truth_19b[-1]], s=80, marker="x", label="Truth end")

plt.xlabel(r"$x_n$")
plt.ylabel(r"$x_p$")
plt.title("TRUE nonlinearity with actual visited trajectory overlaid")
plt.grid(True, alpha=0.25)
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 19B.8 — Evaluate truth and learned surfaces along the actual paths
# ---------------------------------------------------------
z_truth_along_truth_path_19b = np.array([
    truth_surface_fn_nl(xn, xp)
    for xn, xp in zip(xn_truth_19b, xp_truth_19b)
], dtype=np.float64)

z_learned_along_learned_path_19b = np.array([
    learned_surface_fn_nl(xn, xp)
    for xn, xp in zip(xn_learned_19b, xp_learned_19b)
], dtype=np.float64)

z_truth_along_learned_path_19b = np.array([
    truth_surface_fn_nl(xn, xp)
    for xn, xp in zip(xn_learned_19b, xp_learned_19b)
], dtype=np.float64)

z_learned_minus_truth_along_learned_path_19b = (
    z_learned_along_learned_path_19b - z_truth_along_learned_path_19b
)

plt.figure(figsize=(11, 5))
plt.plot(
    t_19b,
    z_truth_along_truth_path_19b - z_truth_along_truth_path_19b[0],
    linewidth=2,
    label="Truth surface along truth path, centered",
)
plt.plot(
    t_19b,
    z_truth_along_learned_path_19b - z_truth_along_learned_path_19b[0],
    linewidth=2,
    label="Truth surface along learned path, centered",
)
plt.plot(
    t_19b,
    z_learned_along_learned_path_19b - z_learned_along_learned_path_19b[0],
    "--",
    linewidth=2,
    label="Learned surface along learned path, centered",
)
plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Centered nonlinear contribution")
plt.title("Nonlinearity along the actual visited state paths")
plt.legend(loc="best")
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 4))
plt.plot(
    t_19b,
    z_learned_minus_truth_along_learned_path_19b,
    linewidth=2,
)
plt.grid(True, alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Learned - truth")
plt.title("Surface mismatch along learned proxy path")
plt.tight_layout()
plt.show()

# %% =====================================================
# CELL 19B.8b — xn-xp trajectory colored by nonlinear voltage map
# =====================================================
# Purpose:
#   Plot the truth and learned proxy trajectories in the x_n-x_p plane,
#   but color the path by the nonlinear voltage-map value instead of time.
#
#   Left panel:
#       Truth trajectory colored by truth nonlinear voltage map:
#           z = f_truth(x_n_truth, x_p_truth)
#
#   Right panel:
#       Learned proxy trajectory colored by learned nonlinear voltage map:
#           z = f_learned(x_n_learned, x_p_learned)
#
# Why this is better than coloring by time:
#   The question is not mainly "when did the model pass here?"
#   The question is "what nonlinear voltage value is the model using
#   at this visited location in the x_n-x_p plane?"
#
# Placement:
#   Put this immediately AFTER CELL 19B.8.
#   It uses z-values already computed in CELL 19B.8.
# =====================================================

from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

required_19b8b = [
    "xn_truth_19b",
    "xp_truth_19b",
    "xn_learned_19b",
    "xp_learned_19b",
    "z_truth_along_truth_path_19b",
    "z_learned_along_learned_path_19b",
]

missing_19b8b = [name for name in required_19b8b if name not in globals()]
if missing_19b8b:
    raise RuntimeError(
        "CELL 19B.8b is missing required variables. "
        "Run CELL 19B.8 first. Missing: "
        + ", ".join(missing_19b8b)
    )


def _as_1d_float_19b8b(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _finite_path_arrays_19b8b(xn, xp, z):
    xn = _as_1d_float_19b8b(xn)
    xp = _as_1d_float_19b8b(xp)
    z = _as_1d_float_19b8b(z)

    n = min(len(xn), len(xp), len(z))
    xn = xn[:n]
    xp = xp[:n]
    z = z[:n]

    mask = np.isfinite(xn) & np.isfinite(xp) & np.isfinite(z)
    return xn[mask], xp[mask], z[mask]


def _colored_path_19b8b(
    ax,
    xn,
    xp,
    z,
    norm,
    cmap,
    label,
    line_width=3.0,
    start_end=True,
):
    """
    Draw a trajectory as colored line segments.
    This avoids plotting hundreds of individual scatter points,
    so the path does not look clustered or messy.
    """
    xn, xp, z = _finite_path_arrays_19b8b(xn, xp, z)

    if len(xn) < 2:
        raise ValueError(f"{label}: need at least two finite points to draw a path.")

    points = np.column_stack([xn, xp]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Segment color uses the average nonlinear-map value of the two segment endpoints.
    z_seg = 0.5 * (z[:-1] + z[1:])

    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=line_width,
        capstyle="round",
        joinstyle="round",
    )
    lc.set_array(z_seg)
    ax.add_collection(lc)

    if start_end:
        ax.scatter(
            xn[0],
            xp[0],
            s=95,
            marker="o",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
            label="Start",
        )
        ax.scatter(
            xn[-1],
            xp[-1],
            s=110,
            marker="X",
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
            label="End",
        )

    ax.plot([], [], linewidth=line_width, label=label)

    return lc


# ---------------------------------------------------------
# Choose whether to color by absolute or centered nonlinear map value.
# Centered is usually better for thesis comparison because it shows
# nonlinear variation along the path without being dominated by offsets.
# ---------------------------------------------------------
COLOR_NONLINEAR_MAP_MODE_19B8B = "centered"   # "centered" or "absolute"

z_truth_color_19b8b = _as_1d_float_19b8b(z_truth_along_truth_path_19b)
z_learned_color_19b8b = _as_1d_float_19b8b(z_learned_along_learned_path_19b)

if COLOR_NONLINEAR_MAP_MODE_19B8B == "centered":
    z_truth_color_19b8b = z_truth_color_19b8b - z_truth_color_19b8b[0]
    z_learned_color_19b8b = z_learned_color_19b8b - z_learned_color_19b8b[0]
    color_label_19b8b = "Centered nonlinear voltage-map value [V]"
elif COLOR_NONLINEAR_MAP_MODE_19B8B == "absolute":
    color_label_19b8b = "Nonlinear voltage-map value [V]"
else:
    raise ValueError(
        "Unsupported COLOR_NONLINEAR_MAP_MODE_19B8B. "
        "Use 'centered' or 'absolute'."
    )


# ---------------------------------------------------------
# Use one shared color scale so the two panels are visually comparable.
# ---------------------------------------------------------
z_all_19b8b = np.concatenate([
    z_truth_color_19b8b[np.isfinite(z_truth_color_19b8b)],
    z_learned_color_19b8b[np.isfinite(z_learned_color_19b8b)],
])

if len(z_all_19b8b) == 0:
    raise RuntimeError("No finite nonlinear-map values available for CELL 19B.8b.")

zmin_19b8b = float(np.min(z_all_19b8b))
zmax_19b8b = float(np.max(z_all_19b8b))

if abs(zmax_19b8b - zmin_19b8b) < 1e-14:
    zpad_19b8b = 1e-6
else:
    zpad_19b8b = 0.03 * (zmax_19b8b - zmin_19b8b)

norm_19b8b = Normalize(
    vmin=zmin_19b8b - zpad_19b8b,
    vmax=zmax_19b8b + zpad_19b8b,
)

cmap_19b8b = "viridis"


# ---------------------------------------------------------
# Axis limits: common x/y limits for fair visual comparison.
# Add padding so the path does not touch the plot boundary.
# ---------------------------------------------------------
xn_all_19b8b = np.concatenate([
    _as_1d_float_19b8b(xn_truth_19b),
    _as_1d_float_19b8b(xn_learned_19b),
])
xp_all_19b8b = np.concatenate([
    _as_1d_float_19b8b(xp_truth_19b),
    _as_1d_float_19b8b(xp_learned_19b),
])

xn_all_19b8b = xn_all_19b8b[np.isfinite(xn_all_19b8b)]
xp_all_19b8b = xp_all_19b8b[np.isfinite(xp_all_19b8b)]

xpad_19b8b = 0.06 * max(float(np.max(xn_all_19b8b) - np.min(xn_all_19b8b)), 1e-6)
ypad_19b8b = 0.06 * max(float(np.max(xp_all_19b8b) - np.min(xp_all_19b8b)), 1e-6)

xlim_19b8b = (
    float(np.min(xn_all_19b8b) - xpad_19b8b),
    float(np.max(xn_all_19b8b) + xpad_19b8b),
)
ylim_19b8b = (
    float(np.min(xp_all_19b8b) - ypad_19b8b),
    float(np.max(xp_all_19b8b) + ypad_19b8b),
)


# ---------------------------------------------------------
# Clean figure layout:
#   - two main panels
#   - one horizontal colorbar under each panel
# This avoids the ugly colorbar squeezed into the right side.
# ---------------------------------------------------------
fig = plt.figure(figsize=(14, 6.8))
gs = gridspec.GridSpec(
    nrows=2,
    ncols=2,
    height_ratios=[20, 1.1],
    hspace=0.30,
    wspace=0.16,
)

ax_truth = fig.add_subplot(gs[0, 0])
ax_learned = fig.add_subplot(gs[0, 1], sharex=ax_truth, sharey=ax_truth)

cax_truth = fig.add_subplot(gs[1, 0])
cax_learned = fig.add_subplot(gs[1, 1])


lc_truth = _colored_path_19b8b(
    ax=ax_truth,
    xn=xn_truth_19b,
    xp=xp_truth_19b,
    z=z_truth_color_19b8b,
    norm=norm_19b8b,
    cmap=cmap_19b8b,
    label="Truth path",
    line_width=3.2,
)

lc_learned = _colored_path_19b8b(
    ax=ax_learned,
    xn=xn_learned_19b,
    xp=xp_learned_19b,
    z=z_learned_color_19b8b,
    norm=norm_19b8b,
    cmap=cmap_19b8b,
    label="Learned proxy path",
    line_width=3.2,
)


for ax in [ax_truth, ax_learned]:
    ax.set_xlim(xlim_19b8b)
    ax.set_ylim(ylim_19b8b)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel(r"$x_n$ / negative-electrode stoichiometry")
    ax.set_ylabel(r"$x_p$ / positive-electrode stoichiometry")
    ax.set_aspect("auto")
    ax.legend(loc="best", frameon=True)


ax_truth.set_title(
    r"Truth trajectory colored by $f_{\mathrm{truth}}(x_n,x_p)$"
)
ax_learned.set_title(
    r"Learned proxy trajectory colored by $f_{\mathrm{learned}}(x_n,x_p)$"
)


cb_truth = fig.colorbar(lc_truth, cax=cax_truth, orientation="horizontal")
cb_truth.set_label(color_label_19b8b)

cb_learned = fig.colorbar(lc_learned, cax=cax_learned, orientation="horizontal")
cb_learned.set_label(color_label_19b8b)


fig.suptitle(
    r"Visited $x_n$-$x_p$ trajectories colored by nonlinear voltage-map value",
    y=0.98,
)

plt.show()


# ---------------------------------------------------------
# Numerical summary for interpretation
# ---------------------------------------------------------
summary_19b8b = pd.DataFrame(
    {
        "quantity": [
            "truth_map_min",
            "truth_map_max",
            "truth_map_span",
            "learned_map_min",
            "learned_map_max",
            "learned_map_span",
            "shared_color_min",
            "shared_color_max",
        ],
        "value": [
            float(np.min(z_truth_color_19b8b)),
            float(np.max(z_truth_color_19b8b)),
            float(np.max(z_truth_color_19b8b) - np.min(z_truth_color_19b8b)),
            float(np.min(z_learned_color_19b8b)),
            float(np.max(z_learned_color_19b8b)),
            float(np.max(z_learned_color_19b8b) - np.min(z_learned_color_19b8b)),
            float(norm_19b8b.vmin),
            float(norm_19b8b.vmax),
        ]
    }
)

print("=" * 72)
print("CELL 19B.8b — NONLINEAR MAP COLOR SUMMARY")
print("=" * 72)
print("Color mode:", COLOR_NONLINEAR_MAP_MODE_19B8B)
display(summary_19b8b)


# %% =====================================================
# CELL 19B.8c — nonlinear voltage-map value vs cumulative path distance
# =====================================================
# Purpose:
#   Convert the 2D visited path in the (x_n, x_p) plane into a 1D
#   cumulative path coordinate s, then plot the nonlinear voltage-map
#   value along that path.
#
#   This answers:
#   "Along the actually visited state trajectory, does the learned
#    nonlinear map reproduce the truth nonlinear map?"
#
# Placement:
#   Put this immediately AFTER CELL 19B.8b.
# =====================================================

required_19b8c = [
    "xn_truth_19b",
    "xp_truth_19b",
    "xn_learned_19b",
    "xp_learned_19b",
    "z_truth_along_truth_path_19b",
    "z_learned_along_learned_path_19b",
]

missing_19b8c = [name for name in required_19b8c if name not in globals()]
if missing_19b8c:
    raise RuntimeError(
        "CELL 19B.8c is missing required variables. "
        "Run CELL 19B.8 and CELL 19B.8b first. Missing: "
        + ", ".join(missing_19b8c)
    )


def _as_1d_float_19b8c(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _finite_triplet_19b8c(xn, xp, z):
    xn = _as_1d_float_19b8c(xn)
    xp = _as_1d_float_19b8c(xp)
    z = _as_1d_float_19b8c(z)

    n = min(len(xn), len(xp), len(z))
    xn = xn[:n]
    xp = xp[:n]
    z = z[:n]

    mask = np.isfinite(xn) & np.isfinite(xp) & np.isfinite(z)
    return xn[mask], xp[mask], z[mask]


def _cumulative_path_length_19b8c(xn, xp):
    """
    Compute cumulative arc length along the path in the (x_n, x_p) plane.
    s[0] = 0, and s[-1] is the total path length.
    """
    xn = _as_1d_float_19b8c(xn)
    xp = _as_1d_float_19b8c(xp)

    if len(xn) < 2:
        return np.zeros_like(xn, dtype=np.float64)

    dx = np.diff(xn)
    dy = np.diff(xp)
    ds = np.sqrt(dx**2 + dy**2)

    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s


def _safe_normalize_path_19b8c(s):
    s = _as_1d_float_19b8c(s)
    if len(s) == 0:
        return s.copy()
    s_end = float(s[-1])
    if s_end <= 1e-15:
        return np.zeros_like(s)
    return s / s_end


# ---------------------------------------------------------
# User controls
# ---------------------------------------------------------
COLOR_NONLINEAR_MAP_MODE_19B8C = "centered"   # "centered" or "absolute"
PATH_DISTANCE_MODE_19B8C = "absolute"         # "absolute" or "normalized"
PLOT_LAYOUT_19B8C = "overlay"                 # "overlay" or "stacked"

# ---------------------------------------------------------
# Gather and clean arrays
# ---------------------------------------------------------
xn_truth_c, xp_truth_c, z_truth_c = _finite_triplet_19b8c(
    xn_truth_19b,
    xp_truth_19b,
    z_truth_along_truth_path_19b,
)

xn_learned_c, xp_learned_c, z_learned_c = _finite_triplet_19b8c(
    xn_learned_19b,
    xp_learned_19b,
    z_learned_along_learned_path_19b,
)

if len(xn_truth_c) < 2 or len(xn_learned_c) < 2:
    raise RuntimeError(
        "Not enough points to build path-distance plots. "
        "Need at least 2 finite samples in each path."
    )

# ---------------------------------------------------------
# Match the same convention used in your previous plot
# ---------------------------------------------------------
if COLOR_NONLINEAR_MAP_MODE_19B8C == "centered":
    z_truth_plot = z_truth_c - z_truth_c[0]
    z_learned_plot = z_learned_c - z_learned_c[0]
    y_label_19b8c = "Centered nonlinear voltage-map value [V]"
elif COLOR_NONLINEAR_MAP_MODE_19B8C == "absolute":
    z_truth_plot = z_truth_c.copy()
    z_learned_plot = z_learned_c.copy()
    y_label_19b8c = "Nonlinear voltage-map value [V]"
else:
    raise ValueError(
        "Unsupported COLOR_NONLINEAR_MAP_MODE_19B8C. "
        "Use 'centered' or 'absolute'."
    )

# ---------------------------------------------------------
# Compute cumulative path distance
# ---------------------------------------------------------
s_truth = _cumulative_path_length_19b8c(xn_truth_c, xp_truth_c)
s_learned = _cumulative_path_length_19b8c(xn_learned_c, xp_learned_c)

if PATH_DISTANCE_MODE_19B8C == "normalized":
    s_truth_plot = _safe_normalize_path_19b8c(s_truth)
    s_learned_plot = _safe_normalize_path_19b8c(s_learned)
    x_label_19b8c = "Normalized cumulative path distance"
elif PATH_DISTANCE_MODE_19B8C == "absolute":
    s_truth_plot = s_truth.copy()
    s_learned_plot = s_learned.copy()
    x_label_19b8c = r"Cumulative path distance in the $(x_n,x_p)$ plane"
else:
    raise ValueError(
        "Unsupported PATH_DISTANCE_MODE_19B8C. "
        "Use 'absolute' or 'normalized'."
    )

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
if PLOT_LAYOUT_19B8C == "overlay":
    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    ax.plot(
        s_truth_plot,
        z_truth_plot,
        linewidth=3.0,
        label=r"Truth: $f_{\mathrm{truth}}(x_n(s),x_p(s))$",
    )
    ax.plot(
        s_learned_plot,
        z_learned_plot,
        "--",
        linewidth=3.0,
        label=r"Learned: $f_{\mathrm{learned}}(x_n(s),x_p(s))$",
    )

    ax.scatter(
        s_truth_plot[0],
        z_truth_plot[0],
        s=80,
        marker="o",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Truth start",
    )
    ax.scatter(
        s_truth_plot[-1],
        z_truth_plot[-1],
        s=95,
        marker="X",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Truth end",
    )

    ax.scatter(
        s_learned_plot[0],
        z_learned_plot[0],
        s=80,
        marker="s",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Learned start",
    )
    ax.scatter(
        s_learned_plot[-1],
        z_learned_plot[-1],
        s=90,
        marker="D",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Learned end",
    )

    ax.grid(True, alpha=0.3)
    ax.set_xlabel(x_label_19b8c)
    ax.set_ylabel(y_label_19b8c)
    ax.set_title("Nonlinear voltage-map value along cumulative path distance")
    ax.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.show()

elif PLOT_LAYOUT_19B8C == "stacked":
    fig, axs = plt.subplots(2, 1, figsize=(10.5, 7.2), sharex=True)

    axs[0].plot(
        s_truth_plot,
        z_truth_plot,
        linewidth=3.0,
        label=r"$f_{\mathrm{truth}}(x_n(s),x_p(s))$",
    )
    axs[0].scatter(
        s_truth_plot[0], z_truth_plot[0], s=80, marker="o",
        edgecolor="black", linewidth=0.8, zorder=4, label="Start"
    )
    axs[0].scatter(
        s_truth_plot[-1], z_truth_plot[-1], s=95, marker="X",
        edgecolor="black", linewidth=0.8, zorder=4, label="End"
    )
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylabel(y_label_19b8c)
    axs[0].set_title("Truth nonlinear map along truth path")
    axs[0].legend(loc="best", frameon=True)

    axs[1].plot(
        s_learned_plot,
        z_learned_plot,
        "--",
        linewidth=3.0,
        label=r"$f_{\mathrm{learned}}(x_n(s),x_p(s))$",
    )
    axs[1].scatter(
        s_learned_plot[0], z_learned_plot[0], s=80, marker="o",
        edgecolor="black", linewidth=0.8, zorder=4, label="Start"
    )
    axs[1].scatter(
        s_learned_plot[-1], z_learned_plot[-1], s=95, marker="X",
        edgecolor="black", linewidth=0.8, zorder=4, label="End"
    )
    axs[1].grid(True, alpha=0.3)
    axs[1].set_xlabel(x_label_19b8c)
    axs[1].set_ylabel(y_label_19b8c)
    axs[1].set_title("Learned nonlinear map along learned path")
    axs[1].legend(loc="best", frameon=True)

    plt.tight_layout()
    plt.show()

else:
    raise ValueError(
        "Unsupported PLOT_LAYOUT_19B8C. Use 'overlay' or 'stacked'."
    )

# ---------------------------------------------------------
# Helpful numerical summary for interpretation
# ---------------------------------------------------------
truth_total_path = float(s_truth[-1])
learned_total_path = float(s_learned[-1])

truth_straight = float(np.sqrt(
    (xn_truth_c[-1] - xn_truth_c[0])**2 +
    (xp_truth_c[-1] - xp_truth_c[0])**2
))
learned_straight = float(np.sqrt(
    (xn_learned_c[-1] - xn_learned_c[0])**2 +
    (xp_learned_c[-1] - xp_learned_c[0])**2
))

truth_tortuosity = truth_total_path / max(truth_straight, 1e-15)
learned_tortuosity = learned_total_path / max(learned_straight, 1e-15)

coverage_summary_19b8c = pd.DataFrame(
    {
        "quantity": [
            "truth_total_path_length",
            "learned_total_path_length",
            "truth_straight_line_distance",
            "learned_straight_line_distance",
            "truth_tortuosity",
            "learned_tortuosity",
            "truth_xn_span",
            "learned_xn_span",
            "truth_xp_span",
            "learned_xp_span",
            "truth_map_min",
            "truth_map_max",
            "truth_map_span",
            "learned_map_min",
            "learned_map_max",
            "learned_map_span",
        ],
        "value": [
            truth_total_path,
            learned_total_path,
            truth_straight,
            learned_straight,
            truth_tortuosity,
            learned_tortuosity,
            float(np.max(xn_truth_c) - np.min(xn_truth_c)),
            float(np.max(xn_learned_c) - np.min(xn_learned_c)),
            float(np.max(xp_truth_c) - np.min(xp_truth_c)),
            float(np.max(xp_learned_c) - np.min(xp_learned_c)),
            float(np.min(z_truth_plot)),
            float(np.max(z_truth_plot)),
            float(np.max(z_truth_plot) - np.min(z_truth_plot)),
            float(np.min(z_learned_plot)),
            float(np.max(z_learned_plot)),
            float(np.max(z_learned_plot) - np.min(z_learned_plot)),
        ],
    }
)

print("=" * 72)
print("CELL 19B.8c — PATH-DISTANCE NONLINEAR-MAP SUMMARY")
print("=" * 72)
print("Color mode      :", COLOR_NONLINEAR_MAP_MODE_19B8C)
print("Distance mode   :", PATH_DISTANCE_MODE_19B8C)
print("Plot layout     :", PLOT_LAYOUT_19B8C)
display(coverage_summary_19b8c)

# ---------------------------------------------------------
# 19B.9 — Final printed interpretation for thesis notes
# ---------------------------------------------------------
print("\n" + "=" * 72)
print("CELL 19B INTERPRETATION")
print("=" * 72)
print(
    "1) If xp/xn spans are very small, the experiment only excites a narrow slice "
    "of the nonlinear voltage surface."
)
print(
    "2) In that case, a model can fit voltage very well while learning a surface "
    "that differs away from the visited slice."
)
print(
    "3) The broad full-grid truth-vs-learned surface comparison is therefore a "
    "global extrapolation test, not only a trajectory-fit test."
)
print(
    "4) The visited-range truth surface and the path-overlay plot are the most "
    "important supervisor plots for explaining why the learned nonlinearity "
    "can look different from truth."
)
print("=" * 72)

# %% =====================================================
# CELL 20 — Surface plots for thesis/paper
# =====================================================
XN = surface_compare_nl.XN
XP = surface_compare_nl.XP
Z_truth = surface_compare_nl.Z_ref
Z_hat = surface_compare_nl.Z_hat
Z_res = surface_compare_nl.residual

plt.figure(figsize=(7, 5))
plt.contourf(XN, XP, Z_truth, levels=30)
plt.colorbar(label="Z_truth [V]")
plt.xlabel("x_n surface stoichiometry")
plt.ylabel("x_p surface stoichiometry")
plt.title("Truth equilibrium-voltage surface")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.contourf(XN, XP, Z_hat, levels=30)
plt.colorbar(label="Z_hat [V]")
plt.xlabel("x_n surface stoichiometry")
plt.ylabel("x_p surface stoichiometry")
plt.title(f"Learned equilibrium-voltage surface ({final_nl_stage_name})")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.contourf(XN, XP, Z_res, levels=30)
plt.colorbar(label="Z_hat - Z_truth [V]")
plt.xlabel("x_n surface stoichiometry")
plt.ylabel("x_p surface stoichiometry")
plt.title("Surface residual: learned minus truth")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(XN, XP, Z_truth, alpha=0.85)
ax.set_xlabel("x_n")
ax.set_ylabel("x_p")
ax.set_zlabel("Z_truth [V]")
ax.set_title("Truth equilibrium-voltage surface (3D)")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(XN, XP, Z_hat, alpha=0.85)
ax.set_xlabel("x_n")
ax.set_ylabel("x_p")
ax.set_zlabel("Z_hat [V]")
ax.set_title(f"Learned equilibrium-voltage surface (3D, {final_nl_stage_name})")
plt.tight_layout()
plt.show()


# %% =====================================================
# CELL 21 — Parameter summary for thesis
# =====================================================
print("Final nonlinear stage:", final_nl_stage_name)

if "R0_hat" in stage2_nl:
    print("\nStage 2:")
    print("  R0_hat =", stage2_nl["R0_hat"])
    print("  ||thetaZ||_2 =", np.linalg.norm(stage2_nl["thetaZ_hat"]))

if stage3a_nl is not None:
    print("\nStage 3a:")
    print("  thetaA_hat_stage3a =", stage3a_nl["thetaA_hat_stage3a"])
    print("  thetaB_hat_stage3a =", stage3a_nl["thetaB_hat_stage3a"])

if stage3b_nl is not None:
    print("\nStage 3b:")
    if "R0_hat" in stage3b_nl:
        print("  R0_hat =", stage3b_nl["R0_hat"])
    if "thetaA_hat_stage3b" in stage3b_nl:
        print("  thetaA_hat_stage3b =", stage3b_nl["thetaA_hat_stage3b"])
    if "thetaB_hat_stage3b" in stage3b_nl:
        print("  thetaB_hat_stage3b =", stage3b_nl["thetaB_hat_stage3b"])
    if "thetaZ_hat" in stage3b_nl:
        print("  ||thetaZ_hat_stage3b||_2 =", np.linalg.norm(stage3b_nl["thetaZ_hat"]))

# %% =====================================================
# CELL 22 — Compact interpretation block
# =====================================================
print("\n" + "=" * 70)
print("NONLINEAR SYNTHETIC-VALIDATION INTERPRETATION")
print("=" * 70)

print("Dataset:")
print("  Source:", synthetic_cycle_meta["source"])
print("  Samples:", synthetic_cycle_meta["n_samples"])
print("  Ts [s]:", synthetic_cycle_meta["Ts"])
print("  Using full_14 proxy for nonlinear ID:", True)

print("\nStage-fit summary:")
print(stage_summary_nl_df)

print("\nSurface summary:")
print("  Truth surface range :", surface_truth_nl.complexity["z_range"])
print("  Learned surface range:", surface_learned_nl.complexity["z_range"])
print("  Surface RMSE        :", surface_compare_nl.rmse)
print("  Surface MAE         :", surface_compare_nl.mae)
print("  Surface max_abs     :", surface_compare_nl.max_abs)
print("  Shape drift         :", surface_compare_nl.drift)

print("\nInterpretation:")
print("  Stage 2 tests whether the fixed nominal 14-state proxy plus learned static nonlinearity")
print("  can explain the frozen synthetic voltage trace.")
print("  Stage 3a tests whether improving the linear dynamics alone reduces the mismatch.")
print("  Stage 3b tests whether joint refinement of dynamics, nonlinear surface, and R0")
print("  can recover the generating system more closely.")
print("  The surface-comparison plots show whether the learned Z(x) matches the truth nonlinear map,")
print("  not just whether the output trace is fitted well in time.")


print("=" * 72)
print("STANDALONE SCRIPT COMPLETE")
print("Figures saved under:", PROJECT_DIR / "results" / "figures" / "nonlinear_synthetic")
print("=" * 72)
