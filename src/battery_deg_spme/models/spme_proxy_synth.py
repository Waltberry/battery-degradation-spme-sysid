# battery_deg_spme/models/spme_proxy_synth.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import control as ct
import numpy as np
from scipy.linalg import block_diag

from .state_space_builders import build_Ae, build_An, build_Ap, build_Be, build_Bn, build_Bp


IDX = {
    "cn": slice(0, 4),
    "cp": slice(4, 8),
    "ce": slice(8, 14),
    "cn_surf": 3,
    "cp_surf": 7,
    "ce_left": 8,
    "ce_right": 13,
}


@dataclass
class Config:
    R: float = 8.314462618
    F: float = 96485.33212
    T: float = 298.15
    T_ref: float = 298.15

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
    k_n0: float = 2.0e-11
    k_p0: float = 2.0e-11
    use_arrhenius: bool = False
    Ea_n: float = 0.0
    Ea_p: float = 0.0

    lam_n: float = 0.0
    lam_p: float = 0.0

    csn_max: float = 3.1e4
    csp_max: float = 3.1e4

    ce0: float = 1000.0
    t_plus: float = 0.38
    k_f: float = 1.0
    R_ohm: float = 0.0
    use_dynamic_film: bool = False
    Rf: float = 0.0
    L_sei: float = 0.0
    kappa_sei: float = 1.0

    ce_is_deviation: bool = True
    discharge_positive: bool = False
    ln_orientation: str = "right_over_left"
    eta_mode: str = "diff"

    I_dyn: float = 2.0
    I_for_voltage: float = 2.0

    theta_guard: float = 1e-3
    I0_floor_p: float = 1e-2
    I0_floor_n: float = 1e-2
    bv_scale: float = 0.7
    N_series: int = 1

    # -----------------------------------------------------
    # SYNTHETIC-ONLY additions
    # These must mirror the notebook truth generator when
    # USE_SOLID_STOICH_RATE_SCALE = True.
    # -----------------------------------------------------
    use_solid_stoich_rate_scale: bool = False
    solid_stoich_rate_scale: float = 1.0


def _solid_scale(cfg: Config) -> float:
    if bool(getattr(cfg, "use_solid_stoich_rate_scale", False)):
        return float(getattr(cfg, "solid_stoich_rate_scale", 1.0))
    return 1.0


def assemble_system(cfg: Config):
    An = build_An(cfg)
    Ap = build_Ap(cfg)
    Ae = build_Ae(cfg)

    Bn = build_Bn(cfg).astype(np.float64, copy=True)
    Bp = build_Bp(cfg).astype(np.float64, copy=True)
    Be = build_Be(cfg).astype(np.float64, copy=True)

    ssolid = _solid_scale(cfg)

    # Only solid-state input channels get the synthetic scaling.
    Bn *= ssolid
    Bp *= ssolid

    Aglob = block_diag(An, Ap, Ae)
    Bglob = np.vstack([Bn, Bp, Be])

    S = ct.ss(
        Aglob,
        Bglob,
        np.eye(14, dtype=np.float64),
        np.zeros((14, 1), dtype=np.float64),
    )
    return S, Aglob, Bglob


def make_x0(
    cfg: Config,
    theta_n0: float = 0.60,
    theta_p0: float = 0.60,
    ce0: float = 0.0,
):
    x0 = np.zeros(14, dtype=np.float64)
    x0[IDX["cn"]] = float(theta_n0) * cfg.csn_max
    x0[IDX["cp"]] = float(theta_p0) * cfg.csp_max
    x0[IDX["ce"]] = float(ce0)
    return x0


def build_proxy_signals(
    t_np: np.ndarray,
    u_np: np.ndarray,
    cfg: Config,
    xn0: float = 0.60,
    xp0: float = 0.60,
    ce0_dev: float = 0.0,
) -> dict[str, Any]:
    t_np = np.asarray(t_np, dtype=np.float64).reshape(-1)
    u_np = np.asarray(u_np, dtype=np.float64)
    if u_np.ndim == 1:
        u_np = u_np.reshape(-1, 1)

    if len(t_np) != len(u_np):
        raise ValueError(
            f"Length mismatch in build_proxy_signals: len(t_np)={len(t_np)} "
            f"but len(u_np)={len(u_np)}"
        )

    Sx, A_nom_np, B_nom_np = assemble_system(cfg)
    x0_nom = make_x0(cfg, theta_n0=xn0, theta_p0=xp0, ce0=ce0_dev)

    resp = ct.forced_response(Sx, T=t_np, U=u_np[:, 0], X0=x0_nom)
    X_proxy = np.asarray(resp.states, dtype=np.float64).T

    xp_sig = (X_proxy[:, IDX["cp_surf"]] / cfg.csp_max).astype(np.float64)
    xn_sig = (X_proxy[:, IDX["cn_surf"]] / cfg.csn_max).astype(np.float64)

    ceL_raw_sig = X_proxy[:, IDX["ce_left"]].astype(np.float64)
    ceR_raw_sig = X_proxy[:, IDX["ce_right"]].astype(np.float64)

    if cfg.ce_is_deviation:
        ceL_sig = cfg.ce0 + ceL_raw_sig
        ceR_sig = cfg.ce0 + ceR_raw_sig
    else:
        ceL_sig = ceL_raw_sig
        ceR_sig = ceR_raw_sig

    if str(cfg.ln_orientation).lower() == "left_over_right":
        ln_ce_ratio_sig = np.log(
            np.maximum(ceL_sig / np.maximum(ceR_sig, 1e-12), 1e-12)
        )
    else:
        ln_ce_ratio_sig = np.log(
            np.maximum(ceR_sig / np.maximum(ceL_sig, 1e-12), 1e-12)
        )

    return {
        "Sx": Sx,
        "A_nom_np": A_nom_np,
        "B_nom_np": B_nom_np,
        "x0_nom": x0_nom,
        "X_proxy": X_proxy,
        "xp_sig": xp_sig,
        "xn_sig": xn_sig,
        "ceL_sig": ceL_sig,
        "ceR_sig": ceR_sig,
        "ln_ce_ratio_sig": ln_ce_ratio_sig,
        "xp_rng": float(xp_sig.max() - xp_sig.min()) if xp_sig.size else 0.0,
        "xn_rng": float(xn_sig.max() - xn_sig.min()) if xn_sig.size else 0.0,
        "ceL_rng": float(ceL_sig.max() - ceL_sig.min()) if ceL_sig.size else 0.0,
        "ceR_rng": float(ceR_sig.max() - ceR_sig.min()) if ceR_sig.size else 0.0,
    }