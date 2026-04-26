# src/battery_deg_spme/model/synthetic_truth
from __future__ import annotations

from typing import Any

import control as ct
import numpy as np

from .spme_proxy import Config, IDX, assemble_system, make_x0


def ocp_p(xp: np.ndarray) -> np.ndarray:
    x = np.clip(xp, 1e-9, 1.0 - 1e-9)
    return 4.15 - 0.12 * np.tanh((x - 0.60) / 0.08)


def ocp_n(xn: np.ndarray) -> np.ndarray:
    x = np.clip(xn, 1e-9, 1.0 - 1e-9)
    return 0.10 + 0.80 * (1.0 / (1.0 + np.exp(-(x - 0.50) / 0.04)))


def _arrhenius(k0: float, Ea: float, cfg: Config) -> float:
    if (not cfg.use_arrhenius) or (Ea == 0.0):
        return k0
    return k0 * np.exp(-Ea / cfg.R * (1.0 / cfg.T - 1.0 / cfg.T_ref))


def i0_current_scales(
    xp: float,
    xn: float,
    ceL: float,
    ceR: float,
    cfg: Config,
) -> tuple[float, float]:
    ce_avg = 0.5 * (ceL + ceR)
    ce_avg = float(np.clip(ce_avg, 1e-12, 10.0 * cfg.ce0))

    xp_eff = float(np.clip(xp, cfg.theta_guard, 1.0 - cfg.theta_guard))
    xn_eff = float(np.clip(xn, cfg.theta_guard, 1.0 - cfg.theta_guard))

    Sp = (1.0 - cfg.lam_p) * cfg.a_s_p * cfg.A * cfg.L3
    Sn = (1.0 - cfg.lam_n) * cfg.a_s_n * cfg.A * cfg.L1

    kp = _arrhenius(cfg.k_p0, cfg.Ea_p, cfg)
    kn = _arrhenius(cfg.k_n0, cfg.Ea_n, cfg)

    i0p = cfg.F * kp * cfg.csp_max * np.sqrt(ce_avg) * np.sqrt(xp_eff * (1.0 - xp_eff))
    i0n = cfg.F * kn * cfg.csn_max * np.sqrt(ce_avg) * np.sqrt(xn_eff * (1.0 - xn_eff))

    I0p = max(float(Sp * i0p), cfg.I0_floor_p)
    I0n = max(float(Sn * i0n), cfg.I0_floor_n)
    return I0p, I0n


def electrolyte_log_term(ceL: float, ceR: float, cfg: Config) -> float:
    ceL = max(float(ceL), 1e-12)
    ceR = max(float(ceR), 1e-12)
    ln_arg = (ceR / ceL) if (cfg.ln_orientation == "right_over_left") else (ceL / ceR)
    return (2.0 * cfg.R * cfg.T / cfg.F) * (1.0 - cfg.t_plus) * cfg.k_f * np.log(ln_arg)


def electrolyte_resistance(cfg: Config) -> float:
    return (
        cfg.L1 / cfg.kappa_n_eff
        + 2.0 * cfg.L2 / cfg.kappa_s_eff
        + cfg.L3 / cfg.kappa_p_eff
    ) / (2.0 * cfg.A)


def film_resistance(cfg: Config) -> float:
    if cfg.use_dynamic_film and cfg.L_sei > 0.0:
        return cfg.L_sei / (cfg.kappa_sei * cfg.a_s_n * cfg.A * cfg.L1)
    return cfg.Rf


def truth_z_from_state(x: np.ndarray, cfg: Config, I: float) -> float:
    """
    Pack-scale nonlinear truth Z(x) = Ns * [(Up-Un) + BV + dphi_e]
    This EXCLUDES ohmic drop so it matches the surrogate target.
    """
    xp = np.clip(x[IDX["cp_surf"]] / cfg.csp_max, 1e-9, 1.0 - 1e-9)
    xn = np.clip(x[IDX["cn_surf"]] / cfg.csn_max, 1e-9, 1.0 - 1e-9)

    Up = float(ocp_p(np.array([xp]))[0])
    Un = float(ocp_n(np.array([xn]))[0])

    ceL_raw = float(x[IDX["ce_left"]])
    ceR_raw = float(x[IDX["ce_right"]])

    ceL = (cfg.ce0 + ceL_raw) if cfg.ce_is_deviation else ceL_raw
    ceR = (cfg.ce0 + ceR_raw) if cfg.ce_is_deviation else ceR_raw

    I0p, I0n = i0_current_scales(xp, xn, ceL, ceR, cfg)

    eta_p = (2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(I / (2.0 * max(I0p, 1e-20)))
    eta_n = (2.0 * cfg.R * cfg.T / cfg.F) * np.arcsinh(I / (2.0 * max(I0n, 1e-20)))

    eta_p *= cfg.bv_scale
    eta_n *= cfg.bv_scale

    eta_combo = (eta_p - eta_n) if (cfg.eta_mode == "diff") else (eta_p + eta_n)
    dphi_e = electrolyte_log_term(ceL, ceR, cfg)

    z_cell = (Up - Un) + eta_combo + dphi_e
    return float(cfg.N_series) * float(z_cell)


def truth_z_from_xn_xp(
    cfg: Config,
    xn: float,
    xp: float,
    I: float,
    ceL: float | None = None,
    ceR: float | None = None,
) -> float:
    x = np.zeros(14, dtype=np.float64)
    x[IDX["cn_surf"]] = float(xn) * cfg.csn_max
    x[IDX["cp_surf"]] = float(xp) * cfg.csp_max

    if ceL is None:
        ceL = float(cfg.ce0)
    if ceR is None:
        ceR = float(cfg.ce0)

    if cfg.ce_is_deviation:
        x[IDX["ce_left"]] = float(ceL) - float(cfg.ce0)
        x[IDX["ce_right"]] = float(ceR) - float(cfg.ce0)
    else:
        x[IDX["ce_left"]] = float(ceL)
        x[IDX["ce_right"]] = float(ceR)

    return truth_z_from_state(x, cfg, I=I)


def terminal_voltage_truth(x: np.ndarray, cfg: Config, I: float | None = None) -> float:
    I_use = float(cfg.I_for_voltage if I is None else I)

    z_pack = truth_z_from_state(x, cfg, I=I_use)
    ohmic = -I_use * (cfg.R_ohm + electrolyte_resistance(cfg) + film_resistance(cfg))
    return float(z_pack + cfg.N_series * ohmic)


def battery_update(t, x, u, params):
    A = params["A"]
    B = params["B"]
    cfg = params["cfg"]

    if u is None or (hasattr(u, "size") and u.size == 0):
        I = float(cfg.I_dyn)
    else:
        I = float(np.asarray(u).reshape(-1)[0])

    return A @ x + B[:, 0] * I


def battery_output(t, x, u, params):
    cfg = params["cfg"]

    if u is None or (hasattr(u, "size") and u.size == 0):
        I = float(cfg.I_dyn)
    else:
        I = float(np.asarray(u).reshape(-1)[0])

    V = terminal_voltage_truth(x, cfg, I=I)
    return np.hstack([x, V])


def generate_discharge_data(
    cfg: Config,
    I_const: float = 2.0,
    sim_t_end: float = 25.0,
    sim_dt: float = 0.1,
    theta_n0: float = 0.8,
    theta_p0: float = 0.4,
    ce0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T = np.arange(0.0, sim_t_end + sim_dt, sim_dt, dtype=np.float64)

    cfg.I_dyn = float(I_const)
    cfg.I_for_voltage = float(I_const)

    _, A, B = assemble_system(cfg)
    nl_params = {"A": A, "B": B, "cfg": cfg}

    battery_nl = ct.nlsys(
        battery_update,
        battery_output,
        name="battery_truth_discharge",
        params=nl_params,
        states=14,
        outputs=15,
        inputs=0,
    )

    x0 = make_x0(cfg, theta_n0=theta_n0, theta_p0=theta_p0, ce0=ce0)
    resp = ct.input_output_response(battery_nl, T, 0, X0=x0)

    X = resp.states.T
    Y_full = resp.outputs.T
    V = Y_full[:, -1:].astype(np.float64)
    U = np.full((len(T), 1), float(I_const), dtype=np.float64)

    return T, U, X, V, Y_full


def generate_profile_data(
    cfg: Config,
    I_profile: np.ndarray,
    T: np.ndarray,
    theta_n0: float = 0.8,
    theta_p0: float = 0.4,
    ce0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I_profile = np.asarray(I_profile, dtype=np.float64).reshape(-1)
    T = np.asarray(T, dtype=np.float64).reshape(-1)

    if I_profile.shape[0] != T.shape[0]:
        raise ValueError("I_profile and T must have the same length.")

    cfg.I_for_voltage = float(I_profile[0])

    _, A, B = assemble_system(cfg)
    nl_params = {"A": A, "B": B, "cfg": cfg}

    def _update(t, x, u, params):
        A = params["A"]
        B = params["B"]
        I = float(np.asarray(u).reshape(-1)[0])
        return A @ x + B[:, 0] * I

    def _output(t, x, u, params):
        cfg = params["cfg"]
        I = float(np.asarray(u).reshape(-1)[0])
        V = terminal_voltage_truth(x, cfg, I=I)
        return np.hstack([x, V])

    battery_nl = ct.nlsys(
        _update,
        _output,
        name="battery_truth_profile",
        params=nl_params,
        states=14,
        outputs=15,
        inputs=1,
    )

    x0 = make_x0(cfg, theta_n0=theta_n0, theta_p0=theta_p0, ce0=ce0)
    resp = ct.input_output_response(battery_nl, T, I_profile.reshape(1, -1), X0=x0)

    X = resp.states.T
    Y_full = resp.outputs.T
    V = Y_full[:, -1:].astype(np.float64)
    U = I_profile.reshape(-1, 1)

    return T, U, X, V, Y_full