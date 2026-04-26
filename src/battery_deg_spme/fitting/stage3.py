# battery_deg_spme/fitting/stage3.py

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from battery_deg_spme.evaluation.metrics import report_fit, summarize_err
from battery_deg_spme.models.ct_model_wrappers import (
    build_ct_model,
    configure_integration,
    configure_loss,
    configure_optimizer,
)
from battery_deg_spme.models.parameterization import make_builders, pos, thetaA_nom_from_cfg, thetaB_nom_from_cfg


def fit_stage3a_for_cycle(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    stage2_result: dict[str, Any],
    cfg,
    settings,
    dtype=jnp.float64,
):
    X_proxy = proxy["X_proxy"]
    thetaZ_hat = stage2_result["thetaZ_hat"]
    rawR0_hat = stage2_result["rawR0_hat"]
    zhat_from_thetaZ_stage2 = stage2_result["zhat_from_thetaZ"]

    build_A_from_thetaA, build_B_from_thetaB = make_builders(dtype=dtype)

    thetaA_nom_init = thetaA_nom_from_cfg(cfg)
    thetaB_nom_init = thetaB_nom_from_cfg(cfg)

    rawA_init = np.log(np.maximum(thetaA_nom_init, 1e-12))
    rawB_init = thetaB_nom_init.copy()
    rawAB_init = np.concatenate([rawA_init, rawB_init]).astype(np.float64)

    @jax.jit
    def unpack_ab(raw_ab: jnp.ndarray):
        raw_ab = raw_ab.reshape(-1)
        rawA = jnp.clip(raw_ab[0:7], -dtype(settings.surrogate.clip_raw_a), dtype(settings.surrogate.clip_raw_a))
        rawB = jnp.clip(raw_ab[7:11], -dtype(settings.surrogate.clip_raw_b), dtype(settings.surrogate.clip_raw_b))
        thetaA = jnp.exp(rawA)
        thetaB = rawB
        return thetaA, thetaB

    @jax.jit
    def state_fcn_stage3a(x, u, t, params):
        I = u[0]
        (raw_ab,) = params
        thetaA, thetaB = unpack_ab(raw_ab)
        A = build_A_from_thetaA(thetaA)
        B = build_B_from_thetaB(thetaB)
        return A @ x + (B[:, 0] * I)

    @jax.jit
    def output_fcn_stage3a(x, u, t, params):
        I = u[0]
        Z = zhat_from_thetaZ_stage2(x, jnp.array(thetaZ_hat, dtype=dtype))
        R0 = pos(jnp.array(rawR0_hat, dtype=dtype), 1e-12)
        Vhat = Z - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    model_stage3a = build_ct_model(14, 1, 1, state_fcn_stage3a, output_fcn_stage3a)
    model_stage3a.init(params=[rawAB_init], x0=X_proxy[0].astype(np.float64))

    configure_loss(model_stage3a, rho_th=settings.optimization.rho_th_stage3a)
    configure_optimizer(
        model_stage3a,
        adam_epochs=settings.optimization.adam_epochs_stage3a,
        adam_eta=settings.optimization.adam_eta_stage3a,
        lbfgs_epochs=0,
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0
    configure_integration(
        model_stage3a,
        dt=dt,
        dt0_div=settings.solver.dt0_div,
        max_steps=settings.solver.max_steps,
        solver_rtol=settings.solver.solver_rtol,
        solver_atol=settings.solver.solver_atol,
    )

    Y0_stage3a, _ = model_stage3a.predict(model_stage3a.x0, u_np, t_np)
    prefit = report_fit("Stage 3a pre-fit", y_np, Y0_stage3a)

    model_stage3a.fit(y_np, u_np, t_np)

    Yhat_stage3a, _ = model_stage3a.predict(model_stage3a.x0, u_np, t_np)
    postfit = report_fit("Stage 3a post-fit", y_np, Yhat_stage3a)
    metrics = summarize_err(y_np, Yhat_stage3a, name=f"Stage 3a (train, deg={settings.surrogate.poly_deg})")

    rawAB_hat = np.asarray(model_stage3a.params[0]).reshape(-1)
    rawA_hat_stage3a = rawAB_hat[0:7].copy()
    rawB_hat_stage3a = rawAB_hat[7:11].copy()

    thetaA_hat_stage3a = np.exp(rawA_hat_stage3a)
    thetaB_hat_stage3a = rawB_hat_stage3a.copy()

    return {
        "model": model_stage3a,
        "yhat": Yhat_stage3a,
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,
        "rawAB_hat": rawAB_hat,
        "rawA_hat_stage3a": rawA_hat_stage3a,
        "rawB_hat_stage3a": rawB_hat_stage3a,
        "thetaA_hat_stage3a": thetaA_hat_stage3a,
        "thetaB_hat_stage3a": thetaB_hat_stage3a,
    }


def fit_stage3b_for_cycle(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    stage2_result: dict[str, Any],
    stage3a_result: dict[str, Any],
    cfg,
    settings,
    dtype=jnp.float64,
):
    X_proxy = proxy["X_proxy"]
    thetaZ_hat_stage2 = stage2_result["thetaZ_hat"]
    zhat_from_thetaZ_stage2 = stage2_result["zhat_from_thetaZ"]
    rawR0_hat_stage2 = stage2_result["rawR0_hat"]

    rawA_hat_stage3a = stage3a_result["rawA_hat_stage3a"]
    rawB_hat_stage3a = stage3a_result["rawB_hat_stage3a"]

    build_A_from_thetaA, build_B_from_thetaB = make_builders(dtype=dtype)

    n_thetaZ = int(thetaZ_hat_stage2.shape[0])

    raw_init_full = np.concatenate(
        [
            rawA_hat_stage3a.astype(np.float64),
            rawB_hat_stage3a.astype(np.float64),
            thetaZ_hat_stage2.astype(np.float64),
            np.array([rawR0_hat_stage2], dtype=np.float64),
        ]
    ).astype(np.float64)

    def unpack_full(raw_theta: jnp.ndarray):
        raw_theta = raw_theta.reshape(-1)
        rawA = jnp.clip(raw_theta[0:7], -dtype(settings.surrogate.clip_raw_a), dtype(settings.surrogate.clip_raw_a))
        rawB = jnp.clip(raw_theta[7:11], -dtype(settings.surrogate.clip_raw_b), dtype(settings.surrogate.clip_raw_b))
        thetaZ = jnp.clip(
            raw_theta[11:11 + n_thetaZ],
            -dtype(settings.surrogate.clip_raw_z),
            dtype(settings.surrogate.clip_raw_z),
        )
        rawR0 = raw_theta[11 + n_thetaZ]
        thetaA = jnp.exp(rawA)
        thetaB = rawB
        return thetaA, thetaB, thetaZ, rawR0

    @jax.jit
    def state_fcn_stage3b(x, u, t, params):
        I = u[0]
        (raw_theta,) = params
        thetaA, thetaB, _, _ = unpack_full(raw_theta)
        A = build_A_from_thetaA(thetaA)
        B = build_B_from_thetaB(thetaB)
        return A @ x + (B[:, 0] * I)

    @jax.jit
    def output_fcn_stage3b(x, u, t, params):
        I = u[0]
        (raw_theta,) = params
        _, _, thetaZ, rawR0 = unpack_full(raw_theta)
        Z = zhat_from_thetaZ_stage2(x, thetaZ)
        R0 = pos(rawR0, 1e-12)
        Vhat = Z - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    model_stage3b = build_ct_model(14, 1, 1, state_fcn_stage3b, output_fcn_stage3b)
    model_stage3b.init(params=[raw_init_full], x0=X_proxy[0].astype(np.float64))

    configure_loss(model_stage3b, rho_th=settings.optimization.rho_th_stage3b)
    configure_optimizer(
        model_stage3b,
        adam_epochs=settings.optimization.adam_epochs_stage3b,
        adam_eta=settings.optimization.adam_eta_stage3b,
        lbfgs_epochs=0,
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0
    configure_integration(
        model_stage3b,
        dt=dt,
        dt0_div=settings.solver.dt0_div,
        max_steps=settings.solver.max_steps,
        solver_rtol=settings.solver.solver_rtol,
        solver_atol=settings.solver.solver_atol,
    )

    Y0_stage3b, _ = model_stage3b.predict(model_stage3b.x0, u_np, t_np)
    prefit = report_fit("Stage 3b pre-fit", y_np, Y0_stage3b)

    model_stage3b.fit(y_np, u_np, t_np)

    Yhat_stage3b, Xhat_stage3b = model_stage3b.predict(model_stage3b.x0, u_np, t_np)
    postfit = report_fit("Stage 3b post-fit", y_np, Yhat_stage3b)
    metrics = summarize_err(y_np, Yhat_stage3b, name=f"Stage 3b (train, deg={settings.surrogate.poly_deg})")

    raw_theta_hat = np.asarray(model_stage3b.params[0]).reshape(-1)

    rawA_hat_stage3b = raw_theta_hat[0:7].copy()
    rawB_hat_stage3b = raw_theta_hat[7:11].copy()
    thetaZ_hat_stage3b = raw_theta_hat[11:11 + n_thetaZ].copy()
    rawR0_hat_stage3b = float(raw_theta_hat[11 + n_thetaZ])

    thetaA_hat_stage3b = np.exp(rawA_hat_stage3b)
    thetaB_hat_stage3b = rawB_hat_stage3b.copy()
    R0_hat_stage3b = float(np.asarray(pos(jnp.array(rawR0_hat_stage3b, dtype=dtype), 1e-12)))

    return {
        "model": model_stage3b,
        "yhat": Yhat_stage3b,
        "xhat": Xhat_stage3b,
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,
        "raw_theta_hat": raw_theta_hat,
        "rawA_hat_stage3b": rawA_hat_stage3b,
        "rawB_hat_stage3b": rawB_hat_stage3b,
        "thetaA_hat_stage3b": thetaA_hat_stage3b,
        "thetaB_hat_stage3b": thetaB_hat_stage3b,
        "thetaZ_hat": thetaZ_hat_stage3b,
        "rawR0_hat": rawR0_hat_stage3b,
        "R0_hat": R0_hat_stage3b,
    }