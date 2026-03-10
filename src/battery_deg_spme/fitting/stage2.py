from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from battery_deg_spme.evaluation.metrics import report_fit, summarize_err
from battery_deg_spme.fitting.least_squares import compare_ls_solvers
from battery_deg_spme.fitting.optimization import effective_lbfgs_epochs
from battery_deg_spme.models.ct_model_wrappers import (
    build_ct_model,
    configure_integration,
    configure_loss,
    configure_optimizer,
)
from battery_deg_spme.models.parameterization import pos, raw_from_pos
from battery_deg_spme.models.surrogate_polynomial import (
    build_feature_matrix,
    make_additive_poly_surrogate_fns,
)


def fit_stage2_for_cycle(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    cfg,
    settings,
    dtype=jnp.float64,
):
    xp_sig = proxy["xp_sig"]
    xn_sig = proxy["xn_sig"]
    ceL_sig = proxy["ceL_sig"]
    ceR_sig = proxy["ceR_sig"]
    X_proxy = proxy["X_proxy"]
    A_nom_np = proxy["A_nom_np"]
    B_nom_np = proxy["B_nom_np"]

    xp_ref = float(np.mean(xp_sig))
    xn_ref = float(np.mean(xn_sig))
    xp_scale = float(max(np.std(xp_sig), 1e-6))
    xn_scale = float(max(np.std(xn_sig), 1e-6))

    make_thetaZ0, zhat_from_thetaZ_stage2, metaZ = make_additive_poly_surrogate_fns(
        cfg,
        settings.surrogate.poly_deg,
        xp_ref=xp_ref,
        xn_ref=xn_ref,
        xp_scale=xp_scale,
        xn_scale=xn_scale,
        use_ln_feature=settings.surrogate.use_ln_feature,
        dtype=dtype,
        clip_raw_z=settings.surrogate.clip_raw_z,
    )

    Phi = build_feature_matrix(
        xp_sig=xp_sig,
        xn_sig=xn_sig,
        ceL_sig=ceL_sig,
        ceR_sig=ceR_sig,
        deg=settings.surrogate.poly_deg,
        xp_ref=xp_ref,
        xn_ref=xn_ref,
        xp_scale=xp_scale,
        xn_scale=xn_scale,
        use_ln_feature=settings.surrogate.use_ln_feature,
    )

    cmp_ls = compare_ls_solvers(Phi, y_np[:, 0], ridge=1e-12, label="Static LS")
    coef_ls_manual = cmp_ls["beta_qr"]
    Y_ls_manual = Phi @ coef_ls_manual

    Phi_plus_R = np.column_stack([Phi, -u_np[:, 0]])
    cmp_lsr = compare_ls_solvers(Phi_plus_R, y_np[:, 0], ridge=1e-12, label="Static LS+R")
    coef_ls_r_manual = cmp_lsr["beta_qr"]
    Y_ls_r_manual = Phi_plus_R @ coef_ls_r_manual

    thetaZ0 = make_thetaZ0(y_np)
    rawR0_0 = raw_from_pos(0.05, floor=1e-12)

    params_stage2_init = [
        thetaZ0.astype(np.float64),
        np.array([rawR0_0], dtype=np.float64),
    ]

    x0_stage2 = X_proxy[0].astype(np.float64)
    A_nom = jnp.array(A_nom_np, dtype=dtype)
    B_nom = jnp.array(B_nom_np, dtype=dtype)

    @jax.jit
    def state_fcn_stage2(x, u, t, params):
        I = u[0]
        return A_nom @ x + (B_nom[:, 0] * I)

    @jax.jit
    def output_fcn_stage2(x, u, t, params):
        thetaZ, rawR0 = params
        I = u[0]
        Z = zhat_from_thetaZ_stage2(x, thetaZ)
        R0 = pos(rawR0[0], 1e-12)
        Vhat = Z - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    thetaZ_min = -10.0 * np.ones_like(thetaZ0, dtype=np.float64)
    thetaZ_max = 10.0 * np.ones_like(thetaZ0, dtype=np.float64)
    thetaZ_min[0] = float(np.min(y_np[:, 0]) - 0.5)
    thetaZ_max[0] = float(np.max(y_np[:, 0]) + 0.5)

    rawR0_min = np.array([raw_from_pos(1e-4, floor=1e-12)], dtype=np.float64)
    rawR0_max = np.array([raw_from_pos(5.0, floor=1e-12)], dtype=np.float64)

    model_stage2 = build_ct_model(14, 1, 1, state_fcn_stage2, output_fcn_stage2)
    model_stage2.init(params=params_stage2_init, x0=x0_stage2)

    configure_loss(model_stage2, rho_th=settings.optimization.rho_th_stage2)
    configure_optimizer(
        model_stage2,
        adam_epochs=settings.optimization.adam_epochs_stage2,
        adam_eta=settings.optimization.adam_eta_stage2,
        params_min=[thetaZ_min, rawR0_min],
        params_max=[thetaZ_max, rawR0_max],
        lbfgs_epochs=effective_lbfgs_epochs(
            settings.optimization.use_lbfgs,
            settings.optimization.lbfgs_epochs,
        ),
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0
    configure_integration(
        model_stage2,
        dt=dt,
        dt0_div=settings.solver.dt0_div,
        max_steps=settings.solver.max_steps,
        solver_rtol=settings.solver.solver_rtol,
        solver_atol=settings.solver.solver_atol,
    )

    Y0_stage2, _ = model_stage2.predict(model_stage2.x0, u_np, t_np)
    prefit = report_fit("Stage 2 pre-fit", y_np, Y0_stage2)

    model_stage2.fit(y_np, u_np, t_np)

    Yhat_stage2, Xhat_stage2 = model_stage2.predict(model_stage2.x0, u_np, t_np)
    postfit = report_fit("Stage 2 post-fit", y_np, Yhat_stage2)
    metrics = summarize_err(y_np, Yhat_stage2, name=f"Stage 2 (train, deg={settings.surrogate.poly_deg})")

    thetaZ_hat = np.asarray(model_stage2.params[0]).reshape(-1)
    rawR0_hat = float(np.asarray(model_stage2.params[1]).reshape(-1)[0])
    R0_hat = float(np.asarray(pos(jnp.array(rawR0_hat, dtype=dtype), 1e-12)))

    return {
        "model": model_stage2,
        "yhat": Yhat_stage2,
        "xhat": Xhat_stage2,
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,
        "metaZ": metaZ,
        "Phi": Phi,
        "static_ls": cmp_ls,
        "static_lsr": cmp_lsr,
        "Y_ls_manual": Y_ls_manual.reshape(-1, 1),
        "Y_ls_r_manual": Y_ls_r_manual.reshape(-1, 1),
        "thetaZ_hat": thetaZ_hat,
        "rawR0_hat": rawR0_hat,
        "R0_hat": R0_hat,
        "zhat_from_thetaZ": zhat_from_thetaZ_stage2,
        "center_scale": {
            "xp_ref": xp_ref,
            "xn_ref": xn_ref,
            "xp_scale": xp_scale,
            "xn_scale": xn_scale,
        },
    }