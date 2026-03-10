#src/battery_deg_spme/evaluation/degree_comparison.py

from __future__ import annotations

from typing import List, Dict

import numpy as np

from battery_deg_spme.models.surrogate_polynomial import make_additive_poly_surrogate_fns, build_feature_matrix
from battery_deg_spme.models.parameterization import raw_from_pos, pos
from battery_deg_spme.models.ct_model_wrappers import build_ct_model, configure_loss, configure_optimizer, configure_integration
from battery_deg_spme.fitting.least_squares import solve_ls_qr
from battery_deg_spme.fitting.optimization import effective_lbfgs_epochs
from battery_deg_spme.evaluation.metrics import summarize_err
import jax
import jax.numpy as jnp


def run_stage2_for_degree_real(
    t_np,
    u_np,
    y_np,
    proxy,
    cfg,
    settings,
    deg: int,
    use_ln_feature: bool = False,
    adam_epochs: int = 600,
):
    xp_sig = proxy["xp_sig"]
    xn_sig = proxy["xn_sig"]
    ceL_sig = proxy["ceL_sig"]
    ceR_sig = proxy["ceR_sig"]
    X_proxy = proxy["X_proxy"]
    A_nom_np = proxy["A_nom_np"]
    B_nom_np = proxy["B_nom_np"]

    xp_ref_d = float(np.mean(xp_sig))
    xn_ref_d = float(np.mean(xn_sig))
    xp_scale_d = float(max(np.std(xp_sig), 1e-6))
    xn_scale_d = float(max(np.std(xn_sig), 1e-6))

    make_thetaZ0_d, zhat_from_thetaZ_d, meta = make_additive_poly_surrogate_fns(
        cfg,
        deg,
        xp_ref=xp_ref_d,
        xn_ref=xn_ref_d,
        xp_scale=xp_scale_d,
        xn_scale=xn_scale_d,
        use_ln_feature=use_ln_feature,
        dtype=jnp.float64,
        clip_raw_z=settings.surrogate.clip_raw_z,
    )

    Phi_d = build_feature_matrix(
        xp_sig=xp_sig,
        xn_sig=xn_sig,
        ceL_sig=ceL_sig,
        ceR_sig=ceR_sig,
        deg=deg,
        xp_ref=xp_ref_d,
        xn_ref=xn_ref_d,
        xp_scale=xp_scale_d,
        xn_scale=xn_scale_d,
        use_ln_feature=use_ln_feature,
    )

    beta_ls_d = solve_ls_qr(Phi_d, y_np[:, 0])
    Y_ls_d = Phi_d @ beta_ls_d

    Phi_plus_R_d = np.column_stack([Phi_d, -u_np[:, 0]])
    beta_lsr_d = solve_ls_qr(Phi_plus_R_d, y_np[:, 0])
    Y_ls_r_d = Phi_plus_R_d @ beta_lsr_d

    thetaZ0_d = make_thetaZ0_d(y_np)
    rawR0_0_d = raw_from_pos(0.05, floor=1e-12)

    params0_d = [
        thetaZ0_d.astype(np.float64),
        np.array([rawR0_0_d], dtype=np.float64),
    ]

    x0_d = X_proxy[0].astype(np.float64)

    A_nom = jnp.array(A_nom_np, dtype=jnp.float64)
    B_nom = jnp.array(B_nom_np, dtype=jnp.float64)

    @jax.jit
    def state_fcn_d(x, u, t, params):
        I = u[0]
        return A_nom @ x + (B_nom[:, 0] * I)

    @jax.jit
    def output_fcn_d(x, u, t, params):
        thetaZ, rawR0 = params
        I = u[0]
        Z = zhat_from_thetaZ_d(x, thetaZ)
        R0 = pos(rawR0[0], 1e-12)
        Vhat = Z - jnp.float64(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=jnp.float64)

    thetaZ_min_d = -10.0 * np.ones_like(thetaZ0_d, dtype=np.float64)
    thetaZ_max_d =  10.0 * np.ones_like(thetaZ0_d, dtype=np.float64)
    thetaZ_min_d[0] = float(np.min(y_np[:, 0]) - 0.5)
    thetaZ_max_d[0] = float(np.max(y_np[:, 0]) + 0.5)

    rawR0_min_d = np.array([raw_from_pos(1e-4, floor=1e-12)], dtype=np.float64)
    rawR0_max_d = np.array([raw_from_pos(5.0, floor=1e-12)], dtype=np.float64)

    m = build_ct_model(14, 1, 1, state_fcn_d, output_fcn_d)
    m.init(params=params0_d, x0=x0_d)

    configure_loss(m, rho_th=settings.optimization.rho_th_stage2)
    configure_optimizer(
        m,
        adam_epochs=adam_epochs,
        adam_eta=settings.optimization.adam_eta_stage2,
        params_min=[thetaZ_min_d, rawR0_min_d],
        params_max=[thetaZ_max_d, rawR0_max_d],
        lbfgs_epochs=effective_lbfgs_epochs(
            settings.optimization.use_lbfgs,
            settings.optimization.lbfgs_epochs,
        ),
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0
    configure_integration(
        m,
        dt=dt,
        dt0_div=settings.solver.dt0_div,
        max_steps=settings.solver.max_steps,
        solver_rtol=settings.solver.solver_rtol,
        solver_atol=settings.solver.solver_atol,
    )

    m.fit(y_np, u_np, t_np)
    Yhat_d, _ = m.predict(m.x0, u_np, t_np)

    ct_metrics = summarize_err(y_np, Yhat_d, name=f"CT Stage 2 train (deg={deg})")
    ls_metrics = summarize_err(y_np, Y_ls_d.reshape(-1, 1), name=f"Static LS train (deg={deg})")
    lsr_metrics = summarize_err(y_np, Y_ls_r_d.reshape(-1, 1), name=f"Static LS+R train (deg={deg})")

    return dict(
        deg=deg,
        n_thetaZ=meta["n_thetaZ"],
        ct=ct_metrics,
        ls=ls_metrics,
        lsr=lsr_metrics,
        structure=meta["structure"],
    )


def run_degree_sweep(
    t_np,
    u_np,
    y_np,
    proxy,
    cfg,
    settings,
) -> List[Dict]:
    sweep = []
    for d in settings.surrogate.degree_sweep:
        sweep.append(
            run_stage2_for_degree_real(
                t_np=t_np,
                u_np=u_np,
                y_np=y_np,
                proxy=proxy,
                cfg=cfg,
                settings=settings,
                deg=d,
                use_ln_feature=settings.surrogate.use_ln_feature,
                adam_epochs=600,
            )
        )
    return sweep