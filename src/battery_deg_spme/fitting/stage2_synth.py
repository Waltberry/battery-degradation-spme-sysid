# battery_deg_spme/fitting/stage2_synth.py

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
from battery_deg_spme.models.parameterization_synth import pos, raw_from_pos
from battery_deg_spme.models.surrogate_polynomial import (
    build_feature_matrix,
    make_additive_poly_surrogate_fns,
)


def _safe_attr(obj, name: str, default):
    return getattr(obj, name, default)


def _estimate_step_r0_bound(
    *,
    cfg,
    u_np: np.ndarray,
    y_np: np.ndarray,
    y_span: float,
    u_abs_max: float,
) -> dict[str, float | int | None]:
    """
    Estimate a safe R0 upper bound from the first current step.

    The immediate voltage edge is mainly:
        Delta V_edge ~= -N_series * Delta I * R0

    This function gives Stage 2 enough room to capture the sharp drop,
    while still capping R0 to avoid unphysical overfitting.
    """
    r0_truth_scale = (
        float(getattr(cfg, "R_ohm", 1e-3))
        + float(getattr(cfg, "Rf", 1e-3))
    )

    u_flat = np.asarray(u_np[:, 0], dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np[:, 0], dtype=np.float64).reshape(-1)

    u0_est = float(np.median(u_flat[: max(3, min(10, len(u_flat)))]))
    du_abs = np.abs(u_flat - u0_est)

    step_threshold = max(
        1e-9,
        0.05 * max(float(np.max(np.abs(u_flat))), 1e-12),
    )

    step_candidates = np.where(du_abs > step_threshold)[0]

    if len(step_candidates) > 0:
        k_step = int(step_candidates[0])
        k_pre0 = max(0, k_step - 5)

        if k_step > k_pre0:
            y_pre = float(np.mean(y_flat[k_pre0:k_step]))
        else:
            y_pre = float(y_flat[0])

        y_post = float(y_flat[k_step])
        du_step = float(abs(u_flat[k_step] - u0_est))

        if du_step > 1e-12:
            # Use 95% of the first observed jump, not 70%.
            # This gives enough room to capture the sharp edge.
            r0_from_step = 0.95 * abs(y_post - y_pre) / (
                float(cfg.N_series) * du_step
            )
        else:
            r0_from_step = np.inf

    else:
        k_step = None
        y_pre = np.nan
        y_post = np.nan
        du_step = np.nan
        r0_from_step = np.inf

    # The synthetic truth has R_ohm + Rf, but the learned R0 may need
    # a larger effective value because the comparison synthetic has been
    # shifted/scaled and the static surrogate also shares the drop.
    r0_from_truth = 5.0 * r0_truth_scale

    # Allow sharper edge than the old 0.008 cap, but still bound it.
    r0_upper_guess = min(
        0.012,
        max(0.003, min(r0_from_truth, r0_from_step)),
    )

    return {
        "r0_truth_scale": float(r0_truth_scale),
        "r0_from_truth": float(r0_from_truth),
        "r0_from_step": float(r0_from_step),
        "r0_upper_guess": float(r0_upper_guess),
        "k_step": k_step,
        "y_pre": float(y_pre),
        "y_post": float(y_post),
        "du_step": float(du_step),
        "y_span": float(y_span),
        "u_abs_max": float(u_abs_max),
    }


def fit_stage2_for_cycle_synth(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    cfg,
    settings,
    dtype=jnp.float64,
):
    """
    Synthetic-only Stage 2 fit.

    Differences vs standard stage2.py:
    - keeps the same model structure;
    - anchors the learned static surface at the initial rest state;
    - uses synthetic-safe R0 bounds;
    - gives R0 enough room to capture the immediate current-step edge;
    - remains drop-in compatible with the notebook adapter.
    """
    t_np = np.asarray(t_np, dtype=np.float64).reshape(-1)
    u_np = np.asarray(u_np, dtype=np.float64)
    y_np = np.asarray(y_np, dtype=np.float64)

    if u_np.ndim == 1:
        u_np = u_np.reshape(-1, 1)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    xp_sig = np.asarray(proxy["xp_sig"], dtype=np.float64).reshape(-1)
    xn_sig = np.asarray(proxy["xn_sig"], dtype=np.float64).reshape(-1)
    ceL_sig = np.asarray(proxy["ceL_sig"], dtype=np.float64).reshape(-1)
    ceR_sig = np.asarray(proxy["ceR_sig"], dtype=np.float64).reshape(-1)

    X_proxy = np.asarray(proxy["X_proxy"], dtype=np.float64)
    A_nom_np = np.asarray(proxy["A_nom_np"], dtype=np.float64)
    B_nom_np = np.asarray(proxy["B_nom_np"], dtype=np.float64)

    xp_ref = float(np.mean(xp_sig))
    xn_ref = float(np.mean(xn_sig))
    xp_scale = float(max(np.std(xp_sig), 1e-6))
    xn_scale = float(max(np.std(xn_sig), 1e-6))

    clip_raw_z = float(_safe_attr(settings.surrogate, "clip_raw_z", 20.0))

    make_thetaZ0, zhat_from_thetaZ_stage2, metaZ = make_additive_poly_surrogate_fns(
        cfg,
        int(settings.surrogate.poly_deg),
        xp_ref=xp_ref,
        xn_ref=xn_ref,
        xp_scale=xp_scale,
        xn_scale=xn_scale,
        use_ln_feature=bool(settings.surrogate.use_ln_feature),
        dtype=dtype,
        clip_raw_z=clip_raw_z,
    )

    Phi = build_feature_matrix(
        xp_sig=xp_sig,
        xn_sig=xn_sig,
        ceL_sig=ceL_sig,
        ceR_sig=ceR_sig,
        deg=int(settings.surrogate.poly_deg),
        xp_ref=xp_ref,
        xn_ref=xn_ref,
        xp_scale=xp_scale,
        xn_scale=xn_scale,
        use_ln_feature=bool(settings.surrogate.use_ln_feature),
    )

    cmp_ls = compare_ls_solvers(
        Phi,
        y_np[:, 0],
        ridge=1e-12,
        label="Static LS (synth)",
    )
    coef_ls_manual = cmp_ls["beta_qr"]
    Y_ls_manual = Phi @ coef_ls_manual

    Phi_plus_R = np.column_stack([Phi, -u_np[:, 0]])
    cmp_lsr = compare_ls_solvers(
        Phi_plus_R,
        y_np[:, 0],
        ridge=1e-12,
        label="Static LS+R (synth)",
    )
    coef_ls_r_manual = cmp_lsr["beta_qr"]
    Y_ls_r_manual = Phi_plus_R @ coef_ls_r_manual

    thetaZ0 = np.asarray(make_thetaZ0(y_np), dtype=np.float64).reshape(-1)

    # Initial R0. This is only an initialization; bounds are set below.
    rawR0_0 = raw_from_pos(0.005, floor=1e-12)

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

    # Anchor the learned static surface at the initial rest state.
    # The nonlinear notebook fits centered voltage, so the model must
    # output zero at the initial state when I = 0.
    x0_anchor_stage2 = jnp.array(x0_stage2, dtype=dtype)

    @jax.jit
    def output_fcn_stage2(x, u, t, params):
        thetaZ, rawR0 = params
        I = u[0]

        Z = zhat_from_thetaZ_stage2(x, thetaZ)
        Z0 = zhat_from_thetaZ_stage2(x0_anchor_stage2, thetaZ)

        R0 = pos(rawR0[0], 1e-12)

        Vhat = (Z - Z0) - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    # ---------------------------------------------------------
    # Synthetic-safe parameter bounds
    # ---------------------------------------------------------
    y_min = float(np.min(y_np[:, 0]))
    y_max = float(np.max(y_np[:, 0]))
    y_span = float(max(y_max - y_min, 1e-6))
    u_abs_max = float(max(np.max(np.abs(u_np[:, 0])), 1e-9))

    thetaZ_min = -4.0 * np.ones_like(thetaZ0, dtype=np.float64)
    thetaZ_max = +4.0 * np.ones_like(thetaZ0, dtype=np.float64)

    # Let constant term cover observed centered-voltage range with margin.
    thetaZ_min[0] = y_min - 0.25 * max(1.0, y_span)
    thetaZ_max[0] = y_max + 0.25 * max(1.0, y_span)

    # ---------------------------------------------------------
    # R0 bound for centered nonlinear synthetic ID
    # ---------------------------------------------------------
    r0_info = _estimate_step_r0_bound(
        cfg=cfg,
        u_np=u_np,
        y_np=y_np,
        y_span=y_span,
        u_abs_max=u_abs_max,
    )

    r0_upper_guess = float(r0_info["r0_upper_guess"])

    rawR0_min = np.array(
        [raw_from_pos(1e-6, floor=1e-12)],
        dtype=np.float64,
    )
    rawR0_max = np.array(
        [raw_from_pos(r0_upper_guess, floor=1e-12)],
        dtype=np.float64,
    )

    # Ensure initial R0 is inside bounds.
    params_stage2_init[1] = np.clip(
        params_stage2_init[1],
        rawR0_min + 1e-9,
        rawR0_max - 1e-9,
    )

    print("Stage 2 synthetic R0 bound:")
    print("  y_span:", r0_info["y_span"])
    print("  u_abs_max:", r0_info["u_abs_max"])
    print("  r0_truth_scale:", r0_info["r0_truth_scale"])
    print("  r0_from_truth:", r0_info["r0_from_truth"])
    print("  r0_from_step:", r0_info["r0_from_step"])
    print("  r0_upper_guess:", r0_info["r0_upper_guess"])
    print("  detected step index:", r0_info["k_step"])
    print("  y_pre:", r0_info["y_pre"])
    print("  y_post:", r0_info["y_post"])
    print("  du_step:", r0_info["du_step"])
    print("  rawR0_min:", rawR0_min)
    print("  rawR0_max:", rawR0_max)

    model_stage2 = build_ct_model(14, 1, 1, state_fcn_stage2, output_fcn_stage2)
    model_stage2.init(params=params_stage2_init, x0=x0_stage2)

    configure_loss(
        model_stage2,
        rho_th=float(settings.optimization.rho_th_stage2),
    )

    configure_optimizer(
        model_stage2,
        adam_epochs=int(settings.optimization.adam_epochs_stage2),
        adam_eta=float(settings.optimization.adam_eta_stage2),
        params_min=[thetaZ_min, rawR0_min],
        params_max=[thetaZ_max, rawR0_max],
        lbfgs_epochs=effective_lbfgs_epochs(
            bool(settings.optimization.use_lbfgs),
            int(settings.optimization.lbfgs_epochs),
        ),
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0

    configure_integration(
        model_stage2,
        dt=dt,
        dt0_div=float(settings.solver.dt0_div),
        max_steps=int(settings.solver.max_steps),
        solver_rtol=float(settings.solver.solver_rtol),
        solver_atol=float(settings.solver.solver_atol),
    )

    Y0_stage2, _ = model_stage2.predict(model_stage2.x0, u_np, t_np)
    prefit = report_fit("Stage 2 pre-fit [synth]", y_np, Y0_stage2)

    model_stage2.fit(y_np, u_np, t_np)

    Yhat_stage2, Xhat_stage2 = model_stage2.predict(model_stage2.x0, u_np, t_np)

    postfit = report_fit("Stage 2 post-fit [synth]", y_np, Yhat_stage2)

    metrics = summarize_err(
        y_np,
        Yhat_stage2,
        name=f"Stage 2 synth (train, deg={settings.surrogate.poly_deg})",
    )

    thetaZ_hat = np.asarray(model_stage2.params[0], dtype=np.float64).reshape(-1)
    rawR0_hat = float(
        np.asarray(model_stage2.params[1], dtype=np.float64).reshape(-1)[0]
    )
    R0_hat = float(np.asarray(pos(jnp.array(rawR0_hat, dtype=dtype), 1e-12)))

    return {
        "model": model_stage2,
        "yhat": np.asarray(Yhat_stage2, dtype=np.float64),
        "xhat": np.asarray(Xhat_stage2, dtype=np.float64),
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,
        "metaZ": metaZ,
        "Phi": Phi,
        "static_ls": cmp_ls,
        "static_lsr": cmp_lsr,
        "Y_ls_manual": np.asarray(Y_ls_manual, dtype=np.float64).reshape(-1, 1),
        "Y_ls_r_manual": np.asarray(Y_ls_r_manual, dtype=np.float64).reshape(-1, 1),
        "thetaZ_hat": thetaZ_hat,
        "rawR0_hat": rawR0_hat,
        "R0_hat": R0_hat,
        "r0_bound_info": r0_info,
        "zhat_from_thetaZ": zhat_from_thetaZ_stage2,
        "xp_ref": xp_ref,
        "xn_ref": xn_ref,
        "xp_scale": xp_scale,
        "xn_scale": xn_scale,
    }