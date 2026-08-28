# battery_deg_spme/fitting/stage3_synth.py

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from battery_deg_spme.evaluation.metrics import report_fit, summarize_err
from battery_deg_spme.fitting.optimization import effective_lbfgs_epochs
from battery_deg_spme.models.ct_model_wrappers import (
    build_ct_model,
    configure_integration,
    configure_loss,
    configure_optimizer,
)
from battery_deg_spme.models.parameterization_synth import (
    make_builders,
    pos,
    raw_from_pos,
    thetaA_nom_from_cfg,
    thetaB_nom_from_cfg,
)


def _safe_attr(obj, name: str, default):
    return getattr(obj, name, default)


def _stage_lbfgs_epochs(settings, stage_name: str) -> int:
    """
    Return the LBFGS budget for a specific Stage 3 substage.

    Priority:
      1) settings.optimization.lbfgs_epochs_stage3a / lbfgs_epochs_stage3b, if present
      2) settings.optimization.lbfgs_epochs
      3) zero if settings.optimization.use_lbfgs is False
    """
    opt = settings.optimization

    use_lbfgs = bool(getattr(opt, "use_lbfgs", False))
    if not use_lbfgs:
        return 0

    stage_attr = f"lbfgs_epochs_{stage_name}"
    stage_epochs = int(getattr(opt, stage_attr, getattr(opt, "lbfgs_epochs", 0)))

    return effective_lbfgs_epochs(use_lbfgs, stage_epochs)

def _assemble_stage_dynamics_from_raw(
    rawA: np.ndarray,
    rawB: np.ndarray,
    build_A_from_thetaA,
    build_B_from_thetaB,
    *,
    clip_raw_a: float,
    clip_raw_b: float,
    dtype=jnp.float64,
) -> dict[str, np.ndarray]:
    """
    Reconstruct the assembled continuous-time dynamic matrices from
    Stage 3 raw dynamic parameters.

    This is required for pole recovery.

    Stage 3 internally parameterizes the linear dynamic block using:
        rawA -> thetaA = exp(rawA)
        rawB -> thetaB = rawB

    The actual continuous-time poles are not eigenvalues of rawA.
    They are eigenvalues of the assembled matrix:

        A_hat = build_A_from_thetaA(thetaA_hat)

    Returns
    -------
    dict
        Contains raw vectors, positive/physical theta vectors,
        assembled A/B matrices, and continuous-time poles.
    """
    rawA_np = np.asarray(rawA, dtype=np.float64).reshape(-1)
    rawB_np = np.asarray(rawB, dtype=np.float64).reshape(-1)

    rawA_np = np.clip(rawA_np, -float(clip_raw_a), float(clip_raw_a))
    rawB_np = np.clip(rawB_np, -float(clip_raw_b), float(clip_raw_b))

    thetaA_np = np.exp(rawA_np).astype(np.float64)
    thetaB_np = rawB_np.astype(np.float64)

    thetaA_jax = jnp.array(thetaA_np, dtype=dtype)
    thetaB_jax = jnp.array(thetaB_np, dtype=dtype)

    A_hat = np.asarray(build_A_from_thetaA(thetaA_jax), dtype=np.float64)
    B_hat = np.asarray(build_B_from_thetaB(thetaB_jax), dtype=np.float64)

    if A_hat.ndim != 2 or A_hat.shape[0] != A_hat.shape[1]:
        raise ValueError(
            f"Reconstructed A_hat is not square. Got shape {A_hat.shape}."
        )

    if not np.all(np.isfinite(A_hat)):
        raise ValueError("Reconstructed A_hat contains non-finite values.")

    if not np.all(np.isfinite(B_hat)):
        raise ValueError("Reconstructed B_hat contains non-finite values.")

    poles_ct = np.linalg.eigvals(A_hat)

    return {
        "rawA_clipped": rawA_np,
        "rawB_clipped": rawB_np,
        "thetaA_hat": thetaA_np,
        "thetaB_hat": thetaB_np,
        "A_hat": A_hat,
        "B_hat": B_hat,
        "poles_ct": poles_ct,
    }

def _build_stage3_bounds(
    cfg,
    settings,
    thetaZ_hat_stage2: np.ndarray,
    y_np: np.ndarray,
    u_np: np.ndarray,
):
    """
    Build safe Stage 3 bounds.

    Important:
    - Stage 3a optimizes rawA/rawB only.
    - Stage 3b optimizes rawA/rawB/thetaZ/R0 jointly.
    - Bounds must always satisfy lower <= upper.
    - Initial guesses must lie inside the bounds, otherwise scipy/JAX
      optimizers may fail before fitting starts.
    """
    user_clip_raw_a = float(_safe_attr(settings.surrogate, "clip_raw_a", 20.0))
    user_clip_raw_b = float(_safe_attr(settings.surrogate, "clip_raw_b", 20.0))
    user_clip_raw_z = float(_safe_attr(settings.surrogate, "clip_raw_z", 20.0))

    thetaA_nom = thetaA_nom_from_cfg(cfg).astype(np.float64).reshape(-1)
    thetaB_nom = thetaB_nom_from_cfg(cfg).astype(np.float64).reshape(-1)

    rawA_nom = np.log(np.maximum(thetaA_nom, 1e-12))
    rawB_nom = thetaB_nom.copy()

    # ---------------------------------------------------------
    # Effective clips
    # ---------------------------------------------------------
    # If user_clip_raw_a is too small, rawA_nom may sit outside
    # [-clip, +clip], causing upper < lower.
    # Therefore expand the effective clip to include the nominal
    # dynamics plus a small window.
    # Very small dynamics movement for Stage 3.
    # Stage 2 is already good, so Stage 3 must behave like a refinement,
    # not a full re-identification.
    rawA_window = 0.10
    rawB_rel_window = 0.05
    rawB_abs_window = 1e-5

    effective_clip_raw_a = max(
        user_clip_raw_a,
        float(np.max(np.abs(rawA_nom))) + rawA_window + 1e-6,
    )

    effective_clip_raw_b = max(
        user_clip_raw_b,
        float(np.max(np.abs(rawB_nom))) + rawB_abs_window + 1e-6,
    )

    clip_raw_z = user_clip_raw_z

    # ---------------------------------------------------------
    # rawA bounds: small window around nominal log-thetaA
    # ---------------------------------------------------------
    rawA_center = np.clip(
        rawA_nom,
        -effective_clip_raw_a + 1e-9,
        +effective_clip_raw_a - 1e-9,
    )

    rawA_min = np.maximum(rawA_center - rawA_window, -effective_clip_raw_a)
    rawA_max = np.minimum(rawA_center + rawA_window, +effective_clip_raw_a)

    # ---------------------------------------------------------
    # rawB bounds: relative + absolute window around nominal thetaB
    # ---------------------------------------------------------
    rawB_window = np.maximum(
        rawB_abs_window,
        rawB_rel_window * np.maximum(np.abs(rawB_nom), rawB_abs_window),
    )

    rawB_center = np.clip(
        rawB_nom,
        -effective_clip_raw_b + 1e-9,
        +effective_clip_raw_b - 1e-9,
    )

    rawB_min = np.maximum(rawB_center - rawB_window, -effective_clip_raw_b)
    rawB_max = np.minimum(rawB_center + rawB_window, +effective_clip_raw_b)

    # ---------------------------------------------------------
    # thetaZ bounds: tight window around Stage 2 learned surface
    # ---------------------------------------------------------
    thetaZ_hat_stage2 = np.asarray(thetaZ_hat_stage2, dtype=np.float64).reshape(-1)

    y_min = float(np.min(y_np[:, 0]))
    y_max = float(np.max(y_np[:, 0]))
    y_span = float(max(y_max - y_min, 1e-6))

    thetaZ_window = 0.01

    thetaZ_center = np.clip(
        thetaZ_hat_stage2,
        -clip_raw_z + 1e-9,
        +clip_raw_z - 1e-9,
    )

    thetaZ_min = np.maximum(thetaZ_center - thetaZ_window, -clip_raw_z)
    thetaZ_max = np.minimum(thetaZ_center + thetaZ_window, +clip_raw_z)

    # Keep constant term flexible enough for centered-voltage range.
    thetaZ_min[0] = y_min - 0.25 * max(1.0, y_span)
    thetaZ_max[0] = y_max + 0.25 * max(1.0, y_span)

    # ---------------------------------------------------------
    # R0 bounds: match safe Stage 2 logic
    # ---------------------------------------------------------
    r0_truth_scale = (
        float(getattr(cfg, "R_ohm", 1e-3))
        + float(getattr(cfg, "Rf", 1e-3))
    )

    u_flat = np.asarray(u_np[:, 0], dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np[:, 0], dtype=np.float64).reshape(-1)

    u0_est = float(np.median(u_flat[:max(3, min(10, len(u_flat)))]))
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
            r0_from_step = 0.70 * abs(y_post - y_pre) / (
                float(cfg.N_series) * du_step
            )
        else:
            r0_from_step = np.inf
    else:
        r0_from_step = np.inf

    r0_from_truth = 4.0 * r0_truth_scale

    r0_upper_guess = min(
        0.008,
        max(0.003, min(r0_from_truth, r0_from_step)),
    )

    rawR0_min = raw_from_pos(1e-6, floor=1e-12)
    rawR0_max = raw_from_pos(r0_upper_guess, floor=1e-12)

    # ---------------------------------------------------------
    # Final safety check
    # ---------------------------------------------------------
    def _check_bounds(name: str, lo: np.ndarray, hi: np.ndarray):
        lo = np.asarray(lo, dtype=np.float64).reshape(-1)
        hi = np.asarray(hi, dtype=np.float64).reshape(-1)

        bad = np.where(hi < lo)[0]
        if len(bad) > 0:
            print(f"\nBAD BOUNDS in {name}")
            print("  bad indices:", bad)
            print("  lower[bad]:", lo[bad])
            print("  upper[bad]:", hi[bad])
            raise ValueError(f"{name}: upper bound is less than lower bound.")

    _check_bounds("rawA", rawA_min, rawA_max)
    _check_bounds("rawB", rawB_min, rawB_max)
    _check_bounds("thetaZ", thetaZ_min, thetaZ_max)

    if rawR0_max < rawR0_min:
        raise ValueError(
            f"rawR0 bounds invalid: rawR0_min={rawR0_min}, rawR0_max={rawR0_max}"
        )

    print("Stage 3 bounds summary:")
    print("  user_clip_raw_a:", user_clip_raw_a)
    print("  effective_clip_raw_a:", effective_clip_raw_a)
    print("  user_clip_raw_b:", user_clip_raw_b)
    print("  effective_clip_raw_b:", effective_clip_raw_b)
    print("  clip_raw_z:", clip_raw_z)
    print("  rawA min/max:", float(np.min(rawA_min)), float(np.max(rawA_max)))
    print("  rawB min/max:", float(np.min(rawB_min)), float(np.max(rawB_max)))
    print("  thetaZ min/max:", float(np.min(thetaZ_min)), float(np.max(thetaZ_max)))
    print("  r0_truth_scale:", r0_truth_scale)
    print("  r0_from_truth:", r0_from_truth)
    print("  r0_from_step:", r0_from_step)
    print("  r0_upper_guess:", r0_upper_guess)

    return {
        "rawA_min": rawA_min.astype(np.float64),
        "rawA_max": rawA_max.astype(np.float64),
        "rawB_min": rawB_min.astype(np.float64),
        "rawB_max": rawB_max.astype(np.float64),
        "thetaZ_min": thetaZ_min.astype(np.float64),
        "thetaZ_max": thetaZ_max.astype(np.float64),
        "rawR0_min": float(rawR0_min),
        "rawR0_max": float(rawR0_max),
    }


def fit_stage3a_for_cycle_synth(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    stage2_result: dict[str, Any],
    cfg,
    settings,
    dtype=jnp.float64,
):
    """
    Synthetic-only Stage 3a fit.
    Same model structure as standard stage3.py, but with bounds.
    """
    t_np = np.asarray(t_np, dtype=np.float64).reshape(-1)
    u_np = np.asarray(u_np, dtype=np.float64)
    y_np = np.asarray(y_np, dtype=np.float64)

    if u_np.ndim == 1:
        u_np = u_np.reshape(-1, 1)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    X_proxy = np.asarray(proxy["X_proxy"], dtype=np.float64)
    thetaZ_hat = np.asarray(stage2_result["thetaZ_hat"], dtype=np.float64).reshape(-1)
    rawR0_hat = float(stage2_result["rawR0_hat"])
    zhat_from_thetaZ_stage2 = stage2_result["zhat_from_thetaZ"]

    build_A_from_thetaA, build_B_from_thetaB = make_builders(dtype=dtype)

    thetaA_nom_init = thetaA_nom_from_cfg(cfg)
    thetaB_nom_init = thetaB_nom_from_cfg(cfg)

    rawA_init = np.log(np.maximum(thetaA_nom_init, 1e-12))
    rawB_init = thetaB_nom_init.copy()
    rawAB_init = np.concatenate([rawA_init, rawB_init]).astype(np.float64)

    bounds = _build_stage3_bounds(
        cfg=cfg,
        settings=settings,
        thetaZ_hat_stage2=thetaZ_hat,
        y_np=y_np,
        u_np=u_np,
    )

    rawAB_min = np.concatenate([bounds["rawA_min"], bounds["rawB_min"]]).astype(np.float64)
    rawAB_max = np.concatenate([bounds["rawA_max"], bounds["rawB_max"]]).astype(np.float64)

    # Ensure initial point is inside optimizer bounds.
    rawAB_init = np.clip(rawAB_init, rawAB_min + 1e-9, rawAB_max - 1e-9)

    print("Stage 3a bound sanity:")
    print("  any rawAB_max < rawAB_min:", bool(np.any(rawAB_max < rawAB_min)))
    print("  rawAB_init inside bounds:", bool(np.all((rawAB_init >= rawAB_min) & (rawAB_init <= rawAB_max))))
    print("  min margin lower:", float(np.min(rawAB_init - rawAB_min)))
    print("  min margin upper:", float(np.min(rawAB_max - rawAB_init)))

    clip_raw_a = float(_safe_attr(settings.surrogate, "clip_raw_a", 20.0))
    clip_raw_b = float(_safe_attr(settings.surrogate, "clip_raw_b", 20.0))

    @jax.jit
    def unpack_ab(raw_ab: jnp.ndarray):
        raw_ab = raw_ab.reshape(-1)
        rawA = jnp.clip(raw_ab[0:7], -dtype(clip_raw_a), dtype(clip_raw_a))
        rawB = jnp.clip(raw_ab[7:11], -dtype(clip_raw_b), dtype(clip_raw_b))
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

    x0_anchor_stage3a = jnp.array(X_proxy[0].astype(np.float64), dtype=dtype)
    thetaZ_hat_jax_stage3a = jnp.array(thetaZ_hat, dtype=dtype)

    @jax.jit
    def output_fcn_stage3a(x, u, t, params):
        I = u[0]

        Z = zhat_from_thetaZ_stage2(x, thetaZ_hat_jax_stage3a)
        Z0 = zhat_from_thetaZ_stage2(x0_anchor_stage3a, thetaZ_hat_jax_stage3a)

        R0 = pos(jnp.array(rawR0_hat, dtype=dtype), 1e-12)

        Vhat = (Z - Z0) - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    model_stage3a = build_ct_model(14, 1, 1, state_fcn_stage3a, output_fcn_stage3a)
    model_stage3a.init(params=[rawAB_init], x0=X_proxy[0].astype(np.float64))

    configure_loss(model_stage3a, rho_th=float(settings.optimization.rho_th_stage3a))
    stage3a_lbfgs_epochs = _stage_lbfgs_epochs(settings, "stage3a")

    print("Stage 3a optimizer:")
    print("  adam_epochs:", int(settings.optimization.adam_epochs_stage3a))
    print("  adam_eta:", float(settings.optimization.adam_eta_stage3a))
    print("  lbfgs_epochs:", stage3a_lbfgs_epochs)


    configure_optimizer(
        model_stage3a,
        adam_epochs=int(settings.optimization.adam_epochs_stage3a),
        adam_eta=float(settings.optimization.adam_eta_stage3a),
        params_min=[rawAB_min],
        params_max=[rawAB_max],
        lbfgs_epochs=stage3a_lbfgs_epochs,
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0
    configure_integration(
        model_stage3a,
        dt=dt,
        dt0_div=float(settings.solver.dt0_div),
        max_steps=int(settings.solver.max_steps),
        solver_rtol=float(settings.solver.solver_rtol),
        solver_atol=float(settings.solver.solver_atol),
    )

    Y0_stage3a, _ = model_stage3a.predict(model_stage3a.x0, u_np, t_np)
    prefit = report_fit("Stage 3a pre-fit [synth]", y_np, Y0_stage3a)

    model_stage3a.fit(y_np, u_np, t_np)

    Yhat_stage3a, Xhat_stage3a = model_stage3a.predict(model_stage3a.x0, u_np, t_np)
    postfit = report_fit("Stage 3a post-fit [synth]", y_np, Yhat_stage3a)
    metrics = summarize_err(
        y_np,
        Yhat_stage3a,
        name=f"Stage 3a synth (train, deg={settings.surrogate.poly_deg})",
    )

    rawAB_hat = np.asarray(model_stage3a.params[0], dtype=np.float64).reshape(-1)
    rawA_hat_stage3a = rawAB_hat[0:7].copy()
    rawB_hat_stage3a = rawAB_hat[7:11].copy()

    thetaA_hat_stage3a = np.exp(rawA_hat_stage3a)
    thetaB_hat_stage3a = rawB_hat_stage3a.copy()

    dyn_hat_stage3a = _assemble_stage_dynamics_from_raw(
        rawA=rawA_hat_stage3a,
        rawB=rawB_hat_stage3a,
        build_A_from_thetaA=build_A_from_thetaA,
        build_B_from_thetaB=build_B_from_thetaB,
        clip_raw_a=clip_raw_a,
        clip_raw_b=clip_raw_b,
        dtype=dtype,
    )

    A_hat_stage3a = dyn_hat_stage3a["A_hat"]
    B_hat_stage3a = dyn_hat_stage3a["B_hat"]
    poles_ct_stage3a = dyn_hat_stage3a["poles_ct"]

    print("[INFO] Stage 3a assembled A_hat_stage3a shape:", A_hat_stage3a.shape)
    print("[INFO] Stage 3a estimated CT poles from eig(A_hat_stage3a):")
    print(poles_ct_stage3a)

    # return {
    #     "model": model_stage3a,
    return {
        "model": model_stage3a,
        "yhat": np.asarray(Yhat_stage3a, dtype=np.float64),
        "xhat": np.asarray(Xhat_stage3a, dtype=np.float64),
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,
        "rawAB_hat": rawAB_hat,
        "rawA_hat_stage3a": rawA_hat_stage3a,
        "rawB_hat_stage3a": rawB_hat_stage3a,
        "thetaA_hat_stage3a": thetaA_hat_stage3a,
        "thetaB_hat_stage3a": thetaB_hat_stage3a,
        "A_hat_stage3a": A_hat_stage3a,
        "B_hat_stage3a": B_hat_stage3a,
        "poles_ct_stage3a": poles_ct_stage3a,
        "A_hat_stage3a_source": "reconstructed_from_rawA_rawB_inside_fit_stage3a_for_cycle_synth",
        "poles_ct_stage3a_source": "eig(A_hat_stage3a)",
    }


def fit_stage3b_for_cycle_synth(
    t_np: np.ndarray,
    u_np: np.ndarray,
    y_np: np.ndarray,
    proxy: dict[str, Any],
    stage2_result: dict[str, Any],
    stage3a_result: dict[str, Any] | None,
    cfg,
    settings,
    dtype=jnp.float64,
):
    """
    Synthetic-only Stage 3b fit.

    This version is safer than the original because:
    - it can initialize from Stage 3a if Stage 3a is accepted;
    - it can initialize from nominal / Stage 2 dynamics if Stage 3a is rejected;
    - it anchors the learned static voltage surface at the initial state;
    - it clips the initial parameter vector inside optimizer bounds;
    - it uses the Stage 3b LBFGS budget from settings instead of hard-coding zero;
    - it reconstructs and returns the assembled continuous-time A_hat/B_hat
      matrices so Stage 3b poles can be computed correctly.

    Important pole interpretation:
    - rawA_hat_stage3b is NOT the pole vector.
    - thetaA_hat_stage3b = exp(rawA_hat_stage3b).
    - A_hat_stage3b = build_A_from_thetaA(thetaA_hat_stage3b).
    - Stage 3b poles are eig(A_hat_stage3b).
    """
    t_np = np.asarray(t_np, dtype=np.float64).reshape(-1)
    u_np = np.asarray(u_np, dtype=np.float64)
    y_np = np.asarray(y_np, dtype=np.float64)

    if u_np.ndim == 1:
        u_np = u_np.reshape(-1, 1)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    X_proxy = np.asarray(proxy["X_proxy"], dtype=np.float64)

    thetaZ_hat_stage2 = np.asarray(
        stage2_result["thetaZ_hat"],
        dtype=np.float64,
    ).reshape(-1)

    zhat_from_thetaZ_stage2 = stage2_result["zhat_from_thetaZ"]
    rawR0_hat_stage2 = float(stage2_result["rawR0_hat"])

    # ---------------------------------------------------------
    # Choose Stage 3b initialization source
    # ---------------------------------------------------------
    if stage3a_result is not None:
        rawA_init = np.asarray(
            stage3a_result["rawA_hat_stage3a"],
            dtype=np.float64,
        ).reshape(-1)

        rawB_init = np.asarray(
            stage3a_result["rawB_hat_stage3a"],
            dtype=np.float64,
        ).reshape(-1)

        init_source = "stage3a"
        print("Stage 3b init source: Stage 3a")

    else:
        thetaA_nom_init = thetaA_nom_from_cfg(cfg)
        thetaB_nom_init = thetaB_nom_from_cfg(cfg)

        rawA_init = np.log(np.maximum(thetaA_nom_init, 1e-12)).astype(np.float64)
        rawB_init = thetaB_nom_init.astype(np.float64)

        init_source = "nominal_stage2"
        print("Stage 3b init source: nominal / Stage 2 dynamics")

    build_A_from_thetaA, build_B_from_thetaB = make_builders(dtype=dtype)

    n_thetaZ = int(thetaZ_hat_stage2.shape[0])

    bounds = _build_stage3_bounds(
        cfg=cfg,
        settings=settings,
        thetaZ_hat_stage2=thetaZ_hat_stage2,
        y_np=y_np,
        u_np=u_np,
    )

    raw_init_full = np.concatenate(
        [
            rawA_init.astype(np.float64),
            rawB_init.astype(np.float64),
            thetaZ_hat_stage2.astype(np.float64),
            np.array([rawR0_hat_stage2], dtype=np.float64),
        ]
    ).astype(np.float64)

    raw_full_min = np.concatenate(
        [
            bounds["rawA_min"],
            bounds["rawB_min"],
            bounds["thetaZ_min"],
            np.array([bounds["rawR0_min"]], dtype=np.float64),
        ]
    ).astype(np.float64)

    raw_full_max = np.concatenate(
        [
            bounds["rawA_max"],
            bounds["rawB_max"],
            bounds["thetaZ_max"],
            np.array([bounds["rawR0_max"]], dtype=np.float64),
        ]
    ).astype(np.float64)

    # ---------------------------------------------------------
    # Safety check and initial clipping
    # ---------------------------------------------------------
    bad_bounds = np.where(raw_full_max < raw_full_min)[0]
    if len(bad_bounds) > 0:
        print("\nBAD BOUNDS in Stage 3b full vector")
        print("  bad indices:", bad_bounds)
        print("  lower[bad]:", raw_full_min[bad_bounds])
        print("  upper[bad]:", raw_full_max[bad_bounds])
        raise ValueError("Stage 3b: upper bound is less than lower bound.")

    raw_init_full = np.clip(
        raw_init_full,
        raw_full_min + 1e-9,
        raw_full_max - 1e-9,
    )

    print("Stage 3b bound sanity:")
    print("  any raw_full_max < raw_full_min:", bool(np.any(raw_full_max < raw_full_min)))
    print(
        "  raw_init_full inside bounds:",
        bool(np.all((raw_init_full >= raw_full_min) & (raw_init_full <= raw_full_max))),
    )
    print("  min margin lower:", float(np.min(raw_init_full - raw_full_min)))
    print("  min margin upper:", float(np.min(raw_full_max - raw_init_full)))

    clip_raw_a = float(_safe_attr(settings.surrogate, "clip_raw_a", 20.0))
    clip_raw_b = float(_safe_attr(settings.surrogate, "clip_raw_b", 20.0))
    clip_raw_z = float(_safe_attr(settings.surrogate, "clip_raw_z", 20.0))

    @jax.jit
    def unpack_full(raw_theta: jnp.ndarray):
        raw_theta = raw_theta.reshape(-1)

        rawA = jnp.clip(
            raw_theta[0:7],
            -dtype(clip_raw_a),
            dtype(clip_raw_a),
        )

        rawB = jnp.clip(
            raw_theta[7:11],
            -dtype(clip_raw_b),
            dtype(clip_raw_b),
        )

        thetaZ = jnp.clip(
            raw_theta[11:11 + n_thetaZ],
            -dtype(clip_raw_z),
            dtype(clip_raw_z),
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

    # Anchor Stage 3b static surface at initial state.
    x0_anchor_stage3b = jnp.array(X_proxy[0].astype(np.float64), dtype=dtype)

    @jax.jit
    def output_fcn_stage3b(x, u, t, params):
        I = u[0]
        (raw_theta,) = params

        _, _, thetaZ, rawR0 = unpack_full(raw_theta)

        Z = zhat_from_thetaZ_stage2(x, thetaZ)
        Z0 = zhat_from_thetaZ_stage2(x0_anchor_stage3b, thetaZ)

        R0 = pos(rawR0, 1e-12)

        Vhat = (Z - Z0) - dtype(cfg.N_series) * I * R0
        return jnp.array([Vhat], dtype=dtype)

    model_stage3b = build_ct_model(14, 1, 1, state_fcn_stage3b, output_fcn_stage3b)
    model_stage3b.init(params=[raw_init_full], x0=X_proxy[0].astype(np.float64))

    configure_loss(
        model_stage3b,
        rho_th=float(settings.optimization.rho_th_stage3b),
    )

    stage3b_lbfgs_epochs = _stage_lbfgs_epochs(settings, "stage3b")

    print("Stage 3b optimizer:")
    print("  adam_epochs:", int(settings.optimization.adam_epochs_stage3b))
    print("  adam_eta:", float(settings.optimization.adam_eta_stage3b))
    print("  lbfgs_epochs:", stage3b_lbfgs_epochs)

    configure_optimizer(
        model_stage3b,
        adam_epochs=int(settings.optimization.adam_epochs_stage3b),
        adam_eta=float(settings.optimization.adam_eta_stage3b),
        params_min=[raw_full_min],
        params_max=[raw_full_max],
        lbfgs_epochs=stage3b_lbfgs_epochs,
    )

    dt = float(t_np[1] - t_np[0]) if len(t_np) > 1 else 1.0

    configure_integration(
        model_stage3b,
        dt=dt,
        dt0_div=float(settings.solver.dt0_div),
        max_steps=int(settings.solver.max_steps),
        solver_rtol=float(settings.solver.solver_rtol),
        solver_atol=float(settings.solver.solver_atol),
    )

    Y0_stage3b, _ = model_stage3b.predict(model_stage3b.x0, u_np, t_np)
    prefit = report_fit("Stage 3b pre-fit [synth]", y_np, Y0_stage3b)

    model_stage3b.fit(y_np, u_np, t_np)

    Yhat_stage3b, Xhat_stage3b = model_stage3b.predict(model_stage3b.x0, u_np, t_np)

    postfit = report_fit("Stage 3b post-fit [synth]", y_np, Yhat_stage3b)

    metrics = summarize_err(
        y_np,
        Yhat_stage3b,
        name="Stage 3b synth",
    )

    # ---------------------------------------------------------
    # Extract fitted Stage 3b parameters
    # ---------------------------------------------------------
    raw_hat_full = np.asarray(model_stage3b.params[0], dtype=np.float64).reshape(-1)

    rawA_hat = raw_hat_full[0:7].copy()
    rawB_hat = raw_hat_full[7:11].copy()
    thetaZ_hat = raw_hat_full[11:11 + n_thetaZ].copy()
    rawR0_hat = float(raw_hat_full[11 + n_thetaZ])

    R0_hat = float(np.asarray(pos(jnp.array(rawR0_hat, dtype=dtype), 1e-12)))

    # ---------------------------------------------------------
    # NEW: reconstruct assembled Stage 3b A/B and poles
    # ---------------------------------------------------------
    dyn_hat = _assemble_stage_dynamics_from_raw(
        rawA=rawA_hat,
        rawB=rawB_hat,
        build_A_from_thetaA=build_A_from_thetaA,
        build_B_from_thetaB=build_B_from_thetaB,
        clip_raw_a=clip_raw_a,
        clip_raw_b=clip_raw_b,
        dtype=dtype,
    )

    thetaA_hat = dyn_hat["thetaA_hat"]
    thetaB_hat = dyn_hat["thetaB_hat"]
    A_hat_stage3b = dyn_hat["A_hat"]
    B_hat_stage3b = dyn_hat["B_hat"]
    poles_ct_stage3b = dyn_hat["poles_ct"]

    print("[INFO] Stage 3b assembled A_hat_stage3b shape:", A_hat_stage3b.shape)
    print("[INFO] Stage 3b assembled B_hat_stage3b shape:", B_hat_stage3b.shape)
    print("[INFO] Stage 3b estimated CT poles from eig(A_hat_stage3b):")
    print(poles_ct_stage3b)

    return {
        "model": model_stage3b,
        "yhat": np.asarray(Yhat_stage3b, dtype=np.float64),
        "xhat": np.asarray(Xhat_stage3b, dtype=np.float64),
        "prefit": prefit,
        "postfit": postfit,
        "err_summary": metrics,
        "metrics": metrics,

        # Raw fitted Stage 3b vector
        "raw_full_hat_stage3b": raw_hat_full,
        "rawA_hat_stage3b": rawA_hat,
        "rawB_hat_stage3b": rawB_hat,
        "thetaZ_hat": thetaZ_hat,
        "rawR0_hat": rawR0_hat,
        "R0_hat": R0_hat,

        # Reconstructed physical/dynamic parameters
        "thetaA_hat_stage3b": thetaA_hat,
        "thetaB_hat_stage3b": thetaB_hat,

        # Assembled continuous-time linear block
        "A_hat_stage3b": A_hat_stage3b,
        "B_hat_stage3b": B_hat_stage3b,

        # Estimated continuous-time poles
        "poles_ct_stage3b": poles_ct_stage3b,
        "A_hat_stage3b_source": "reconstructed_from_rawA_rawB_inside_fit_stage3b_for_cycle_synth",
        "poles_ct_stage3b_source": "eig(A_hat_stage3b)",

        "init_source": init_source,
    }