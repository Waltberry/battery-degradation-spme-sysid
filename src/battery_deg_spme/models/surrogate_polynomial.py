from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .spme_proxy import IDX


def build_feature_matrix(
    xp_sig: np.ndarray,
    xn_sig: np.ndarray,
    ceL_sig: np.ndarray,
    ceR_sig: np.ndarray,
    deg: int,
    xp_ref: float,
    xn_ref: float,
    xp_scale: float,
    xn_scale: float,
    use_ln_feature: bool = False,
) -> np.ndarray:
    xp_sig = np.asarray(xp_sig, dtype=np.float64).reshape(-1)
    xn_sig = np.asarray(xn_sig, dtype=np.float64).reshape(-1)
    ceL_sig = np.asarray(ceL_sig, dtype=np.float64).reshape(-1)
    ceR_sig = np.asarray(ceR_sig, dtype=np.float64).reshape(-1)

    dxp = (xp_sig - float(xp_ref)) / float(max(xp_scale, 1e-12))
    dxn = (xn_sig - float(xn_ref)) / float(max(xn_scale, 1e-12))

    cols = [np.ones_like(dxp)]
    for k in range(1, deg + 1):
        cols.append(dxp ** k)
    for k in range(1, deg + 1):
        cols.append(dxn ** k)

    if use_ln_feature:
        ln_ratio = np.log(np.maximum(ceR_sig / np.maximum(ceL_sig, 1e-12), 1e-12))
        cols.append(ln_ratio)

    return np.column_stack(cols)


def make_additive_poly_surrogate_fns(
    cfg,
    deg: int,
    xp_ref: float,
    xn_ref: float,
    xp_scale: float,
    xn_scale: float,
    use_ln_feature: bool = False,
    dtype=jnp.float64,
    clip_raw_z: float = 50.0,
):
    xp_powers = jnp.arange(1, deg + 1, dtype=dtype)
    xn_powers = jnp.arange(1, deg + 1, dtype=dtype)

    xp_ref_j = dtype(float(xp_ref))
    xn_ref_j = dtype(float(xn_ref))
    xp_scale_j = dtype(float(max(xp_scale, 1e-12)))
    xn_scale_j = dtype(float(max(xn_scale, 1e-12)))

    @jax.jit
    def additive_features_from_x(x14: jnp.ndarray) -> jnp.ndarray:
        xp = jnp.clip(x14[IDX["cp_surf"]] / dtype(cfg.csp_max), 1e-9, 1.0 - 1e-9)
        xn = jnp.clip(x14[IDX["cn_surf"]] / dtype(cfg.csn_max), 1e-9, 1.0 - 1e-9)

        dxp = (xp - xp_ref_j) / xp_scale_j
        dxn = (xn - xn_ref_j) / xn_scale_j

        xp_feats = dxp ** xp_powers
        xn_feats = dxn ** xn_powers

        return jnp.concatenate(
            [jnp.ones((1,), dtype=dtype), xp_feats, xn_feats],
            axis=0,
        )

    @jax.jit
    def ln_ce_ratio_from_x(x14: jnp.ndarray) -> jnp.ndarray:
        ceL_raw = x14[IDX["ce_left"]]
        ceR_raw = x14[IDX["ce_right"]]

        ceL = (dtype(cfg.ce0) + ceL_raw) if cfg.ce_is_deviation else ceL_raw
        ceR = (dtype(cfg.ce0) + ceR_raw) if cfg.ce_is_deviation else ceR_raw

        ceL = jnp.maximum(ceL, 1e-12)
        ceR = jnp.maximum(ceR, 1e-12)

        ln_arg = (ceR / ceL) if (cfg.ln_orientation == "right_over_left") else (ceL / ceR)
        return jnp.log(jnp.maximum(ln_arg, 1e-12))

    n_base = 1 + deg + deg
    n_thetaZ = n_base + (1 if use_ln_feature else 0)

    def unpack_thetaZ(thetaZ: jnp.ndarray):
        thetaZ = thetaZ.reshape(-1)
        theta_base = thetaZ[:n_base]

        if use_ln_feature:
            k_ln = thetaZ[n_base]
            return theta_base, k_ln
        return theta_base, None

    @jax.jit
    def zhat_from_thetaZ(x14: jnp.ndarray, thetaZ: jnp.ndarray) -> jnp.ndarray:
        thetaZ = jnp.clip(thetaZ.reshape(-1), -dtype(clip_raw_z), dtype(clip_raw_z))
        phi = additive_features_from_x(x14)
        theta_base, k_ln = unpack_thetaZ(thetaZ)

        z = jnp.dot(phi, theta_base)

        if use_ln_feature:
            lnratio = ln_ce_ratio_from_x(x14)
            z = z + k_ln * lnratio

        return z

    def make_thetaZ0_from_voltage(y_np: np.ndarray) -> np.ndarray:
        y_np = np.asarray(y_np, dtype=np.float64)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

        thetaZ0 = np.zeros(n_thetaZ, dtype=np.float64)
        thetaZ0[0] = float(np.mean(y_np[:, 0]))
        return thetaZ0

    meta = {
        "deg": deg,
        "n_thetaZ": n_thetaZ,
        "use_ln_feature": use_ln_feature,
        "xp_ref": float(xp_ref),
        "xn_ref": float(xn_ref),
        "xp_scale": float(xp_scale),
        "xn_scale": float(xn_scale),
        "structure": "c0 + sum a_k*dxp^k + sum b_k*dxn^k [+ k_ln ln(ceR/ceL)]",
    }

    return make_thetaZ0_from_voltage, zhat_from_thetaZ, meta