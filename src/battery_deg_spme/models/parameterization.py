from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def softplus(x):
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


def pos(x, floor=1e-12):
    return softplus(x) + floor


def softplus_inv(y):
    y = float(max(y, 1e-12))
    if y > 20:
        return y
    return np.log(np.expm1(y))


def raw_from_pos(val, floor=1e-12):
    val = float(max(val, floor + 1e-12))
    return softplus_inv(val - floor)


def thetaA_nom_from_cfg(cfg) -> np.ndarray:
    th1 = cfg.Dn / (cfg.Rn ** 2)
    th2 = cfg.Dp / (cfg.Rp ** 2)

    K = cfg.De / cfg.eps
    th3 = K / (cfg.L1 ** 2)
    th4 = K / ((cfg.L1 + cfg.L2) ** 2)
    th5 = K / (cfg.L2 ** 2)
    th6 = K / ((cfg.L2 + cfg.L3) ** 2)
    th7 = K / (cfg.L3 ** 2)

    return np.array([th1, th2, th3, th4, th5, th6, th7], dtype=np.float64)


def thetaB_nom_from_cfg(cfg) -> np.ndarray:
    sign_n = -1.0 if cfg.discharge_positive else +1.0
    sign_p = +1.0 if cfg.discharge_positive else -1.0

    th8 = sign_n * (1.0 / cfg.Rn) * (1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1))
    th9 = sign_p * (1.0 / cfg.Rp) * (1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3))

    sign_left = -1.0 if cfg.discharge_positive else +1.0
    sign_right = +1.0 if cfg.discharge_positive else -1.0
    th10 = sign_left * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L1 * cfg.eps)
    th11 = sign_right * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L3 * cfg.eps)

    return np.array([th8, th9, th10, th11], dtype=np.float64)


def make_builders(dtype=jnp.float64):
    @jax.jit
    def build_A_from_thetaA(thetaA: jnp.ndarray) -> jnp.ndarray:
        th1, th2, th3, th4, th5, th6, th7 = thetaA

        An = jnp.array(
            [
                [-24 * th1, 24 * th1, 0.0, 0.0],
                [16 * th1, -40 * th1, 24 * th1, 0.0],
                [0.0, 16 * th1, -40 * th1, 24 * th1],
                [0.0, 0.0, 16 * th1, -16 * th1],
            ],
            dtype=dtype,
        )

        Ap = jnp.array(
            [
                [-24 * th2, 24 * th2, 0.0, 0.0],
                [16 * th2, -40 * th2, 24 * th2, 0.0],
                [0.0, 16 * th2, -40 * th2, 24 * th2],
                [0.0, 0.0, 16 * th2, -16 * th2],
            ],
            dtype=dtype,
        )

        Ae = jnp.array(
            [
                [-4 * th3, 4 * th3, 0.0, 0.0, 0.0, 0.0],
                [4 * th3, -(4 * th3 + 16 * th4), 16 * th4, 0.0, 0.0, 0.0],
                [0.0, 16 * th4, -(16 * th4 + 4 * th5), 4 * th5, 0.0, 0.0],
                [0.0, 0.0, 4 * th5, -(4 * th5 + 16 * th6), 16 * th6, 0.0],
                [0.0, 0.0, 0.0, 16 * th6, -(16 * th6 + 4 * th7), 4 * th7],
                [0.0, 0.0, 0.0, 0.0, 4 * th7, -4 * th7],
            ],
            dtype=dtype,
        )

        A = jnp.zeros((14, 14), dtype=dtype)
        A = A.at[0:4, 0:4].set(An)
        A = A.at[4:8, 4:8].set(Ap)
        A = A.at[8:14, 8:14].set(Ae)
        return A

    @jax.jit
    def build_B_from_thetaB(thetaB: jnp.ndarray) -> jnp.ndarray:
        th8, th9, th10, th11 = thetaB
        B = jnp.zeros((14, 1), dtype=dtype)
        B = B.at[3, 0].set(6.0 * th8)
        B = B.at[7, 0].set(6.0 * th9)
        B = B.at[8, 0].set(th10)
        B = B.at[9, 0].set(th10)
        B = B.at[12, 0].set(th11)
        B = B.at[13, 0].set(th11)
        return B

    return build_A_from_thetaA, build_B_from_thetaB