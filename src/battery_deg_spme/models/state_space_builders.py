# battery_deg_spme/models/state_space_builders.py
from __future__ import annotations

import numpy as np


def build_An(cfg) -> np.ndarray:
    s = cfg.Dn / (cfg.Rn ** 2)
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0], A[0, 1] = -24 * s, 24 * s
    A[1, 0], A[1, 1], A[1, 2] = 16 * s, -40 * s, 24 * s
    A[2, 1], A[2, 2], A[2, 3] = 16 * s, -40 * s, 24 * s
    A[3, 2], A[3, 3] = 16 * s, -16 * s
    return A


def build_Bn(cfg) -> np.ndarray:
    sign = -1.0 if cfg.discharge_positive else +1.0
    b = np.zeros((4, 1), dtype=np.float64)
    b[-1, 0] = sign * (6.0 / cfg.Rn) * (1.0 / (cfg.F * cfg.a_s_n * cfg.A * cfg.L1))
    return b


def build_Ap(cfg) -> np.ndarray:
    s = cfg.Dp / (cfg.Rp ** 2)
    A = np.zeros((4, 4), dtype=np.float64)
    A[0, 0], A[0, 1] = -24 * s, 24 * s
    A[1, 0], A[1, 1], A[1, 2] = 16 * s, -40 * s, 24 * s
    A[2, 1], A[2, 2], A[2, 3] = 16 * s, -40 * s, 24 * s
    A[3, 2], A[3, 3] = 16 * s, -16 * s
    return A


def build_Bp(cfg) -> np.ndarray:
    sign = +1.0 if cfg.discharge_positive else -1.0
    b = np.zeros((4, 1), dtype=np.float64)
    b[-1, 0] = sign * (6.0 / cfg.Rp) * (1.0 / (cfg.F * cfg.a_s_p * cfg.A * cfg.L3))
    return b


def build_Ae(cfg) -> np.ndarray:
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


def build_Be(cfg) -> np.ndarray:
    b = np.zeros((6, 1), dtype=np.float64)

    sign_left = -1.0 if cfg.discharge_positive else +1.0
    sign_right = +1.0 if cfg.discharge_positive else -1.0

    s1 = sign_left * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L1 * cfg.eps)
    s3 = sign_right * (1.0 - cfg.t_plus) / (cfg.F * cfg.A * cfg.L3 * cfg.eps)

    b[0, 0] = s1
    b[1, 0] = s1
    b[4, 0] = s3
    b[5, 0] = s3

    return b