# battery_deg_spme/models/ct_model_wrappers.py

from __future__ import annotations

import diffrax
from jax_sysid.models import CTModel


def build_ct_model(nx, nu, ny, state_fcn, output_fcn):
    return CTModel(nx, nu, ny, state_fcn=state_fcn, output_fcn=output_fcn)


def configure_loss(model, rho_th: float):
    try:
        model.loss(rho_x0=0.0, rho_th=float(rho_th), train_x0=False, xsat=1e9)
    except TypeError:
        model.loss(rho_x0=0.0, rho_th=float(rho_th))


def configure_optimizer(
    model,
    adam_epochs: int,
    adam_eta: float,
    params_min=None,
    params_max=None,
    lbfgs_epochs: int = 0,
):
    kwargs = {
        "adam_epochs": int(adam_epochs),
        "lbfgs_epochs": int(lbfgs_epochs),
        "adam_eta": float(adam_eta),
    }

    if params_min is not None:
        kwargs["params_min"] = params_min
    if params_max is not None:
        kwargs["params_max"] = params_max

    model.optimization(**kwargs)


def configure_integration(
    model,
    dt: float,
    dt0_div: float,
    max_steps: int,
    solver_rtol: float,
    solver_atol: float,
):
    dt0 = float(dt) / float(dt0_div)
    if dt0 <= 0:
        raise ValueError(f"Computed dt0 must be positive, got {dt0}.")

    model.integration_options(
        ode_solver=diffrax.Tsit5(),
        dt0=dt0,
        max_steps=int(max_steps),
        stepsize_controller=diffrax.PIDController(
            rtol=float(solver_rtol),
            atol=float(solver_atol),
        ),
    )