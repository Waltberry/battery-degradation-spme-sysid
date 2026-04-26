# battery_deg_spme/config/defaults.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    slurm_cpus_env_var: str = "SLURM_CPUS_PER_TASK"
    default_num_threads: int = 8
    jax_enable_x64: bool = True


@dataclass
class DataConfig:
    mpr_path: str = "12to1-25%CNC-3%GQDs _C01.mpr"
    time_col: str = "time/s"
    i_col: str = "I/mA"
    v_col: str = "Ewe/V"

    force_units: str = "A"          # "auto" | "mA" | "A"
    v_ref: str = "none"             # "none" | "first" | "mean"
    resample: bool = True

    enforce_discharge_only: bool = True
    raw_discharge_sign: str = "negative"

    tmax: float = -1.0
    drop_first_n: int = 0


@dataclass
class CycleConfig:
    cycle_mode: str = "random"      # "index" | "random"
    cycle_index: int = 0
    random_seed: Optional[int] = 42

    use_min_cycle_len: bool = False
    min_cycle_len: int = 50

    include_previous_segment: bool = False
    n_prev_points: int = 10


@dataclass
class SurrogateConfig:
    poly_deg: int = 17
    degree_sweep: list[int] = field(default_factory=lambda: list(range(1, 31)))
    use_ln_feature: bool = False

    use_fast_transient_basis: bool = False
    transient_tau_list: list[float] = field(default_factory=list)

    clip_raw_a: float = 4.0
    clip_raw_b: float = 2.0
    clip_raw_z: float = 50.0

    nonlinearity_grid_n: int = 80
    nonlinearity_guard: float = 1e-4


@dataclass
class RCConfig:
    use_rc: bool = False
    r1_init_ohm: float = 0.05
    tau_init_s: float = 20.0


@dataclass
class OptimizationConfig:
    adam_epochs_stage2: int = 1200
    adam_epochs_stage3a: int = 1500
    adam_epochs_stage3b: int = 2000

    use_lbfgs: bool = True
    lbfgs_epochs: int = 300

    adam_eta_stage2: float = 5e-4
    adam_eta_stage3a: float = 5e-4
    adam_eta_stage3b: float = 2e-4

    rho_th_stage2: float = 1e-8
    rho_th_stage3a: float = 1e-8
    rho_th_stage3b: float = 1e-8


@dataclass
class SolverConfig:
    solver_rtol: float = 1e-5
    solver_atol: float = 1e-8
    dt0_div: float = 5.0
    max_steps: int = 10_000_000


@dataclass
class SanityConfig:
    min_med_abs_i_a: float = 5e-3
    min_xp_range: float = 1e-4
    min_xn_range: float = 1e-4


@dataclass
class ProxyInitialStateConfig:
    xn0: float = 0.60
    xp0: float = 0.60
    ce0_dev: float = 0.0


@dataclass
class ExperimentConfig:
    n_series_real: int = 1

    run_degree_sweep: bool = False
    run_stage3: bool = True
    run_stage3a: bool = True
    run_stage3b: bool = True

    run_all_cycles_generalization: bool = True
    run_nonlinearity_analysis: bool = True
    run_parameter_monitoring: bool = True

    save_figures: bool = True
    save_metrics: bool = True
    save_logs: bool = True

    final_stage_name: str = "stage3b"