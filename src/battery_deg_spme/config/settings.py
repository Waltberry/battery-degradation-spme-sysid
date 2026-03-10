from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .defaults import (
    CycleConfig,
    DataConfig,
    ExperimentConfig,
    OptimizationConfig,
    ProxyInitialStateConfig,
    RCConfig,
    RuntimeConfig,
    SanityConfig,
    SolverConfig,
    SurrogateConfig,
)


@dataclass
class Settings:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    rc: RCConfig = field(default_factory=RCConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    sanity: SanityConfig = field(default_factory=SanityConfig)
    initial_state: ProxyInitialStateConfig = field(default_factory=ProxyInitialStateConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def get_default_settings() -> Settings:
    return Settings()