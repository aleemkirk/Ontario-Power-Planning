"""Configuration dataclasses for model and solver settings."""

from dataclasses import dataclass
from .constants import *


@dataclass
class ModelConfig:
    """Configuration for PowerSystemOptimization model."""
    start_year: int = 2025
    end_year: int = 2045
    n_rep_days: int = DEFAULT_N_REP_DAYS
    use_soft_ramp_constraints: bool = True
    ramp_penalty: float = DEFAULT_RAMP_PENALTY
    use_lead_times: bool = True
    use_retirements: bool = False
    use_nuclear_policy: bool = False
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    reserve_margin: float = DEFAULT_RESERVE_MARGIN


@dataclass
class SolverConfig:
    """Configuration for solver."""
    solver_type: str = 'highs'
    time_limit: int = DEFAULT_TIME_LIMIT
    mip_gap: float = 0.01
    tee: bool = False


# Preset configurations
QUICK_TEST_CONFIG = ModelConfig(end_year=2030, n_rep_days=4)
FULL_RUN_CONFIG = ModelConfig(end_year=2045, n_rep_days=12)
NUCLEAR_POLICY_CONFIG = ModelConfig(use_nuclear_policy=True)
