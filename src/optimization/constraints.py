"""
Constraint definitions for the optimization model.

Contains:
- Demand satisfaction constraints
- Capacity constraints
- Reserve margin constraints
- Ramp rate constraints
- Capacity dynamics (construction lead times, retirements)
"""

import pyomo.environ as pyo


def demand_satisfaction_constraint(model):
    """
    Ensure total generation meets demand at all times.

    Σ_i p[t,i,h] ≥ Demand[t,h]  ∀t,h
    """
    # TODO: Implement demand constraint
    pass


def capacity_constraint(model):
    """
    Ensure generation doesn't exceed available capacity.

    p[t,i,h] ≤ N[t,i] × CapacityFactor[i]  ∀t,i,h
    """
    # TODO: Implement capacity constraint
    pass


def reserve_margin_constraint(model):
    """
    Maintain 15% reserve margin above peak demand.

    Σ_i N[t,i] ≥ (1 + ReserveMargin) × PeakDemand[t]  ∀t
    """
    # TODO: Implement reserve margin constraint
    pass


def ramp_rate_constraint(model):
    """
    Limit rate of change in power output.

    |p[t,i,h] - p[t,i,h-1]| ≤ RampRate[i] × N[t,i]  ∀t,i,h
    """
    # TODO: Implement ramp rate constraint
    pass


def capacity_dynamics_constraint(model):
    """
    Track capacity evolution over time.

    Accounts for:
    - New plant construction (with lead times)
    - Plant retirements (after lifespan)

    N[t,i] = N[t-1,i] + NewCapacity[t,i] - Retirements[t,i]  ∀t,i
    """
    # TODO: Implement capacity dynamics
    pass


def add_all_constraints(model):
    """
    Add all constraints to the model.

    Args:
        model: Pyomo ConcreteModel instance
    """
    demand_satisfaction_constraint(model)
    capacity_constraint(model)
    reserve_margin_constraint(model)
    ramp_rate_constraint(model)
    capacity_dynamics_constraint(model)
