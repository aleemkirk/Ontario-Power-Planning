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

    For annual model (prototype): Σ_i p[t,i] ≥ AnnualDemand[t]  ∀t

    Args:
        model: Pyomo model instance
    """
    def demand_rule(m, t):
        return sum(m.p[t, i] for i in m.plant_types) >= m.annual_demand[t]

    model.demand_constraint = pyo.Constraint(model.years, rule=demand_rule)


def capacity_constraint(model):
    """
    Ensure generation doesn't exceed available capacity.

    For annual model: p[t,i] ≤ N[t,i] × CapacityFactor[i] × 8760  ∀t,i
    (8760 = hours per year, converts MW capacity to MWh annual energy)

    Args:
        model: Pyomo model instance
    """
    def capacity_rule(m, t, i):
        return m.p[t, i] <= m.N[t, i] * m.capacity_factor[i] * 8760

    model.capacity_constraint = pyo.Constraint(
        model.years, model.plant_types, rule=capacity_rule
    )


def reserve_margin_constraint(model):
    """
    Maintain 15% reserve margin above peak demand.

    Σ_i N[t,i] ≥ (1 + ReserveMargin) × PeakDemand[t]  ∀t

    Args:
        model: Pyomo model instance
    """
    def reserve_rule(m, t):
        total_capacity = sum(m.N[t, i] for i in m.plant_types)
        required_capacity = (1 + m.reserve_margin) * m.peak_demand[t]
        return total_capacity >= required_capacity

    model.reserve_margin_constraint = pyo.Constraint(model.years, rule=reserve_rule)


def ramp_rate_constraint(model):
    """
    Limit rate of change in power output.

    NOTE: Skipped in prototype (annual resolution).
    Will be implemented in Phase 3 with hourly/monthly resolution.

    |p[t,i,h] - p[t,i,h-1]| ≤ RampRate[i] × N[t,i]  ∀t,i,h

    Args:
        model: Pyomo model instance
    """
    # Skip for prototype - only relevant for sub-annual time resolution
    pass


def capacity_dynamics_constraint(model):
    """
    Track capacity evolution over time.

    Simplified for prototype (no lead times):
    - N[t,i] = N[t-1,i] + x[t,i]  ∀t>start_year, ∀i
    - N[start_year,i] = InitialCapacity[i]  ∀i

    Full version (Phase 3) will include:
    - Construction lead times
    - Plant retirements after lifespan

    Args:
        model: Pyomo model instance
    """
    def initial_capacity_rule(m, i):
        """Set initial capacity for first year."""
        return m.N[m.start_year, i] == m.initial_capacity[i]

    def capacity_evolution_rule(m, t, i):
        """Track capacity evolution year-by-year."""
        if t == m.start_year:
            return pyo.Constraint.Skip
        prev_year = t - 1
        return m.N[t, i] == m.N[prev_year, i] + m.x[t, i]

    model.initial_capacity_constraint = pyo.Constraint(
        model.plant_types, rule=initial_capacity_rule
    )

    model.capacity_evolution_constraint = pyo.Constraint(
        model.years, model.plant_types, rule=capacity_evolution_rule
    )


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
